"""
Kafka CES 컨슈머 — Kafka 토픽에서 CES 이벤트 프레임을 소비하여
앙상블 추론 파이프라인으로 전달하는 비동기 브리지.

데이터 흐름:
  Kafka("shoealls.ces.{left|right}") → KafkaCESConsumer → asyncio.Queue
  → websocket_stream.py 추론 루프 → InfluxDB + SSE 브로드캐스트

실행 (단독):
  python -m src.ingestion.kafka_ces_consumer

환경 변수:
  KAFKA_BOOTSTRAP_SERVERS  (기본: localhost:9092)
  KAFKA_GROUP_ID           (기본: shoealls-ces-consumer)
  KAFKA_TOPICS             (기본: shoealls.ces.left,shoealls.ces.right)
  KAFKA_AUTO_OFFSET_RESET  (기본: latest)
  KAFKA_POLL_TIMEOUT_SEC   (기본: 1.0)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ─── 설정 ────────────────────────────────────────────────────────────────────


@dataclass
class KafkaConsumerConfig:
    bootstrap_servers: str = field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    )
    group_id: str = field(
        default_factory=lambda: os.getenv("KAFKA_GROUP_ID", "shoealls-ces-consumer")
    )
    topics: list[str] = field(
        default_factory=lambda: os.getenv(
            "KAFKA_TOPICS", "shoealls.ces.left,shoealls.ces.right"
        ).split(",")
    )
    auto_offset_reset: str = field(
        default_factory=lambda: os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest")
    )
    poll_timeout_sec: float = field(
        default_factory=lambda: float(os.getenv("KAFKA_POLL_TIMEOUT_SEC", "1.0"))
    )
    max_queue_size: int = 2000
    session_gap_sec: float = 30.0  # 세션 경계 감지 기준 (30초 이상 공백)


# ─── CES Kafka 메시지 스키마 ──────────────────────────────────────────────────
# Kafka 메시지 value JSON 형식 (BLE 게이트웨이가 발행하는 형태):
# {
#   "user_id": "user123",
#   "device_id": "SHOE_L",
#   "session_id": "uuid-optional",
#   "fe_event_type": "acceleration"|"shuffling"|"fall",
#   "fe_event_value": 50.0,
#   "fe_raw_data": {
#     "foot_type": "left"|"right",
#     "label": "normal"|"waiting"|"drag",
#     "speed": 1.2,
#     "tilt": 0.05,
#     "confidence": 87,
#     "probabilities": [0.8, 0.1, 0.1]
#   },
#   "fe_timestamp": "2026-05-11 14:30:00"
# }


@dataclass
class KafkaCESFrame:
    user_id: str
    device_id: str
    session_id: str
    fe_event_type: str
    fe_event_value: float
    fe_raw_data: dict[str, Any]
    fe_timestamp: str
    kafka_topic: str
    kafka_partition: int
    kafka_offset: int


# ─── 세션 추적 ────────────────────────────────────────────────────────────────


class SessionTracker:
    """device_id별 세션 경계를 추적 (30초 이상 공백 → 새 세션)"""

    def __init__(self, gap_sec: float = 30.0) -> None:
        self._last_ts: dict[str, float] = {}
        self._sessions: dict[str, str] = {}
        self.gap_sec = gap_sec

    def get_session_id(self, device_id: str, provided_id: str | None = None) -> str:
        import uuid

        now = time.time()
        last = self._last_ts.get(device_id, 0.0)

        if provided_id:
            self._sessions[device_id] = provided_id
        elif now - last > self.gap_sec or device_id not in self._sessions:
            self._sessions[device_id] = str(uuid.uuid4())
            if device_id in self._last_ts:
                logger.info(
                    "새 세션 시작: device=%s gap=%.1fs session=%s",
                    device_id, now - last, self._sessions[device_id],
                )

        self._last_ts[device_id] = now
        return self._sessions[device_id]


# ─── Kafka 컨슈머 (스레드 기반) ───────────────────────────────────────────────


class KafkaCESConsumer:
    """
    confluent-kafka 기반 CES 이벤트 소비자.

    스레드에서 실행되어 메시지를 파싱한 뒤 asyncio.Queue에 적재.
    FastAPI lifespan에서 start() → shutdown() 패턴으로 사용.

    사용 예시:
        consumer = KafkaCESConsumer(cfg, loop=asyncio.get_event_loop())
        consumer.start()
        frame = await consumer.queue.get()
        ...
        consumer.shutdown()
    """

    def __init__(
        self,
        cfg: KafkaConsumerConfig | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        on_frame: Callable[[KafkaCESFrame], None] | None = None,
    ) -> None:
        self.cfg = cfg or KafkaConsumerConfig()
        self.loop = loop or asyncio.get_event_loop()
        self.on_frame = on_frame
        self.queue: asyncio.Queue[KafkaCESFrame] = asyncio.Queue(
            maxsize=self.cfg.max_queue_size
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._session_tracker = SessionTracker(self.cfg.session_gap_sec)

        # 통계
        self._recv_count = 0
        self._error_count = 0
        self._fall_count = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._consume_loop, daemon=True, name="kafka-ces-consumer")
        self._thread.start()
        logger.info(
            "KafkaCESConsumer 시작: servers=%s topics=%s group=%s",
            self.cfg.bootstrap_servers, self.cfg.topics, self.cfg.group_id,
        )

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        logger.info(
            "KafkaCESConsumer 종료: recv=%d errors=%d falls=%d",
            self._recv_count, self._error_count, self._fall_count,
        )

    def stats(self) -> dict[str, int]:
        return {
            "received": self._recv_count,
            "errors": self._error_count,
            "falls": self._fall_count,
            "queue_size": self.queue.qsize(),
        }

    # ── 내부 ─────────────────────────────────────────────────────────────────

    def _consume_loop(self) -> None:
        """실제 Kafka 소비 루프 (별도 스레드에서 실행)"""
        try:
            from confluent_kafka import Consumer, KafkaError, KafkaException
        except ImportError:
            logger.error(
                "confluent-kafka 미설치. pip install confluent-kafka 후 재시작하세요."
            )
            return

        consumer = Consumer({
            "bootstrap.servers": self.cfg.bootstrap_servers,
            "group.id": self.cfg.group_id,
            "auto.offset.reset": self.cfg.auto_offset_reset,
            "enable.auto.commit": False,  # 수동 커밋 (at-least-once)
            "max.poll.interval.ms": 60000,
        })
        consumer.subscribe(self.cfg.topics)

        try:
            while not self._stop_event.is_set():
                msg = consumer.poll(timeout=self.cfg.poll_timeout_sec)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        logger.error("Kafka 오류: %s", msg.error())
                        self._error_count += 1
                    continue

                frame = self._parse_message(msg)
                if frame is None:
                    consumer.commit(message=msg, asynchronous=True)
                    continue

                self._recv_count += 1
                if frame.fe_event_type == "fall":
                    self._fall_count += 1
                    logger.warning(
                        "낙상 이벤트 수신: user=%s device=%s",
                        frame.user_id, frame.device_id,
                    )

                # asyncio.Queue에 스레드 안전하게 삽입
                asyncio.run_coroutine_threadsafe(
                    self._put_frame(frame), self.loop
                ).result(timeout=2.0)

                if self.on_frame:
                    self.on_frame(frame)

                # 성공 후 오프셋 커밋 (at-least-once 보장)
                consumer.commit(message=msg, asynchronous=True)

                if self._recv_count % 1000 == 0:
                    logger.info("처리 통계: %s", self.stats())

        except KafkaException as e:
            logger.exception("Kafka 예외: %s", e)
        finally:
            consumer.close()
            logger.info("Kafka 컨슈머 종료됨")

    async def _put_frame(self, frame: KafkaCESFrame) -> None:
        try:
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            # 큐 가득 찬 경우: 낙상은 기존 항목을 밀어내고 삽입, 나머지는 드롭
            if frame.fe_event_type == "fall":
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                await self.queue.put(frame)
                logger.warning("낙상 이벤트 우선 처리: 큐 항목 1개 제거 후 삽입")
            else:
                logger.warning("큐 가득 참: 프레임 드롭 device=%s", frame.device_id)
                self._error_count += 1

    def _parse_message(self, msg: Any) -> KafkaCESFrame | None:
        """Kafka 메시지 → KafkaCESFrame 변환. 파싱 실패 시 None 반환"""
        try:
            raw = json.loads(msg.value().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug("JSON 파싱 실패: %s | raw=%.80s", e, msg.value())
            self._error_count += 1
            return None

        # 필수 필드 검증
        required = ("user_id", "device_id", "fe_event_type", "fe_raw_data")
        for key in required:
            if key not in raw:
                logger.debug("필수 필드 누락: %s | keys=%s", key, list(raw.keys()))
                self._error_count += 1
                return None

        # fe_event_type 검증
        valid_event_types = {"acceleration", "shuffling", "fall"}
        ev_type = str(raw["fe_event_type"])
        if ev_type not in valid_event_types:
            logger.debug("알 수 없는 이벤트 타입: %s", ev_type)
            ev_type = "acceleration"

        device_id = str(raw["device_id"])
        session_id = self._session_tracker.get_session_id(
            device_id, raw.get("session_id")
        )

        return KafkaCESFrame(
            user_id=str(raw["user_id"]),
            device_id=device_id,
            session_id=session_id,
            fe_event_type=ev_type,
            fe_event_value=float(raw.get("fe_event_value", 0.0)),
            fe_raw_data=raw.get("fe_raw_data", {}),
            fe_timestamp=str(raw.get("fe_timestamp", "")),
            kafka_topic=msg.topic(),
            kafka_partition=msg.partition(),
            kafka_offset=msg.offset(),
        )


# ─── FastAPI 통합 헬퍼 ────────────────────────────────────────────────────────

_consumer_instance: KafkaCESConsumer | None = None


def get_kafka_consumer() -> KafkaCESConsumer | None:
    """FastAPI 의존성 주입용 싱글톤 접근자"""
    return _consumer_instance


async def kafka_consumer_lifespan(loop: asyncio.AbstractEventLoop) -> KafkaCESConsumer:
    """
    FastAPI lifespan에서 호출 — Kafka 소비 시작.

    사용 예시 (api/main.py):
        @asynccontextmanager
        async def lifespan(app):
            consumer = await kafka_consumer_lifespan(asyncio.get_event_loop())
            yield
            consumer.shutdown()
    """
    global _consumer_instance
    cfg = KafkaConsumerConfig()
    _consumer_instance = KafkaCESConsumer(cfg, loop=loop)
    _consumer_instance.start()
    return _consumer_instance


# ─── Kafka → 추론 파이프라인 루프 ────────────────────────────────────────────


async def kafka_inference_loop(
    consumer: KafkaCESConsumer,
    batch_size: int = 30,
    batch_timeout_sec: float = 0.5,
) -> None:
    """
    Kafka 큐에서 프레임을 배치로 수집하여 앙상블 추론 + SSE 브로드캐스트.

    websocket_stream.py의 ingest_ces_batch와 동일한 파이프라인 재사용.
    FastAPI background task로 실행:
        asyncio.create_task(kafka_inference_loop(consumer))
    """
    from api.routers.websocket_stream import (
        CESEventFrame,
        CESFrame,
        CESStreamPayload,
        ingest_ces_batch,
    )

    # user_id별 배치 버퍼
    buffers: dict[str, list] = {}
    last_flush: dict[str, float] = {}

    logger.info("Kafka 추론 루프 시작 (batch=%d timeout=%.1fs)", batch_size, batch_timeout_sec)

    while True:
        try:
            frame: KafkaCESFrame = await asyncio.wait_for(
                consumer.queue.get(), timeout=batch_timeout_sec
            )

            uid = frame.user_id
            if uid not in buffers:
                buffers[uid] = []
                last_flush[uid] = time.time()

            # KafkaCESFrame → CESEventFrame 변환
            raw = frame.fe_raw_data
            ces_frame = CESEventFrame(
                fe_event_type=frame.fe_event_type,
                fe_event_value=frame.fe_event_value,
                fe_raw_data=CESFrame(
                    foot_type=raw.get("foot_type", "left"),
                    label=raw.get("label", "normal"),
                    speed=float(raw.get("speed", 0.0)),
                    tilt=float(raw.get("tilt", 0.0)),
                    confidence=int(raw.get("confidence", 80)),
                    probabilities=raw.get("probabilities", [1/3, 1/3, 1/3]),
                ),
                fe_timestamp=frame.fe_timestamp,
            )
            buffers[uid].append(ces_frame)

            # 낙상은 즉각 플러시 (배치 크기 무관)
            should_flush = (
                frame.fe_event_type == "fall"
                or len(buffers[uid]) >= batch_size
                or (time.time() - last_flush.get(uid, 0)) > batch_timeout_sec
            )

            if should_flush and buffers[uid]:
                payload = CESStreamPayload(
                    user_id=uid,
                    device_id=frame.device_id,
                    session_id=frame.session_id,
                    frames=buffers[uid],
                )
                asyncio.create_task(ingest_ces_batch(payload))
                buffers[uid] = []
                last_flush[uid] = time.time()

        except asyncio.TimeoutError:
            # 타임아웃 시 남은 버퍼 플러시
            for uid, frames in list(buffers.items()):
                if frames:
                    from api.routers.websocket_stream import CESStreamPayload, ingest_ces_batch
                    payload = CESStreamPayload(
                        user_id=uid,
                        device_id="kafka",
                        session_id=None,
                        frames=frames,
                    )
                    asyncio.create_task(ingest_ces_batch(payload))
                    buffers[uid] = []
        except asyncio.CancelledError:
            logger.info("Kafka 추론 루프 종료")
            break
        except Exception as e:
            logger.exception("추론 루프 오류: %s", e)


# ─── 단독 실행 (디버그용) ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = KafkaConsumerConfig()
    logger.info("설정: %s", cfg)

    loop = asyncio.new_event_loop()
    consumer = KafkaCESConsumer(cfg, loop=loop)
    consumer.start()

    def on_stop(sig, frame):
        consumer.shutdown()
        loop.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)

    async def print_frames():
        while True:
            frame = await consumer.queue.get()
            print(f"[{frame.fe_event_type.upper():12s}] user={frame.user_id} "
                  f"speed={frame.fe_raw_data.get('speed', '?')} "
                  f"session={frame.session_id[:8]}...")

    loop.run_until_complete(print_frames())
