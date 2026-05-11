"""
실시간 CES 데이터 스트리밍 WebSocket + SSE 엔드포인트.

데이터 흐름:
  모바일 앱/게이트웨이 → WS POST /ws/stream/{user_id}
                           → 앙상블 추론
                           → 낙상 시 즉각 알림 브로드캐스트
  보호자 대시보드   ← SSE GET /api/v1/realtime/stream/{user_id}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

# InfluxDB + FCM 통합 (미설정 시 graceful degradation)
try:
    from src.pipeline.influx_writer import async_write_gait_result, get_influx_writer
    _INFLUX_AVAILABLE = True
except ImportError:
    _INFLUX_AVAILABLE = False

try:
    from src.notifications.fcm_push import notify_if_needed
    _FCM_AVAILABLE = True
except ImportError:
    _FCM_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Realtime"])

# ─── 연결 관리 ────────────────────────────────────────────────────────────────

class ConnectionManager:
    """
    user_id → SSE 구독자 큐 목록 관리.
    낙상 이벤트 발생 시 해당 user_id의 모든 보호자 화면에 즉각 브로드캐스트.
    """

    def __init__(self) -> None:
        # user_id → list of asyncio.Queue (각 보호자 연결당 1개)
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)
        # 최근 이벤트 버퍼 (재연결 시 최근 20개 재전송)
        self._recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

    def subscribe(self, user_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._queues[user_id].append(q)
        # 재연결 시 최근 이벤트 즉시 재전송
        for event in self._recent[user_id]:
            q.put_nowait(event)
        return q

    def unsubscribe(self, user_id: str, q: asyncio.Queue) -> None:
        if user_id in self._queues:
            try:
                self._queues[user_id].remove(q)
            except ValueError:
                pass

    async def broadcast(self, user_id: str, event: dict) -> None:
        """user_id 구독자 전체에 이벤트 전송"""
        self._recent[user_id].append(event)
        for q in list(self._queues.get(user_id, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("SSE 큐 가득 참: user_id=%s", user_id)

    def active_subscribers(self, user_id: str) -> int:
        return len(self._queues.get(user_id, []))


manager = ConnectionManager()

# ─── 앙상블 모델 로드 (싱글톤) ───────────────────────────────────────────────

_ensemble_model: Any = None
_model_lock = asyncio.Lock()


async def get_ensemble_model():
    global _ensemble_model
    if _ensemble_model is not None:
        return _ensemble_model
    async with _model_lock:
        if _ensemble_model is not None:
            return _ensemble_model
        try:
            import os
            from pathlib import Path
            from src.models.ces_ensemble import ShoeallsEnsembleModel

            ckpt_path = os.getenv("CES_MODEL_CHECKPOINT", "outputs/ces_ensemble.pt")
            model = ShoeallsEnsembleModel(input_dim=6)

            if Path(ckpt_path).exists():
                ckpt = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                logger.info("앙상블 모델 로드: %s", ckpt_path)
            else:
                logger.warning("체크포인트 없음 — 미학습 모델로 데모 동작: %s", ckpt_path)

            model.eval()
            _ensemble_model = model
        except Exception as e:
            logger.error("앙상블 모델 로드 실패: %s", e)
            _ensemble_model = None
    return _ensemble_model


def _run_inference(model: Any, x: torch.Tensor, ev_enc: np.ndarray, speed_arr: np.ndarray,
                   tilt_arr: np.ndarray, is_fall_rule: bool) -> dict:
    """
    ShoeallsEnsembleModel.predict_status() → websocket_stream 공통 result dict 변환.
    모델 인터페이스 변경 시 이 함수만 수정하면 됩니다.
    """
    from src.models.ces_ensemble import RuleBasedScorer
    # 임상 규칙 점수 → rule_features [batch, 1]
    scorer = RuleBasedScorer()
    rule_score, rule_labels = scorer.score(speed_arr, tilt_arr, ev_enc)
    rule_tensor = torch.tensor([[rule_score]], dtype=torch.float32)

    with torch.no_grad():
        is_anomaly, score_t, details = model.predict_status(x, rule_tensor, threshold=0.65)

    anomaly_score = float(score_t[0, 0].item())
    lstm_probs = details["lstm_probs"][0].cpu().tolist()
    label_names = ["normal", "waiting", "drag"]
    label_idx = int(max(range(len(lstm_probs)), key=lambda i: lstm_probs[i]))

    fall_alert = is_fall_rule or anomaly_score >= 0.80

    if fall_alert:
        recommendation = "낙상 이벤트 감지 — 보호자 즉각 확인 및 응급 의료 연락 권고"
    elif anomaly_score >= 0.65:
        recommendation = "보행 이상 징후 감지 — 전문 의료기관 방문을 권고합니다"
    elif anomaly_score >= 0.40:
        recommendation = "보행 변화 관찰 중 — 추세 모니터링 및 보호자 공유 권고"
    else:
        recommendation = "현재 보행 패턴은 안정적인 범주입니다"

    soft_labels = list(rule_labels)
    if not soft_labels and anomaly_score < 0.40:
        soft_labels = ["정상 보행 범주"]

    return {
        "anomaly_score": round(anomaly_score, 4),
        "gait_label": label_names[label_idx] if label_idx < len(label_names) else "normal",
        "gait_probabilities": {label_names[i]: float(lstm_probs[i]) for i in range(len(label_names))},
        "soft_labels": soft_labels,
        "fall_alert": fall_alert,
        "ae_reconstruction_loss": 0.0,
        "rule_score": rule_score,
        "lstm_risk_score": anomaly_score,
        "inference_ms": 0.0,
        "recommendation": recommendation,
        "meta": {"is_rule_triggered": is_fall_rule},
    }

# ─── Pydantic 스키마 ──────────────────────────────────────────────────────────

class CESFrame(BaseModel):
    """단일 CES 데이터 프레임 (fe_raw_data JSON 파싱 결과)"""
    foot_type: str = "left"
    label: str = "normal"
    speed: float = 0.0
    tilt: float = 0.0
    confidence: int = 80
    probabilities: list[float] = Field(default_factory=lambda: [1/3, 1/3, 1/3])

    @field_validator("speed")
    @classmethod
    def clamp_speed(cls, v: float) -> float:
        return max(0.0, min(5.0, v))


class CESEventFrame(BaseModel):
    """WebSocket으로 수신하는 단일 이벤트 프레임"""
    fe_event_type: str
    fe_event_value: float = 0.0
    fe_raw_data: CESFrame
    fe_timestamp: str | None = None


class CESStreamPayload(BaseModel):
    """WebSocket 수신 페이로드 — 배치 프레임"""
    user_id: str
    device_id: str
    session_id: str | None = None
    frames: list[CESEventFrame]

    @field_validator("frames")
    @classmethod
    def validate_frames(cls, v: list) -> list:
        if not v:
            raise ValueError("frames 리스트가 비어 있습니다")
        if len(v) > 200:
            raise ValueError("한 번에 최대 200 프레임까지 허용")
        return v


class IngestResponse(BaseModel):
    session_id: str
    anomaly_score: float
    gait_label: str
    soft_labels: list[str]
    fall_alert: bool
    recommendation: str
    inference_ms: float
    subscribers_notified: int
    is_demo: bool = False

# ─── 특성 추출 ────────────────────────────────────────────────────────────────

_EVENT_ENC = {"acceleration": 0, "shuffling": 1, "fall": 2}
_FOOT_ENC = {"left": 0, "right": 1}
_LABEL_ENC = {"normal": 0, "waiting": 1, "drag": 2}


def frames_to_tensor(frames: list[CESEventFrame]) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    CES 프레임 배열 → 모델 입력 텐서 변환.

    Returns:
        x:            [1, seq, 6] float32 텐서
        event_enc:    numpy 이벤트 타입 배열
        speed_arr:    numpy 속도 배열
        tilt_arr:     numpy 기울기 배열
        is_fall_rule: 윈도우 내 fall 이벤트 존재 여부
    """
    rows = []
    event_enc_list = []
    speed_list = []
    tilt_list = []
    is_fall_rule = False

    for f in frames:
        ev_enc = _EVENT_ENC.get(f.fe_event_type, 0)
        foot_enc = _FOOT_ENC.get(f.fe_raw_data.foot_type, 0)
        ev_val_norm = min(abs(f.fe_event_value) / 100.0, 1.0)
        conf_norm = min(f.fe_raw_data.confidence / 100.0, 1.0)

        row = [
            f.fe_raw_data.speed,         # speed
            f.fe_raw_data.tilt,          # tilt
            ev_val_norm,                  # fe_event_value_norm
            float(ev_enc),               # event_type_enc
            float(foot_enc),             # foot_type_enc
            conf_norm,                   # confidence_norm
        ]
        rows.append(row)
        event_enc_list.append(ev_enc)
        speed_list.append(f.fe_raw_data.speed)
        tilt_list.append(f.fe_raw_data.tilt)

        if f.fe_event_type == "fall":
            is_fall_rule = True

    x = torch.tensor(rows, dtype=torch.float32).unsqueeze(0)  # [1, seq, 6]
    return (
        x,
        np.array(event_enc_list),
        np.array(speed_list),
        np.array(tilt_list),
        is_fall_rule,
    )


# ─── WebSocket 엔드포인트 ─────────────────────────────────────────────────────

@router.websocket("/ws/stream/{user_id}")
async def websocket_stream(websocket: WebSocket, user_id: str):
    """
    실시간 CES 데이터 수신 WebSocket.

    클라이언트(게이트웨이/모바일 앱)는 CESStreamPayload JSON을 전송.
    서버는 추론 결과 + 낙상 알림을 즉각 반환.
    보호자 SSE 구독자에게도 동시 브로드캐스트.

    연결 예시:
        ws = new WebSocket("ws://localhost:8000/api/v1/ws/stream/user123")
        ws.send(JSON.stringify({ user_id: "user123", device_id: "SHOE_L", frames: [...] }))
    """
    await websocket.accept()
    logger.info("WebSocket 연결: user_id=%s", user_id)

    try:
        model = await get_ensemble_model()

        async for raw_msg in websocket.iter_text():
            t_recv = time.perf_counter()
            try:
                payload = CESStreamPayload.model_validate_json(raw_msg)
            except Exception as e:
                await websocket.send_json({
                    "error": f"페이로드 파싱 실패: {e}",
                    "code": "PARSE_ERROR",
                })
                continue

            session_id = payload.session_id or str(uuid.uuid4())

            # 특성 추출
            x, ev_enc, speed_arr, tilt_arr, is_fall_rule = frames_to_tensor(payload.frames)

            # 앙상블 추론
            is_demo = model is None
            result_dict: dict = {}
            if model is not None:
                try:
                    result_dict = _run_inference(model, x, ev_enc, speed_arr, tilt_arr, is_fall_rule)
                except Exception as e:
                    logger.exception("앙상블 추론 오류: %s", e)
                    result_dict = _mock_result(is_fall_rule)
                    is_demo = True
            else:
                result_dict = _mock_result(is_fall_rule)
                is_demo = True

            total_ms = (time.perf_counter() - t_recv) * 1000

            response = {
                "session_id": session_id,
                "user_id": user_id,
                "device_id": payload.device_id,
                "anomaly_score": result_dict["anomaly_score"],
                "gait_label": result_dict["gait_label"],
                "soft_labels": result_dict["soft_labels"],
                "fall_alert": result_dict["fall_alert"],
                "recommendation": result_dict["recommendation"],
                "inference_ms": result_dict["inference_ms"],
                "total_ms": round(total_ms, 2),
                "is_demo": is_demo,
                "subscribers_notified": manager.active_subscribers(user_id),
            }

            # WebSocket 클라이언트에게 즉각 반환
            await websocket.send_json(response)

            # SSE 보호자 화면에 브로드캐스트
            sse_event_type = "fall_alert" if result_dict["fall_alert"] else "gait_update"
            await manager.broadcast(user_id, {
                "event": sse_event_type,
                "data": response,
                "timestamp": time.time(),
            })

            if result_dict["fall_alert"]:
                logger.warning(
                    "낙상 알림 발송: user_id=%s session=%s score=%.3f",
                    user_id, session_id, result_dict["anomaly_score"],
                )

            # InfluxDB 비동기 저장 (백그라운드 — 응답 지연 없음)
            if _INFLUX_AVAILABLE and not is_demo and result_dict:
                asyncio.create_task(_write_to_influx(
                    user_id, payload.device_id, session_id,
                    result_dict,
                    float(speed_arr.mean()) if len(speed_arr) else 0.0,
                    float(tilt_arr.mean()) if len(tilt_arr) else 0.0,
                ))

            # FCM 푸시 알림 (낙상 또는 고위험 시)
            if _FCM_AVAILABLE and not is_demo and result_dict:
                asyncio.create_task(_send_fcm(user_id, session_id, result_dict))

    except WebSocketDisconnect:
        logger.info("WebSocket 종료: user_id=%s", user_id)
    except Exception as e:
        logger.exception("WebSocket 오류: user_id=%s error=%s", user_id, e)
        try:
            await websocket.send_json({"error": str(e), "code": "INTERNAL_ERROR"})
        except Exception:
            pass


# ─── SSE 보호자 스트림 ────────────────────────────────────────────────────────

@router.get(
    "/realtime/stream/{user_id}",
    summary="보호자 실시간 모니터링 (SSE)",
    description=(
        "Server-Sent Events 스트림. 보호자 대시보드에서 사용자의 실시간 보행 상태를 수신.\n\n"
        "이벤트 타입: `gait_update` | `fall_alert` | `heartbeat` | `session_end`\n\n"
        "낙상 이벤트는 1초 내 전달을 목표로 합니다."
    ),
)
async def sse_guardian_stream(user_id: str):
    """
    보호자용 SSE 스트림.

    사용 예시 (JavaScript):
        const evtSource = new EventSource("/api/v1/realtime/stream/user123")
        evtSource.addEventListener("fall_alert", e => alert(JSON.parse(e.data).recommendation))
        evtSource.addEventListener("gait_update", e => updateDashboard(JSON.parse(e.data)))
    """
    q = manager.subscribe(user_id)
    logger.info("SSE 구독 시작: user_id=%s", user_id)

    async def event_generator():
        try:
            # 초기 연결 확인 이벤트
            yield _sse_format("connected", {
                "user_id": user_id,
                "message": "실시간 보행 모니터링 시작",
                "timestamp": time.time(),
            })

            heartbeat_task = asyncio.create_task(_heartbeat(user_id))

            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield _sse_format(event["event"], event["data"])
                except asyncio.TimeoutError:
                    # 30초 타임아웃 → 연결 유지용 heartbeat
                    yield _sse_format("heartbeat", {"timestamp": time.time()})
        except asyncio.CancelledError:
            pass
        finally:
            heartbeat_task.cancel()
            manager.unsubscribe(user_id, q)
            logger.info("SSE 구독 종료: user_id=%s", user_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _heartbeat(user_id: str, interval: float = 5.0) -> None:
    """5초마다 heartbeat 이벤트를 보호자 큐에 주입"""
    while True:
        await asyncio.sleep(interval)
        await manager.broadcast(user_id, {
            "event": "heartbeat",
            "data": {"timestamp": time.time()},
            "timestamp": time.time(),
        })


def _sse_format(event: str, data: Any) -> str:
    """SSE 형식 문자열 생성"""
    json_data = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {json_data}\n\n"


# ─── 데이터 수집 REST 엔드포인트 ────────────────────────────────────────────────

@router.post(
    "/realtime/ingest",
    response_model=IngestResponse,
    summary="CES 배치 데이터 수집 및 분석",
    description=(
        "모바일 앱이나 BLE 게이트웨이가 수집한 CES 배치 프레임을 분석합니다.\n\n"
        "WebSocket 연결이 어려운 환경에서 대안으로 사용. 최대 200 프레임.\n\n"
        "`fall_alert=true` 반환 시 즉각 응급 절차를 시작하세요."
    ),
)
async def ingest_ces_batch(payload: CESStreamPayload) -> IngestResponse:
    """
    배치 CES 데이터 분석 REST 엔드포인트.
    낙상 이벤트 감지 시 SSE 보호자 화면에도 동시 브로드캐스트.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    x, ev_enc, speed_arr, tilt_arr, is_fall_rule = frames_to_tensor(payload.frames)

    model = await get_ensemble_model()
    is_demo = model is None
    result: dict = {}

    if model is not None:
        try:
            result = _run_inference(model, x, ev_enc, speed_arr, tilt_arr, is_fall_rule)
        except Exception as e:
            logger.exception("ingest 추론 오류: %s", e)
            result = _mock_result(is_fall_rule)
            is_demo = True
    else:
        result = _mock_result(is_fall_rule)

    subs = manager.active_subscribers(payload.user_id)

    # SSE 브로드캐스트
    if result.get("fall_alert"):
        await manager.broadcast(payload.user_id, {
            "event": "fall_alert",
            "data": {**result, "session_id": session_id, "user_id": payload.user_id},
            "timestamp": time.time(),
        })

    # InfluxDB 저장 (백그라운드)
    if _INFLUX_AVAILABLE and not is_demo:
        asyncio.create_task(_write_to_influx(
            payload.user_id, payload.device_id, session_id, result,
            float(speed_arr.mean()) if len(speed_arr) else 0.0,
            float(tilt_arr.mean()) if len(tilt_arr) else 0.0,
        ))

    # FCM 알림 (백그라운드)
    if _FCM_AVAILABLE and not is_demo:
        asyncio.create_task(_send_fcm(payload.user_id, session_id, result))

    return IngestResponse(
        session_id=session_id,
        anomaly_score=result.get("anomaly_score", 0.0),
        gait_label=result.get("gait_label", "normal"),
        soft_labels=result.get("soft_labels", []),
        fall_alert=result.get("fall_alert", False),
        recommendation=result.get("recommendation", ""),
        inference_ms=result.get("inference_ms", 0.0),
        subscribers_notified=subs,
        is_demo=is_demo,
    )


# ─── 데모용 mock 결과 ────────────────────────────────────────────────────────

async def _write_to_influx(
    user_id: str, device_id: str, session_id: str,
    result: dict, speed_mean: float, tilt_mean: float,
) -> None:
    """result dict → InfluxDB 호환 객체로 변환 후 쓰기"""
    from types import SimpleNamespace
    proxy = SimpleNamespace(
        anomaly_score=result.get("anomaly_score", 0.0),
        fall_alert=result.get("fall_alert", False),
        ae_reconstruction_loss=result.get("ae_reconstruction_loss", 0.0),
        rule_score=result.get("rule_score", 0.0),
        lstm_risk_score=result.get("lstm_risk_score", 0.0),
        gait_label=result.get("gait_label", "normal"),
        inference_ms=result.get("inference_ms", 0.0),
    )
    await async_write_gait_result(user_id, device_id, session_id, proxy, speed_mean, tilt_mean)


async def _send_fcm(user_id: str, session_id: str, result: dict) -> None:
    """result dict → FCM notify_if_needed 호환 객체로 변환 후 전송"""
    from types import SimpleNamespace
    proxy = SimpleNamespace(
        anomaly_score=result.get("anomaly_score", 0.0),
        fall_alert=result.get("fall_alert", False),
        recommendation=result.get("recommendation", ""),
        soft_labels=result.get("soft_labels", []),
        gait_label=result.get("gait_label", "normal"),
    )
    await notify_if_needed(user_id, session_id, proxy)


def _mock_result(is_fall: bool) -> dict:
    """모델 미로드 시 안전한 기본값 반환"""
    return {
        "anomaly_score": 1.0 if is_fall else 0.1,
        "gait_label": "normal",
        "gait_probabilities": {"normal": 0.8, "waiting": 0.1, "drag": 0.1},
        "soft_labels": ["낙상 이벤트 감지 — 즉각 알림"] if is_fall else ["정상 보행 범주 (데모)"],
        "fall_alert": is_fall,
        "ae_reconstruction_loss": 0.0,
        "rule_score": 0.0,
        "lstm_risk_score": 0.0,
        "inference_ms": 0.0,
        "recommendation": (
            "낙상 이벤트 감지 — 즉각 응급 절차 권고" if is_fall
            else "데모 모드 — 모델 미탑재"
        ),
        "meta": {"is_demo": True},
    }
