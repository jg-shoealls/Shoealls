"""
InfluxDB 보행 데이터 적재 모듈.

EnsembleOutput → InfluxDB Point 변환 후 비동기 배치 쓰기.
시계열 추세 분석 (일/주/월)의 원본 데이터 저장소 역할.

환경 변수:
  INFLUXDB_URL     (기본: http://localhost:8086)
  INFLUXDB_TOKEN   (필수: InfluxDB API 토큰)
  INFLUXDB_ORG     (기본: shoealls)
  INFLUXDB_BUCKET  (기본: gait_data)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ─── 설정 ────────────────────────────────────────────────────────────────────


@dataclass
class InfluxConfig:
    url: str = ""
    token: str = ""
    org: str = ""
    bucket: str = ""
    batch_size: int = 200
    flush_interval_ms: int = 1000
    write_timeout_sec: int = 10

    def __post_init__(self) -> None:
        self.url = self.url or os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.token = self.token or os.getenv("INFLUXDB_TOKEN", "")
        self.org = self.org or os.getenv("INFLUXDB_ORG", "shoealls")
        self.bucket = self.bucket or os.getenv("INFLUXDB_BUCKET", "gait_data")

    @property
    def is_configured(self) -> bool:
        return bool(self.url and self.token and self.org and self.bucket)


# ─── InfluxDB 클라이언트 래퍼 ─────────────────────────────────────────────────


class GaitInfluxWriter:
    """
    보행 분석 결과를 InfluxDB에 비동기 배치 쓰기.

    measurement 구조:
      gait_sessions:
        tags:   user_id, device_id, session_id, gait_label, risk_band
        fields: anomaly_score, fall_alert, ae_loss, rule_score, lstm_risk,
                speed_mean, tilt_mean, inference_ms
        time:   나노초 타임스탬프

    미설치/미설정 시 WARNING 로그만 출력하고 정상 반환 (graceful degradation).
    """

    def __init__(self, cfg: InfluxConfig | None = None) -> None:
        self.cfg = cfg or InfluxConfig()
        self._client: Any = None
        self._write_api: Any = None
        self._available = False
        self._write_count = 0
        self._error_count = 0

    def connect(self) -> bool:
        """InfluxDB 연결 초기화. 실패 시 False 반환 (서버는 계속 동작)"""
        if not self.cfg.is_configured:
            logger.warning(
                "InfluxDB 미설정 (INFLUXDB_TOKEN 없음) — 보행 데이터 DB 저장 비활성화"
            )
            return False

        try:
            from influxdb_client import InfluxDBClient, WriteOptions
            from influxdb_client.client.write_api import ASYNCHRONOUS

            self._client = InfluxDBClient(
                url=self.cfg.url,
                token=self.cfg.token,
                org=self.cfg.org,
                timeout=self.cfg.write_timeout_sec * 1000,
            )
            self._write_api = self._client.write_api(
                write_options=WriteOptions(
                    batch_size=self.cfg.batch_size,
                    flush_interval=self.cfg.flush_interval_ms,
                )
            )
            # 연결 확인
            health = self._client.health()
            if health.status == "pass":
                self._available = True
                logger.info(
                    "InfluxDB 연결 성공: %s / %s / %s",
                    self.cfg.url, self.cfg.org, self.cfg.bucket,
                )
                return True
            else:
                logger.warning("InfluxDB 상태 이상: %s", health.status)
                return False

        except ImportError:
            logger.warning(
                "influxdb-client 미설치 (pip install influxdb-client) — DB 저장 비활성화"
            )
        except Exception as e:
            logger.warning("InfluxDB 연결 실패: %s — DB 저장 비활성화", e)
        return False

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        logger.info(
            "InfluxDB 종료: writes=%d errors=%d", self._write_count, self._error_count
        )

    def write_gait_result(
        self,
        user_id: str,
        device_id: str,
        session_id: str,
        ensemble_output: Any,           # EnsembleOutput dataclass
        speed_mean: float = 0.0,
        tilt_mean: float = 0.0,
        timestamp_ns: int | None = None,
    ) -> bool:
        """
        단일 추론 결과를 InfluxDB Point로 변환 후 쓰기.

        Args:
            ensemble_output: ShoeallasCESEnsemble.infer() 반환값 (EnsembleOutput)
        Returns:
            True if written (or queued), False if unavailable
        """
        if not self._available or self._write_api is None:
            return False

        try:
            from influxdb_client import Point

            score = float(ensemble_output.anomaly_score)
            risk_band = (
                "high" if score >= 0.70
                else "moderate" if score >= 0.50
                else "low"
            )

            ts = timestamp_ns or int(time.time_ns())

            point = (
                Point("gait_sessions")
                # 태그 (인덱스 대상, 낮은 카디널리티)
                .tag("user_id", user_id)
                .tag("device_id", device_id)
                .tag("session_id", session_id)
                .tag("gait_label", str(ensemble_output.gait_label))
                .tag("risk_band", risk_band)
                # 필드 (측정값)
                .field("anomaly_score", score)
                .field("fall_alert", int(ensemble_output.fall_alert))
                .field("ae_reconstruction_loss", float(ensemble_output.ae_reconstruction_loss))
                .field("rule_score", float(ensemble_output.rule_score))
                .field("lstm_risk_score", float(ensemble_output.lstm_risk_score))
                .field("speed_mean", float(speed_mean))
                .field("tilt_mean", float(tilt_mean))
                .field("inference_ms", float(ensemble_output.inference_ms))
                .time(ts)
            )

            self._write_api.write(bucket=self.cfg.bucket, org=self.cfg.org, record=point)
            self._write_count += 1

            if ensemble_output.fall_alert:
                self._write_fall_alert(user_id, session_id, score, ts)

            return True

        except Exception as e:
            logger.error("InfluxDB 쓰기 실패: %s", e)
            self._error_count += 1
            return False

    def _write_fall_alert(
        self, user_id: str, session_id: str, score: float, ts: int
    ) -> None:
        """낙상 알림을 별도 measurement에도 기록 (보호자 알림 이력용)"""
        try:
            from influxdb_client import Point

            point = (
                Point("fall_alerts")
                .tag("user_id", user_id)
                .tag("session_id", session_id)
                .field("anomaly_score", score)
                .field("acknowledged", 0)
                .time(ts)
            )
            self._write_api.write(bucket=self.cfg.bucket, org=self.cfg.org, record=point)
        except Exception as e:
            logger.error("낙상 알림 기록 실패: %s", e)

    def query_trend(
        self,
        user_id: str,
        period: str = "7d",
        aggregate: str = "mean",
    ) -> list[dict]:
        """
        일/주/월 추세 쿼리.

        Args:
            period: "1d" | "7d" | "30d"
            aggregate: "mean" | "max" | "min"
        Returns:
            [{"time": str, "anomaly_score": float, "fall_count": int}, ...]
        """
        if not self._available or self._client is None:
            return []

        try:
            query_api = self._client.query_api()
            flux = f"""
from(bucket: "{self.cfg.bucket}")
  |> range(start: -{period})
  |> filter(fn: (r) => r._measurement == "gait_sessions")
  |> filter(fn: (r) => r.user_id == "{user_id}")
  |> filter(fn: (r) => r._field == "anomaly_score" or r._field == "fall_alert")
  |> aggregateWindow(every: 1h, fn: {aggregate}, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
"""
            tables = query_api.query(flux, org=self.cfg.org)
            results = []
            for table in tables:
                for record in table.records:
                    results.append({
                        "time": str(record.get_time()),
                        "anomaly_score": float(record.values.get("anomaly_score", 0)),
                        "fall_count": int(record.values.get("fall_alert", 0)),
                    })
            return results

        except Exception as e:
            logger.error("InfluxDB 쿼리 실패: %s", e)
            return []

    def stats(self) -> dict[str, int | bool]:
        return {
            "available": self._available,
            "writes": self._write_count,
            "errors": self._error_count,
        }


# ─── 싱글톤 접근자 ────────────────────────────────────────────────────────────

_writer_instance: GaitInfluxWriter | None = None


def get_influx_writer() -> GaitInfluxWriter:
    """FastAPI 의존성 주입용 싱글톤 접근자"""
    global _writer_instance
    if _writer_instance is None:
        _writer_instance = GaitInfluxWriter()
        _writer_instance.connect()
    return _writer_instance


# ─── 비동기 래퍼 (FastAPI background task용) ─────────────────────────────────


async def async_write_gait_result(
    user_id: str,
    device_id: str,
    session_id: str,
    ensemble_output: Any,
    speed_mean: float = 0.0,
    tilt_mean: float = 0.0,
) -> None:
    """
    FastAPI BackgroundTask로 사용하는 비동기 래퍼.

    사용 예시 (websocket_stream.py):
        from fastapi import BackgroundTasks
        background_tasks.add_task(
            async_write_gait_result, user_id, device_id, session_id, output
        )
    """
    writer = get_influx_writer()
    loop = asyncio.get_event_loop()
    # 동기 쓰기를 스레드풀에서 실행 (블로킹 방지)
    await loop.run_in_executor(
        None,
        writer.write_gait_result,
        user_id, device_id, session_id, ensemble_output, speed_mean, tilt_mean,
    )
