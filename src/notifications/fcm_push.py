"""
Firebase Cloud Messaging (FCM) 푸시 알림 모듈.

낙상(fall_alert=True) 감지 시 보호자/사용자 모바일 앱에 즉각 알림 전달.
FCM 미설정 시 로그만 출력하고 정상 반환 (graceful degradation).

환경 변수:
  FIREBASE_CREDENTIALS_PATH  Firebase Admin SDK JSON 키 파일 경로
  FIREBASE_PROJECT_ID        Firebase 프로젝트 ID

DB 토큰 연동:
  get_fcm_tokens_for_user(user_id) 함수를 실제 DB 조회로 교체하세요.
  현재는 DEVICE_TOKEN_STORE 환경변수 또는 인메모리 레지스트리를 사용합니다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ─── 설정 ────────────────────────────────────────────────────────────────────


@dataclass
class FCMConfig:
    credentials_path: str = ""
    project_id: str = ""
    dry_run: bool = False       # True: 실제 전송 없이 로그만 출력 (테스트용)

    def __post_init__(self) -> None:
        self.credentials_path = (
            self.credentials_path
            or os.getenv("FIREBASE_CREDENTIALS_PATH", "")
        )
        self.project_id = (
            self.project_id or os.getenv("FIREBASE_PROJECT_ID", "")
        )
        if not self.credentials_path:
            self.dry_run = True

    @property
    def is_configured(self) -> bool:
        return bool(self.credentials_path and self.project_id)


# ─── 알림 페이로드 ─────────────────────────────────────────────────────────────


@dataclass
class FallAlertPayload:
    user_id: str
    session_id: str
    anomaly_score: float
    recommendation: str
    soft_labels: list[str]
    gait_label: str

    def to_notification_data(self) -> dict[str, str]:
        """FCM data payload (모든 값은 string)"""
        return {
            "type": "fall_alert",
            "user_id": self.user_id,
            "session_id": self.session_id,
            "anomaly_score": f"{self.anomaly_score:.3f}",
            "gait_label": self.gait_label,
            "soft_labels": json.dumps(self.soft_labels[:3], ensure_ascii=False),
            "recommendation": self.recommendation[:200],
        }

    def to_notification_body(self) -> str:
        score_pct = int(self.anomaly_score * 100)
        return f"이상 점수 {score_pct}% — {self.recommendation[:60]}"


@dataclass
class GaitUpdatePayload:
    user_id: str
    session_id: str
    anomaly_score: float
    gait_label: str

    def to_notification_data(self) -> dict[str, str]:
        return {
            "type": "gait_update",
            "user_id": self.user_id,
            "session_id": self.session_id,
            "anomaly_score": f"{self.anomaly_score:.3f}",
            "gait_label": self.gait_label,
        }


# ─── 기기 토큰 레지스트리 ─────────────────────────────────────────────────────
# 실제 서비스에서는 SQLAlchemy ORM의 device_tokens 테이블 조회로 교체


_token_registry: dict[str, list[str]] = {}


def register_fcm_token(user_id: str, token: str, platform: str = "android") -> None:
    """FCM 토큰 등록 (런타임 인메모리 — 실서비스에서는 DB로 교체)"""
    if user_id not in _token_registry:
        _token_registry[user_id] = []
    if token not in _token_registry[user_id]:
        _token_registry[user_id].append(token)
        logger.info("FCM 토큰 등록: user=%s platform=%s", user_id, platform)


def get_fcm_tokens_for_user(user_id: str) -> list[str]:
    """
    user_id의 FCM 토큰 목록 반환.

    실서비스 교체 예시:
        async with AsyncSession(engine) as session:
            result = await session.execute(
                select(DeviceToken.fcm_token)
                .where(DeviceToken.user_id == user_id)
            )
            return [r[0] for r in result.all()]
    """
    # 환경변수 기반 테스트 토큰 (개발용)
    env_token = os.getenv(f"FCM_TOKEN_{user_id.upper().replace('-', '_')}")
    if env_token:
        return [env_token]
    return list(_token_registry.get(user_id, []))


# ─── FCM 전송 클래스 ─────────────────────────────────────────────────────────


class FCMPushNotifier:
    """
    Firebase Admin SDK를 통한 FCM 푸시 알림 전송.

    특징:
    - 미설치/미설정 시 graceful degradation (서버 중단 없음)
    - dry_run=True 시 실제 전송 없이 로그만 출력
    - 낙상 알림: 높은 우선순위 (priority=high)
    - 일반 업데이트: 낮은 우선순위 (data-only, background)
    """

    FALL_TITLE = "낙상 위험 감지"
    GAIT_ANOMALY_TITLE = "보행 이상 징후"

    def __init__(self, cfg: FCMConfig | None = None) -> None:
        self.cfg = cfg or FCMConfig()
        self._initialized = False
        self._messaging: Any = None
        self._send_count = 0
        self._error_count = 0

    def initialize(self) -> bool:
        """Firebase Admin SDK 초기화. 실패 시 False 반환"""
        if self._initialized:
            return True

        if self.cfg.dry_run:
            logger.warning(
                "FCM dry_run 모드 — 실제 알림 전송 없음 "
                "(FIREBASE_CREDENTIALS_PATH 환경변수 설정 필요)"
            )
            self._initialized = True
            return True

        try:
            import firebase_admin
            from firebase_admin import credentials, messaging

            cred = credentials.Certificate(self.cfg.credentials_path)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)

            self._messaging = messaging
            self._initialized = True
            logger.info(
                "Firebase Admin SDK 초기화 성공: project=%s", self.cfg.project_id
            )
            return True

        except ImportError:
            logger.warning(
                "firebase-admin 미설치 (pip install firebase-admin) — FCM 알림 비활성화"
            )
        except Exception as e:
            logger.warning("Firebase 초기화 실패: %s — FCM 알림 비활성화", e)
        return False

    async def send_fall_alert(
        self,
        user_id: str,
        payload: FallAlertPayload,
        also_notify_guardians: bool = True,
    ) -> dict[str, int]:
        """
        낙상 감지 즉각 알림 전송.

        user_id의 모든 기기 + 연결된 보호자 기기에 HIGH priority로 전송.

        Returns:
            {"sent": n, "failed": n, "skipped": n}
        """
        if not self._initialized:
            self.initialize()

        tokens = get_fcm_tokens_for_user(user_id)
        if also_notify_guardians:
            # 실서비스: guardian_links 테이블 조회로 교체
            guardian_ids = _get_guardian_ids(user_id)
            for gid in guardian_ids:
                tokens.extend(get_fcm_tokens_for_user(gid))

        if not tokens:
            logger.warning("FCM 토큰 없음: user_id=%s — 알림 전송 불가", user_id)
            return {"sent": 0, "failed": 0, "skipped": 1}

        return await self._send_multicast(
            tokens=tokens,
            title=self.FALL_TITLE,
            body=payload.to_notification_body(),
            data=payload.to_notification_data(),
            priority="high",
        )

    async def send_gait_anomaly_alert(
        self,
        user_id: str,
        payload: GaitUpdatePayload,
    ) -> dict[str, int]:
        """이상 징후 경고 알림 (낙상 아님, 일반 우선순위)"""
        if not self._initialized:
            self.initialize()

        tokens = get_fcm_tokens_for_user(user_id)
        if not tokens:
            return {"sent": 0, "failed": 0, "skipped": 1}

        score_pct = int(payload.anomaly_score * 100)
        return await self._send_multicast(
            tokens=tokens,
            title=self.GAIT_ANOMALY_TITLE,
            body=f"보행 이상 점수 {score_pct}% — 모니터링을 확인하세요",
            data=payload.to_notification_data(),
            priority="normal",
        )

    async def _send_multicast(
        self,
        tokens: list[str],
        title: str,
        body: str,
        data: dict[str, str],
        priority: str = "high",
    ) -> dict[str, int]:
        """FCM MulticastMessage 전송 (비동기, 스레드풀)"""
        if self.cfg.dry_run or self._messaging is None:
            logger.info(
                "[FCM DRY-RUN] title=%r body=%r tokens=%d",
                title, body[:50], len(tokens),
            )
            return {"sent": len(tokens), "failed": 0, "skipped": 0}

        try:
            from firebase_admin.messaging import (
                AndroidConfig,
                AndroidNotification,
                MulticastMessage,
                Notification,
            )

            message = MulticastMessage(
                notification=Notification(title=title, body=body),
                android=AndroidConfig(
                    priority="high" if priority == "high" else "normal",
                    notification=AndroidNotification(
                        title=title,
                        body=body,
                        sound="fall_alert.wav" if "낙상" in title else "default",
                        channel_id="shoealls_alerts",
                    ),
                ),
                data=data,
                tokens=tokens[:500],  # FCM 최대 500개
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self._messaging.send_each_for_multicast, message
            )

            sent = response.success_count
            failed = response.failure_count
            self._send_count += sent
            self._error_count += failed

            if failed > 0:
                for i, resp in enumerate(response.responses):
                    if not resp.success:
                        logger.warning(
                            "FCM 전송 실패: token=%s... error=%s",
                            tokens[i][:10], resp.exception,
                        )

            logger.info(
                "FCM 전송 완료: title=%r sent=%d failed=%d",
                title, sent, failed,
            )
            return {"sent": sent, "failed": failed, "skipped": 0}

        except Exception as e:
            logger.error("FCM 전송 오류: %s", e)
            self._error_count += len(tokens)
            return {"sent": 0, "failed": len(tokens), "skipped": 0}

    def stats(self) -> dict[str, int]:
        return {"sent": self._send_count, "errors": self._error_count}


# ─── 보호자 링크 레지스트리 (인메모리, 실서비스는 DB 교체) ───────────────────

_guardian_links: dict[str, list[str]] = {}


def link_guardian(user_id: str, guardian_id: str) -> None:
    """보호자-사용자 연결 등록"""
    if user_id not in _guardian_links:
        _guardian_links[user_id] = []
    if guardian_id not in _guardian_links[user_id]:
        _guardian_links[user_id].append(guardian_id)
        logger.info("보호자 연결: user=%s guardian=%s", user_id, guardian_id)


def _get_guardian_ids(user_id: str) -> list[str]:
    return list(_guardian_links.get(user_id, []))


# ─── 싱글톤 접근자 ────────────────────────────────────────────────────────────

_notifier_instance: FCMPushNotifier | None = None


def get_fcm_notifier() -> FCMPushNotifier:
    """FastAPI 의존성 주입용 싱글톤 접근자"""
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = FCMPushNotifier()
        _notifier_instance.initialize()
    return _notifier_instance


# ─── 통합 알림 함수 (websocket_stream.py에서 호출) ────────────────────────────


async def notify_if_needed(
    user_id: str,
    session_id: str,
    ensemble_output: Any,          # EnsembleOutput
    fall_threshold: float = 0.70,
    anomaly_threshold: float = 0.85,
) -> dict[str, Any]:
    """
    앙상블 출력 기반 FCM 알림 전송 결정 로직.

    - fall_alert=True → 즉각 낙상 알림 (보호자 포함)
    - anomaly_score >= anomaly_threshold → 이상 징후 경고
    - 그 외 → 알림 없음

    Returns:
        {"alert_type": str, "fcm_result": dict}
    """
    notifier = get_fcm_notifier()

    if ensemble_output.fall_alert:
        payload = FallAlertPayload(
            user_id=user_id,
            session_id=session_id,
            anomaly_score=float(ensemble_output.anomaly_score),
            recommendation=str(ensemble_output.recommendation),
            soft_labels=list(ensemble_output.soft_labels),
            gait_label=str(ensemble_output.gait_label),
        )
        result = await notifier.send_fall_alert(user_id, payload, also_notify_guardians=True)
        return {"alert_type": "fall_alert", "fcm_result": result}

    if float(ensemble_output.anomaly_score) >= anomaly_threshold:
        payload = GaitUpdatePayload(
            user_id=user_id,
            session_id=session_id,
            anomaly_score=float(ensemble_output.anomaly_score),
            gait_label=str(ensemble_output.gait_label),
        )
        result = await notifier.send_gait_anomaly_alert(user_id, payload)
        return {"alert_type": "anomaly_warning", "fcm_result": result}

    return {"alert_type": "none", "fcm_result": {"sent": 0, "failed": 0, "skipped": 1}}
