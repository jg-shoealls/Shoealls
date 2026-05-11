import logging
import firebase_admin
from firebase_admin import credentials, messaging
from typing import List, Dict, Any
from src.config import config

logger = logging.getLogger(__name__)

class FCMNotificationService:
    """
    Firebase Cloud Messaging을 이용한 실시간 알림 서비스.
    낙상 감지 및 건강 이상 징후 발생 시 보호자 및 의료진에게 푸시 알림을 전송합니다.
    """
    _initialized = False

    @classmethod
    def initialize(cls):
        if not cls._initialized:
            try:
                if os.path.exists(config.FCM_CREDENTIAL_PATH):
                    cred = credentials.Certificate(config.FCM_CREDENTIAL_PATH)
                    firebase_admin.initialize_app(cred)
                    cls._initialized = True
                    logger.info("FCM Service initialized successfully.")
                else:
                    logger.warning(f"FCM credential not found at {config.FCM_CREDENTIAL_PATH}. Notifications will be logged only.")
            except Exception as e:
                logger.error(f"Failed to initialize FCM: {e}")

    @staticmethod
    def send_push_notification(token: str, title: str, body: str, data: Dict[str, str] = None):
        """단일 디바이스에 푸시 알림 전송"""
        if not FCMNotificationService._initialized:
            logger.info(f"[Mock Push] To: {token}, Title: {title}, Body: {body}")
            return

        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            data=data or {},
            token=token,
        )

        try:
            response = messaging.send(message)
            logger.info(f"Successfully sent message: {response}")
            return response
        except Exception as e:
            logger.error(f"Error sending FCM message: {e}")

    @staticmethod
    def notify_fall_detected(user_id: str, guardian_tokens: List[str], risk_score: float):
        """낙상 감지 시 긴급 알림 전송"""
        title = "⚠️ 긴급: 낙상 위험 감지"
        body = f"사용자({user_id})님의 낙상 위험이 감지되었습니다. 즉시 확인이 필요합니다. (위험도: {risk_score*100:.1f}%)"
        
        data = {
            "type": "FALL_ALERT",
            "user_id": user_id,
            "risk_score": str(risk_score),
            "timestamp": str(datetime.now())
        }

        for token in guardian_tokens:
            FCMNotificationService.send_push_notification(token, title, body, data)

import os
from datetime import datetime
