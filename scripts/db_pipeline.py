"""Shoealls Time-series DB Module.

Kafka로부터 데이터를 소비하여 AES-256 암호화 후 InfluxDB에 적재하는 모듈.
"""

import json
import logging
from typing import Dict, Any

# InfluxDB 및 암호화 라이브러리 가정
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    from cryptography.fernet import Fernet
except ImportError:
    InfluxDBClient = None
    Fernet = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShoeallsDBPipeline:
    def __init__(self, 
                 url: str = "http://localhost:8086", 
                 token: str = "my-token", 
                 org: str = "shoealls", 
                 bucket: str = "sensor_data",
                 enc_key: str = None):
        self.bucket = bucket
        self.org = org
        
        # InfluxDB 초기화
        if InfluxDBClient:
            self.client = InfluxDBClient(url=url, token=token, org=org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            logger.info(f"Connected to InfluxDB at {url}")
        else:
            self.client = None
            logger.warning("InfluxDBClient not found. Mock mode enabled.")

        # AES-256 암호화 (Fernet 사용)
        if Fernet and enc_key:
            self.cipher = Fernet(enc_key.encode())
            logger.info("AES-256 encryption enabled.")
        else:
            self.cipher = None
            logger.warning("Encryption key not provided or Fernet missing. Data will be saved in plain text.")

    def encrypt_data(self, data_str: str) -> str:
        """데이터를 AES-256으로 암호화합니다."""
        if self.cipher:
            return self.cipher.encrypt(data_str.encode()).decode()
        return data_str

    def process_message(self, message: Dict[str, Any]):
        """Kafka 메시지를 처리하여 DB에 저장"""
        try:
            device_id = str(message.get('device_idx', 'unknown'))
            event_type = message.get('event_type', 'unknown')
            event_value = message.get('event_value', 0.0)
            
            # 민감한 로우 데이터는 암호화하여 저장 가능
            raw_payload = json.dumps(message)
            encrypted_payload = self.encrypt_data(raw_payload)

            if self.client:
                point = Point("gait_event") \
                    .tag("device_id", device_id) \
                    .tag("event_type", event_type) \
                    .field("value", event_value) \
                    .field("raw_encrypted", encrypted_payload)
                
                self.write_api.write(bucket=self.bucket, org=self.org, record=point)
                logger.info(f"Saved to InfluxDB: Device {device_id}, Event {event_type}")
            else:
                logger.info(f"[Mock DB] Storing: Device {device_id}, Value {event_value}, Encrypted={bool(self.cipher)}")

        except Exception as e:
            logger.error(f"Error processing message for DB: {e}")

    def run_mock_consumer(self):
        """테스트를 위한 모의 컨슈머 루프"""
        logger.info("Starting Mock Kafka Consumer...")
        sample_messages = [
            {"device_idx": 4, "event_type": "acceleration", "event_value": 0.62, "timestamp": "2026-01-06 08:55:22"},
            {"device_idx": 4, "event_type": "fall", "event_value": 1.0, "timestamp": "2026-01-06 08:56:10"}
        ]
        for msg in sample_messages:
            self.process_message(msg)

if __name__ == "__main__":
    # 테스트용 대칭키 생성: Fernet.generate_key().decode()
    test_key = "uK5h_Z3P8-V6o4n5m5m5m5m5m5m5m5m5m5m5m5m5m5=" # 32-byte base64 key placeholder
    pipeline = ShoeallsDBPipeline(enc_key=test_key)
    pipeline.run_mock_consumer()
