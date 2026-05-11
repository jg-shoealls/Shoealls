"""Shoealls Data Ingestion Module.

BLE 5.0 수신 데이터를 파싱하여 Apache Kafka로 스트리밍하는 모듈.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any

# Kafka 클라이언트는 kafka-python 또는 confluent-kafka를 사용한다고 가정
try:
    from kafka import KafkaProducer
except ImportError:
    KafkaProducer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BLEDataIngestor:
    def __init__(self, bootstrap_servers: str = 'localhost:9092', topic: str = 'shoealls-sensor'):
        self.topic = topic
        if KafkaProducer:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info(f"Kafka Producer initialized for topic: {topic}")
        else:
            self.producer = None
            logger.warning("KafkaProducer not found. Mock mode enabled.")

    def parse_ble_hex(self, hex_data: str) -> Dict[str, Any]:
        """
        BLE 5.0에서 수신된 헥사 데이터를 파싱합니다.
        (실제 하드웨어 사양에 맞춰 바이트 슬라이싱 필요)
        """
        # TODO: 실제 슈즈 바이트 프로토콜에 따라 구현
        # 예시: [Header(1)][DeviceID(2)][SensorType(1)][Value(4)][Checksum(1)]
        try:
            # 여기서는 데모를 위해 가상의 파싱 로직을 구현합니다.
            parsed = {
                "device_idx": int(hex_data[0:2], 16),
                "event_type": "acceleration", # 가속도 센서 고정 예시
                "event_value": int(hex_data[2:6], 16) / 100.0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return parsed
        except Exception as e:
            logger.error(f"Error parsing hex data {hex_data}: {e}")
            return {}

    def stream_data(self, hex_packet: str):
        """파싱된 데이터를 Kafka로 전송"""
        data = self.parse_ble_hex(hex_packet)
        if not data:
            return

        if self.producer:
            self.producer.send(self.topic, value=data)
            logger.info(f"Streamed to Kafka: {data}")
        else:
            logger.info(f"[Mock Kafka] Sending: {data}")

    def run_simulator(self):
        """실제 하드웨어가 없을 때를 위한 시뮬레이터"""
        logger.info("Starting BLE Data Simulator...")
        mock_packets = ["04AABB", "04CCDD", "04FFEE"]
        while True:
            for packet in mock_packets:
                self.stream_data(packet)
                time.sleep(1)

if __name__ == "__main__":
    ingestor = BLEDataIngestor()
    # ingestor.run_simulator() # 실제 환경에서는 하드웨어 루프에서 stream_data 호출
    ingestor.run_simulator()
