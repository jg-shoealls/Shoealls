import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- Kafka ---
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_SENSOR_TOPIC = os.getenv("KAFKA_SENSOR_TOPIC", "shoealls-sensor")
    
    # --- InfluxDB ---
    INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
    INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "your-token")
    INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "shoealls")
    INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "sensor_data")
    
    # --- Security ---
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", None) # AES-256 Key
    
    # --- FCM (Firebase Cloud Messaging) ---
    FCM_CREDENTIAL_PATH = os.getenv("FCM_CREDENTIAL_PATH", "configs/firebase-adminsdk.json")
    FCM_PROJECT_ID = os.getenv("FCM_PROJECT_ID", "shoealls-healthcare")

    # --- Analysis ---
    FALL_THRESHOLD = float(os.getenv("FALL_THRESHOLD", 0.65))

config = Config()
