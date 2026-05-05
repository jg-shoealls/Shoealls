import httpx
import numpy as np
import json

BASE_URL = "http://localhost:8000"

def test_health():
    with httpx.Client() as client:
        res = client.get(f"{BASE_URL}/health")
        print("Health Check:", res.json())

def test_analyze():
    # Dummy data
    imu = np.zeros((128, 6)).tolist()
    pressure = np.zeros((128, 16, 8)).tolist() # T, H, W
    skeleton = np.zeros((128, 17, 3)).tolist()
    
    payload = {
        "imu": imu,
        "pressure": pressure,
        "skeleton": skeleton,
        "gait_profile": "normal"
    }
    
    print("Testing /api/v1/analyze...")
    with httpx.Client(timeout=30.0) as client:
        res = client.post(f"{BASE_URL}/api/v1/analyze", json=payload)
        if res.status_code == 200:
            result = res.json()
            print("Analysis Result:")
            print(f"  Prediction: {result['classify']['prediction_kr']}")
            print(f"  Confidence: {result['classify']['confidence']:.1%}")
            print(f"  Report: {result['reasoning']['report_kr'][:100]}...")
        else:
            print(f"Error: {res.status_code}")
            print(res.text)

if __name__ == "__main__":
    try:
        test_health()
        test_analyze()
    except Exception as e:
        print(f"Test failed: {e}")
