import requests
import json

url = "http://127.0.0.1:8001/api/v1/auth/signup"
payload = {
    "email": "test@shoealls.com",
    "full_name": "테스트 유저",
    "role": "patient",
    "password": "password123"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
