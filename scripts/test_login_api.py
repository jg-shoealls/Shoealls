import requests

url_login = "http://127.0.0.1:8001/api/v1/auth/login"
data = {
    "username": "test@shoealls.com",
    "password": "password123"
}

try:
    response = requests.post(url_login, data=data)
    print(f"Login Status: {response.status_code}")
    if response.status_code == 200:
        token = response.json().get("access_token")
        print(f"Access Token: {token[:20]}...")
        
        # /me 엔드포인트 테스트
        url_me = "http://127.0.0.1:8001/api/v1/auth/me"
        headers = {"Authorization": f"Bearer {token}"}
        res_me = requests.get(url_me, headers=headers)
        print(f"Me Status: {res_me.status_code}")
        print(f"Me Response: {res_me.text}")
    else:
        print(f"Login Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
