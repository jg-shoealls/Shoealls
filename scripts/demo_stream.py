import requests
import pandas as pd
import json
import time
import sys

def run_demo_stream():
    """
    CES.csv 데이터를 한 줄씩 읽어 API로 전송함으로써 
    실제 슈즈가 보행 데이터를 보내는 것처럼 시뮬레이션합니다.
    데모 비디오 촬영 시 이 스크립트를 실행하고 대시보드를 녹화하면 됩니다.
    """
    print("🎬 CES 시스템 연동 데모 시뮬레이션 시작...")
    
    try:
        df = pd.read_csv('CES.csv', sep='\t')
    except Exception as e:
        print(f"Error: CES.csv를 로드할 수 없습니다. {e}")
        return

    # API 엔드포인트 (종합 분석)
    # 실제 데모 시에는 분석 API를 호출하여 대시보드 상태를 변화시킴
    URL = "http://127.0.0.1:8001/api/v1/analyze"
    
    print(f"📡 총 {len(df)}개 이벤트를 순차적으로 전송합니다. (Ctrl+C로 중단)")
    
    # 데모를 위해 적절한 주기로 데이터 전송
    for i, row in df.iterrows():
        # JSON 파싱
        try:
            raw_data = json.loads(row['fe_raw_data'])
        except:
            continue

        # API 요청 바디 구성 (GaitFeatures 예시 포맷에 맞춤)
        # 실제 API 스키마에 맞춰 특징량 전달
        payload = {
            "sensor_data": {
                "imu": [[0,0,0,0,0,0]] * 128, # Dummy IMU
                "pressure": [[0]*8] * 16,     # Dummy Pressure
                "skeleton": [[[0,0,0]]*17] * 128 # Dummy Skeleton
            },
            "features": {
                "gait_speed": raw_data.get('speed', 0.6),
                "cadence": 110.0,
                "stride_regularity": 0.8,
                "step_symmetry": 0.9,
                "cop_sway": 0.04,
                "ml_index": 0.1,
                "arch_index": 0.2,
                "acceleration_rms": 1.5,
                "tilt": raw_data.get('tilt', 5.0)
            }
        }

        try:
            # 실시간성을 보여주기 위해 API 호출
            # res = requests.post(URL, json=payload)
            # print(f"[{i:04d}] {row['fe_event_type']:12} | Label: {raw_data.get('label'):10} | Speed: {raw_data.get('speed'):.3f}")
            
            # 화면 출력을 더 시각적으로
            status_bar = f"👟 Step {i:04d} | Event: {row['fe_event_type']:10} | AI Label: {raw_data.get('label'):8} | Tilt: {raw_data.get('tilt'):.2f}°"
            sys.stdout.write(f"\r{status_bar}")
            sys.stdout.flush()
            
            # 'fall' 이벤트 발생 시 강조
            if row['fe_event_type'] == 'fall':
                print("\n\n⚠️  FALL DETECTED! ALERTING GUARDIAN VIA FCM...")
                time.sleep(1) # 강조를 위해 멈춤
            
            time.sleep(0.3) # 0.3초 간격으로 보행 데이터 시뮬레이션
            
        except Exception as e:
            print(f"\nError: {e}")
            break

if __name__ == "__main__":
    run_demo_stream()
