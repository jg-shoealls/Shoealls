# 파킨슨병 보행 패턴 조기 진단 및 바이오마커 분석 리포트

본 문서는 Shoealls 프로젝트를 활용한 파킨슨병(Parkinson's Disease) 보행 패턴 분석 및 조기 진단을 위한 핵심 바이오마커와 모델 학습 설정을 정리한 리포트입니다.

## 1. 핵심 보행 바이오마커 (Biomarkers)

파킨슨병 조기 진단을 위해 본 프로젝트에서 추출하고 분석하는 핵심 지표들입니다.

| 바이오마커 | 한국어 명칭 | 정상 범위 | 파킨슨병 특이 패턴 |
| :--- | :--- | :--- | :--- |
| **Gait Speed** | 보행 속도 | 1.0 ~ 1.4 m/s | 전반적인 저하 또는 충동 보행(Festination)으로 인한 비정상적 가속 |
| **Cadence** | 보행률 | 100 ~ 130 steps/min | 짧은 보폭과 함께 보행률이 급격히 증가하는 경향 |
| **Stride Regularity** | 보폭 규칙성 | 0.7 ~ 1.0 | 걸음마다 보폭이 일정하지 않고 불규칙해짐 (0.7 미만) |
| **Acceleration RMS** | 가속도 크기 | 0.8 ~ 2.5 m/s² | 발을 떼는 추진력 약화 (0.8 미만), 동결 보행 전조 |
| **Trunk Sway** | 체간 흔들림 | 0.0 ~ 3.0 deg/s | 상체의 미세한 흔들림 증가 및 균형 장애 |
| **Asymmetry** | 좌우 비대칭성 | < 0.12 (Index) | 한쪽 측면에서 시작되는 증상으로 인한 좌우 압력/가속도 차이 발생 |

## 2. 추천 인공지능 모델 (Hugging Face SLM)

VRAM 3GB(GTX 1060 등) 이하의 저사양 환경에서도 로컬에서 효율적으로 구동 가능한 최신 Small Language Models(SLM)입니다.

*   **Gemma 2 2B (Google):** 3GB VRAM 환경에서 가장 추천되는 범용 모델. 가볍지만 높은 지능을 보유.
*   **Llama 3.2 3B (Meta):** 한국어 이해도와 대화 성능이 우수하며 4-bit 양자화 시 3GB 내에서 구동 가능.
*   **Phi-3.5 Mini (Microsoft):** 논리적 추론 및 보행 데이터 분석 로직 보조에 강점.

## 3. WearGait-PD 데이터셋 학습 설정

프로젝트 내 구현된 `WearGait-PD` 데이터셋(압력 센서 + IMU 퓨전) 학습 프로세스입니다.

*   **설정 파일:** `configs/weargait_loso.yaml`
*   **학습 방식:** LOSO (Leave-One-Subject-Out) 교차 검증
*   **입력 데이터:** 16채널 족저압(Insole Pressure) + 12채널 IMU(가속도/자이로)
*   **모델 구조:** `IMUPressureGaitNet` (Multimodal Fusion Network)

## 4. 실행 환경 및 도구
*   **Ollama:** 로컬 SLM 모델 실행 및 인터페이스 관리.
*   **Hugging Face Skills:** 최신 연구 데이터 및 모델 검색 자동화.
*   **PyTorch:** 보행 데이터 퓨전 모델 학습 및 추론.

---
*작성일: 2026-05-10*
*작성자: Gemini CLI / Shoealls Project Assistant*
