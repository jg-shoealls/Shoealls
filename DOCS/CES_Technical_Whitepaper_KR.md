# Shoealls CES Innovation Award 기술 백서 초안

**문서 목적:** CES Innovation Award 신청 및 기술 심사용 초안  
**제품명:** Shoealls Multimodal Smart Gait Healthcare Platform  
**버전:** Draft v0.1  
**작성일:** 2026-05-11  

---

## 1. Executive Summary

Shoealls는 스마트슈즈에 내장된 족저 압력 센서와 IMU 센서 데이터를 활용하여 보행 패턴, 끌림 보행, 낙상 이벤트, 좌우 비대칭, 체중 이동 특성을 실시간으로 분석하는 멀티모달 헬스케어 플랫폼이다.

본 플랫폼은 고령자, 보행 취약자, 재활 사용자, 보호자, 의료진을 대상으로 보행 이상 징후를 조기에 감지하고, 낙상 위험 알림과 보행 리포트를 제공하는 것을 목표로 한다. Shoealls는 질병을 확정 진단하지 않으며, 센서 기반 보행 위험 징후, 이상 패턴, 전문 의료기관 상담 권고 수준의 정보를 제공한다.

핵심 차별점은 다음과 같다.

- 신발 내부 센서 기반의 일상 보행 데이터 수집
- 좌/우 발 분리 분석이 가능한 시계열 데이터 구조
- 룰 기반 낙상 즉시 알림과 LSTM 기반 보행 상태 분류 결합
- 향후 LSTM Autoencoder, Transformer, Ensemble AI 모델로 확장 가능한 inference interface
- 사용자, 보호자, 관리자/의료진용 대시보드 플랫폼 구조
- 고령자 안전, 돌봄 부담 완화, 예방 중심 헬스케어, 이동 약자 지원이라는 사회적 가치

---

## 2. Problem Statement

고령자의 낙상은 골절, 장기 입원, 독립 생활 상실, 보호자 돌봄 부담 증가로 이어질 수 있다. 또한 보행 속도 저하, 좌우 비대칭, 끌림 보행, 체중 이동 불안정성은 건강 상태 변화를 보여주는 중요한 생활 속 신호가 될 수 있다.

기존 보행 평가는 병원, 재활센터, 실험실 환경에서 단기간 측정되는 경우가 많다. 이 방식은 다음 한계를 가진다.

- 일상 환경의 장기 보행 변화를 반영하기 어렵다.
- 고령자와 이동 취약자의 병원 방문 부담이 크다.
- 낙상 위험을 사후에 인지하는 경우가 많다.
- 보호자와 의료진이 연속적인 보행 변화를 확인하기 어렵다.

Shoealls는 사용자가 신발을 착용하고 걷는 과정에서 데이터를 수집하여, 일상적인 보행 변화와 위험 징후를 지속적으로 추적하는 플랫폼을 지향한다.

---

## 3. Product Concept

Shoealls 플랫폼은 다음 4개 계층으로 구성된다.

1. **Smart Shoes Sensor Layer**
   - 족저 압력 센서
   - IMU 센서: 가속도, 자이로스코프, 지자기 또는 각속도 계열
   - 좌/우 발 분리 데이터
   - BLE 기반 실시간 송신

2. **Data Pipeline Layer**
   - BLE gateway
   - 실시간 이벤트 수신
   - 시계열 데이터 정규화
   - 추후 Kafka, InfluxDB 연동 가능 구조

3. **AI & Rule-based Analysis Layer**
   - CES 이벤트 데이터 전처리
   - sliding window time-series dataset
   - GaitLSTM 기반 보행 상태 분류
   - fall event rule-based immediate alert
   - 향후 LSTM Autoencoder, Transformer, Ensemble 확장

4. **Healthcare Platform Layer**
   - 사용자 대시보드
   - 보호자 모니터링
   - 의료진/관리자 관제 화면
   - 낙상 위험 알림
   - 보행 리포트 및 추세 분석

---

## 4. Dataset Schema and Current Data Pipeline

현재 확보된 `CES.csv` 데이터는 TSV 형식이며, 다음 컬럼을 가진다.

| Column | Type | Description |
|---|---:|---|
| `fe_idx` | int | 이벤트 고유 ID |
| `fe_device_idx` | int | 장치 ID |
| `fe_user_idx` | string | 사용자 ID, 결측 가능 |
| `fe_event_type` | string | `acceleration`, `shuffling`, `fall` |
| `fe_event_value` | float | 이벤트 측정값 |
| `fe_raw_data` | JSON string | 센서/분석 원천 JSON |
| `fe_analysis_result` | JSON string | 분석 결과 JSON |
| `fe_timestamp` | datetime string | 이벤트 발생 시각 |

`fe_raw_data` JSON 예시는 다음과 같다.

```json
{
  "foot_type": "right",
  "label": "waiting",
  "speed": 0.620487853490539,
  "tilt": 8.66851515565807,
  "confidence": 1,
  "probabilities": [0, 0, 0]
}
```

전처리 모듈은 다음 작업을 수행한다.

- TSV 파일 로드
- `\N`, 빈 문자열, null 값 처리
- `fe_raw_data` JSON parsing
- `foot_type`, `label`, `speed`, `tilt`, `confidence`, `probabilities` flatten
- `fe_event_type`, `label` integer encoding
- `fe_timestamp` datetime index 변환
- `fall_flag = fe_event_type == "fall"` 생성
- 모델 입력 feature matrix 생성

현재 구현 파일:

- `src/data/ces_processor.py`
- `src/models/gait_lstm.py`
- `scripts/train_ces_model.py`

---

## 5. AI Algorithm Architecture

### 5.1 Current MVP Model: GaitLSTM

현재 MVP는 CES 이벤트 로그를 sliding window로 변환한 뒤 LSTM 기반 시계열 분류 모델에 입력한다.

입력 feature는 기본적으로 다음 3개를 사용한다.

```text
speed, tilt, fe_event_value
```

`event_type_encoded`는 라벨과 직접적으로 연결될 수 있어, 성능 과대평가를 막기 위해 기본 학습 feature에서 제외한다. 다만 비교 실험을 위해 `--feature-set with_event_type` 옵션으로 포함할 수 있다.

모델 구조:

```text
CES.csv
  -> JSON flatten
  -> time sort
  -> normalization
  -> sliding window tensor [batch, window, feature_dim]
  -> BiLSTM
  -> Temporal Attention
  -> gait state classifier
  -> fall risk head
```

출력:

- 보행 상태 분류: `waiting`, `drag`, `normal`
- AI 기반 fall score
- rule 기반 fall alert
- attention weights

### 5.2 Rule-based Fall Detection

낙상은 안전성이 중요한 이벤트이므로, AI 모델의 추론을 기다리지 않고 룰 기반 즉시 알림을 우선 적용한다.

```python
fall_alert = (fe_event_type == "fall") or (ai_fall_score >= threshold)
```

이 구조의 장점은 다음과 같다.

- 실제 낙상 이벤트가 수신되면 즉시 알림 가능
- AI 모델이 충분히 학습되지 않은 초기 단계에서도 안전 장치 제공
- 향후 더 많은 낙상 데이터가 확보되면 AI fall score를 보조 지표로 확장 가능

### 5.3 Future Model Roadmap

#### LSTM Autoencoder

목적:

- 정상 보행 패턴을 학습
- 재구성 오차 기반 이상 점수 산출
- 라벨이 부족한 초기 데이터 환경에서 anomaly detection 가능

예상 구조:

```text
normal gait sequence
  -> LSTM Encoder
  -> latent representation
  -> LSTM Decoder
  -> reconstruction error
  -> anomaly score
```

#### Transformer

목적:

- 일/주/월 단위 장기 보행 변화 감지
- 보행 속도 저하, 리듬 변화, 좌우 비대칭 증가 추세 분석
- 보호자/의료진용 장기 리포트 생성

예상 구조:

```text
daily gait summary sequence
  -> positional encoding
  -> Transformer encoder
  -> trend representation
  -> long-term risk score
```

#### Ensemble Inference

최종 inference는 다음 요소를 결합한다.

```text
final_risk_score =
  w1 * rule_based_score
  + w2 * LSTM_classification_score
  + w3 * autoencoder_anomaly_score
  + w4 * transformer_trend_score
```

출력은 질병 진단이 아니라 다음 형태로 제한한다.

- 보행 이상 징후
- 낙상 위험 증가
- 끌림 보행 관찰
- 좌우 비대칭 증가
- 전문 의료기관 상담 권고

---

## 6. Validation Strategy

### 6.1 Why Accuracy Alone Is Not Enough

현재 `CES.csv` 데이터는 클래스 불균형이 매우 크다.

```text
waiting    28414
drag          11
normal         4
fall           4
```

이 구조에서는 단순 accuracy가 높아도 실제 이상 보행 감지 성능을 의미하지 않는다. 따라서 Shoealls는 다음 검증 지표를 함께 사용한다.

- accuracy
- macro F1
- weighted F1
- class-wise precision
- class-wise recall
- confusion matrix
- fall rule recall
- fall AI recall

### 6.2 Current Validation Result

Sensor-only feature 기준 random split smoke validation 결과:

```text
features = speed, tilt, fe_event_value
val_acc = 0.999
val_macro_f1 = 0.333
fall_rule_recall = 1.000
fall_ai_recall = 0.000
```

해석:

- `waiting` 샘플이 압도적으로 많아 accuracy는 높게 나온다.
- `drag`, `normal` minority class는 아직 충분히 학습되지 않았다.
- `fall`은 현재 4건 수준이므로 AI 학습보다 rule-based immediate alert로 처리하는 것이 안전하다.
- 향후 CES 신청 전에는 정상 보행, 끌림 보행, 낙상 유사 이벤트 데이터를 보강해야 한다.

### 6.3 Target Validation Before CES Submission

7월 신청 전 목표:

| Metric | Target |
|---|---:|
| Fall rule recall | 100% 유지 |
| Drag recall | 80% 이상 |
| Normal recall | 80% 이상 |
| Macro F1 | 0.75 이상 |
| Inference latency | 100ms 이하 |
| Dashboard alert latency | 1초 이하 |

단, 위 목표는 내부 기술 목표이며 의료적 진단 성능으로 표현하지 않는다.

---

## 7. Platform Architecture

```text
Smart Shoes
  -> BLE 5.0
  -> Gateway
  -> Event Parser
  -> Rule-based Fall Alert
  -> AI Inference Server
  -> Time-series Storage
  -> FastAPI
  -> Dashboard / Mobile App / Caregiver Alert
```

현재 구현:

- FastAPI backend
- Next.js dashboard
- mock sensor data fallback
- CES preprocessing pipeline
- GaitLSTM training script
- rule-based fall alert logic

향후 확장:

- Kafka producer/consumer
- InfluxDB time-series storage
- AES-256 encrypted data storage
- WebSocket/SSE real-time dashboard
- FCM or SMS caregiver alert
- model registry and model versioning

---

## 8. Healthcare Dashboard

대시보드는 다음 정보를 제공한다.

사용자 화면:

- 오늘의 보행 점수
- 낙상 위험 알림
- 보행 리듬
- 좌우 대칭성
- 체중 이동 안정성

보호자 화면:

- 실시간 위험 알림
- 최근 보행 변화
- 낙상 이벤트 기록
- 위험도 상승 추세

관리자/의료진 화면:

- 사용자별 위험도 목록
- 일/주/월 추세
- 이상 징후 필터링
- 사용자별 보행 리포트

---

## 9. ESG and Human Impact

### 9.1 Human Security and Aging Society

Shoealls는 고령자의 안전한 이동성과 독립 생활을 지원한다. 낙상 위험을 조기에 감지하고 보호자에게 알림을 제공함으로써, 고령자와 가족의 불안을 줄이고 돌봄 부담을 완화할 수 있다.

사회적 기여:

- 고령자 낙상 위험 조기 인지
- 독거노인 및 이동 취약자 안전 모니터링
- 보호자 돌봄 부담 완화
- 재활 및 지역사회 건강관리 효율 향상
- 병원 방문 전 위험 징후 추적

### 9.2 ESG Value

Shoealls의 ESG 가치는 직접적인 탄소 감축 장치라기보다 예방 중심 헬스케어와 원격 모니터링 효율화에서 나온다.

가능한 ESG 설명:

- 원격 보행 모니터링을 통한 불필요한 이동 감소
- 장기 추세 분석을 통한 예방 중심 관리
- 보호자와 의료진의 관리 효율 개선
- BLE 기반 저전력 데이터 송신
- 장기 사용 가능한 신발형 웨어러블 구조
- 부품 교체형 센서 모듈 설계 가능성

탄소배출 관련 주장은 실측 LCA 데이터가 확보되기 전까지 정량 수치로 과장하지 않는다.

---

## 10. Privacy, Security, and Safety

Shoealls는 보행 데이터를 건강 관련 민감 데이터로 간주한다. 향후 상용화 단계에서는 다음 보안 구조를 적용한다.

- 사용자 식별자 pseudonymization
- 전송 구간 TLS
- 저장 구간 AES-256 암호화
- 접근 권한 분리: 사용자, 보호자, 관리자, 의료진
- audit logging
- 데이터 보존 기간 정책
- 비식별 데이터 기반 모델 학습

의료 안전 원칙:

- 질병 확정 진단 표현 금지
- 위험 징후와 상담 권고 중심 표현
- AI confidence 및 불확실성 표시
- 고위험 이벤트는 보호자/의료진 확인 workflow 제공

---

## 11. Competitive Differentiation

| Category | Conventional Approach | Shoealls Approach |
|---|---|---|
| 측정 환경 | 병원/실험실 중심 | 일상 착용 신발 기반 |
| 데이터 유형 | 단발성 검사 | 연속 시계열 보행 데이터 |
| 낙상 대응 | 사후 신고 중심 | 이벤트 기반 즉시 알림 |
| 보호자 연결 | 수동 확인 | 대시보드 및 알림 |
| AI 구조 | 단일 모델 | rule + LSTM + future Autoencoder/Transformer ensemble |
| 사회적 가치 | 진료 보조 | 고령자 안전, 독립 생활, 돌봄 부담 완화 |

---

## 12. CES Innovation Award Positioning

추천 출품 카테고리:

1. Digital Health
2. Artificial Intelligence
3. Accessibility & Longevity
4. Human Security for All

핵심 심사 메시지:

> Shoealls transforms everyday walking into a real-time health monitoring signal by combining smart shoe sensors, time-series AI, and caregiver-facing healthcare dashboards.

한국어 핵심 메시지:

> 슈올즈는 신발을 신는 일상 행위를 실시간 보행 헬스케어 데이터로 전환하여, 고령자의 낙상 위험과 보행 이상 징후를 조기에 확인할 수 있게 하는 멀티모달 AI 스마트슈즈 플랫폼이다.

---

## 13. Development Roadmap Before July Submission

### May

- CES 기술 백서 작성
- 대시보드 UI 정리
- CES.csv 기반 데이터 품질 리포트 작성
- LSTM 검증 결과 정리
- 제품 렌더링 및 센서 구조도 초안 제작

### June

- 정상/끌림/낙상 이벤트 데이터 보강
- 사용자별 또는 세션별 validation split 도입
- Autoencoder anomaly score prototype
- 실시간 alert demo video 제작
- 영문 application draft 작성

### July

- CES 신청 카테고리 확정
- 영문 백서 및 신청서 제출본 정리
- 제품 이미지 3장 이상 준비
- 3분 데모 영상 제출본 제작
- 심사용 기술 Q&A 문서 작성

---

## 14. Risk and Mitigation

| Risk | Current Status | Mitigation |
|---|---|---|
| 데이터 불균형 | waiting 과다, fall/drag 부족 | 추가 수집, augmentation, class-wise evaluation |
| 낙상 AI 학습 부족 | fall 4건 수준 | rule-based immediate alert 우선 적용 |
| 의료 진단 오해 | 질병명 사용 시 위험 | 위험 징후, 이상 패턴, 상담 권고 표현 |
| 탄소 감축 과장 | LCA 미확보 | 정성 ESG 중심, 정량 주장은 보류 |
| 실시간성 검증 부족 | local demo 중심 | BLE gateway latency, dashboard alert latency 측정 |

---

## 15. Conclusion

Shoealls는 스마트슈즈 센서, 시계열 AI, 룰 기반 안전 알림, 보호자/의료진 대시보드를 결합한 예방 중심 보행 헬스케어 플랫폼이다. 현재 MVP는 CES 이벤트 데이터를 기반으로 보행 상태 분류와 낙상 즉시 알림 구조를 구현했으며, 실제 데이터 파이프라인이 end-to-end로 동작함을 확인했다.

향후 CES Innovation Award 신청 전까지 데이터 균형 개선, 검증 지표 보강, 실시간 데모 영상, ESG 및 인류공헌 메시지를 강화하면 Digital Health, AI, Accessibility & Longevity 영역에서 경쟁력 있는 출품 패키지를 구성할 수 있다.

