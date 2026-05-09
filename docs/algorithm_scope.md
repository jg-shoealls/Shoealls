# Shoealls 알고리즘 범위 정의서 (v0.1.0 MVP)

본 문서는 Shoealls MVP가 **무엇을 측정하고 · 무엇을 분류하며 · 무엇을 예측하는가**를
코드-수준 정확도로 고정하기 위한 단일 기준 문서다. 숫자·명칭 변경 시 이 문서와
`src/analysis/config.py`, `configs/biomarker_analysis.yaml`을 함께 갱신한다.

---

## 1. 입력 데이터 사양

| 센서        | 형상                | 채널/설명                             | 출처                                     |
| ----------- | ------------------- | ------------------------------------- | ---------------------------------------- |
| IMU         | `[128, 6]`          | accel(x,y,z) + gyro(x,y,z)            | `config.py` `IMU_CHANNELS=6`             |
| 족저압      | `[16, 8]`           | 좌/우 통합 16행 × 8열 그리드           | `config.py` `PRESSURE_GRID_SHAPE=(16,8)` |
| 스켈레톤    | `[128, 17, 3]`      | COCO 17관절 × (x, y, z)                | `config.py` `SKELETON_JOINTS=17`         |
| 시퀀스 길이 | 128 프레임          | 128 Hz 기준 약 1.0초 윈도우 리샘플링   | `config.py` `SEQUENCE_LENGTH=128`        |

API 스키마 근거: `api/schemas.py` `SensorData`.

---

## 2. 보행 지표 (13개 base features)

`src/analysis/disease_classifier.py:58` `FEATURE_NAMES` 순서 고정. 정상 범위는
`src/analysis/config.py:45` `BIOMARKER_NORMAL_RANGES`.

### 2.1 시공간 (4)

| 지표                | 정상 범위       | 단위       | 주 센서 |
| ------------------- | --------------- | ---------- | ------- |
| `gait_speed`        | 1.0 – 1.4       | m/s        | IMU     |
| `cadence`           | 100 – 130       | steps/min  | IMU     |
| `stride_regularity` | 0.7 – 1.0       | ratio      | IMU     |
| `step_symmetry`     | 0.85 – 1.0      | ratio      | IMU     |

### 2.2 족저압 (4)

| 지표                       | 정상 범위   | 단위   |
| -------------------------- | ----------- | ------ |
| `heel_pressure_ratio`      | 0.25 – 0.40 | ratio  |
| `forefoot_pressure_ratio`  | 0.35 – 0.55 | ratio  |
| `arch_index`               | 0.15 – 0.35 | ratio  |
| `pressure_asymmetry`       | 0.0 – 0.12  | index  |

### 2.3 균형 / IMU (5)

| 지표                       | 정상 범위   | 단위    |
| -------------------------- | ----------- | ------- |
| `cop_sway`                 | 0.0 – 0.06  | normalized |
| `ml_variability`           | 0.0 – 0.10  | std     |
| `trunk_sway`               | 0.0 – 3.0   | deg/s   |
| `acceleration_rms`         | 0.8 – 2.5   | m/s²    |
| `acceleration_variability` | 0.0 – 0.35  | CV      |

---

## 3. 보행 분류 (4 classes)

`src/analysis/gait_anomaly.py`, `disease_predictor.py:132`.

| 클래스         | 한글명       | 대표 임상 징후                 |
| -------------- | ------------ | ------------------------------ |
| `normal`       | 정상 보행    | 대칭·규칙적 스텝               |
| `parkinsonian` | 파킨슨       | 가속보행(festination), 진전    |
| `antalgic`     | 절뚝거림      | 체중지지 단축, 비대칭          |
| `ataxic`       | 운동실조     | ML 변동성 증가, 경로 이탈      |

---

## 4. 질환 분류 (11 classes)

`src/analysis/disease_classifier.py:43` `DISEASE_LABELS`.
입력 = 13개 base feature 벡터, 모델 = Random Forest + Gradient Boosting 앙상블
(`CLASSIFIER_DEFAULTS`: n_estimators 100, max_depth 8, cv_folds 5).

| ID | key                    | 한글명              | 카테고리   |
| -- | ---------------------- | ------------------- | ---------- |
| 0  | `normal`               | 정상 보행           | —          |
| 1  | `parkinsons`           | 파킨슨병            | 신경계     |
| 2  | `stroke`               | 뇌졸중              | 뇌혈관계   |
| 3  | `diabetic_neuropathy`  | 당뇨 신경병증       | 대사/신경  |
| 4  | `cerebellar_ataxia`    | 소뇌 실조증         | 신경계     |
| 5  | `osteoarthritis`       | 골관절염            | 근골격계   |
| 6  | `dementia`             | 치매 (알츠하이머)   | 신경계     |
| 7  | `cerebral_hemorrhage`  | 뇌출혈              | 뇌혈관계   |
| 8  | `cerebral_infarction`  | 뇌경색              | 뇌혈관계   |
| 9  | `disc_herniation`      | 추간판 탈출증       | 근골격계   |
| 10 | `rheumatoid_arthritis` | 류마티스 관절염     | 근골격계   |

---

## 5. 질환-특이 바이오마커 (10 질환 × 45 지표)

`configs/biomarker_analysis.yaml`. 각 마커는 `normal_range`, `unit`,
`primary_sensor(imu|pressure|skeleton|fusion)`, `sensitivity`, `specificity`를
보유. 하기는 질환당 마커 수 요약:

| 질환                  | 마커 수 | 대표 지표 (예)                                       |
| --------------------- | ------- | ---------------------------------------------------- |
| 파킨슨병              | 5       | festination_index, tremor_power_ratio, fog_score     |
| 치매 (알츠하이머)     | 5       | dual_task_cost, path_deviation_index                 |
| 소뇌 실조증           | 4       | lateral_instability_index, romberg_ratio             |
| 다발성 경화증         | 4       | fatigue_gait_decline, spasticity_index               |
| 뇌졸중                | 4       | circumduction_angle, foot_drop_score, hemi_asymmetry |
| 뇌혈관 질환           | 3       | white_matter_gait_score 등                           |
| 골관절염              | 5       | antalgic_score, knee_adduction_moment               |
| 류마티스 관절염       | 4       | multi_joint_pain_index, morning_stiffness            |
| 추간판 탈출증         | 5       | trendelenburg_sign, lumbar_flexion_limit             |
| 척추관 협착증         | 6       | neurogenic_claudication_index                        |
| **합계**              | **45**  |                                                      |

> 주: `disease_classifier`의 `diabetic_neuropathy` 클래스는 현재 base 13
> 특성만으로 학습하며, 전용 바이오마커 세트는 v0.2에서 추가 예정.

---

## 6. 부상 위험 예측 (6 types)

`src/analysis/injury_risk.py:42`. 입력 = 족저압 시퀀스 파생 집계 지표.
출력 = 각 유형 0–1 위험점수 + 4단계 심각도 (`정상/주의/경고/위험`).

| #  | key                                | 한글명               | 주요 기여 지표                     |
| -- | ---------------------------------- | -------------------- | ---------------------------------- |
| 1  | `plantar_fasciitis`                | 족저근막염           | heel_pressure_ratio ↑, arch_index ↓ |
| 2  | `metatarsal_stress`                | 중족골 피로골절      | forefoot_pressure_ratio ↑          |
| 3  | `ankle_sprain`                     | 발목 염좌            | ml_variability ↑, cop_sway ↑       |
| 4  | `calcaneal_stress` (heel_spur)     | 종골 스트레스        | peak_heel_pressure ↑               |
| 5  | `flat_foot` (overpronation)        | 평발/과회내          | arch_index ↓, ml_index +           |
| 6  | `high_arch` (supination)           | 요족/과회외          | arch_index ↑, ml_index −           |

심각도 경계: `config.py:19` `SEVERITY_THRESHOLDS_4LEVEL` = 위험 0.75 / 경고 0.50
/ 주의 0.25 / 정상 < 0.25.

---

## 7. 파생 지표 및 집계 규칙

- **종합 위험도** (`combined_risk_score`): 질환 확률 · 부상 위험 · 이상 바이오
  마커 비율의 가중 결합. 상세는 `api/services.py`.
- **건강 점수 등급** (`HEALTH_SCORE_GRADES`): 양호 ≥ 85 / 보통 ≥ 70 /
  주의 ≥ 50 / 경고 < 50.
- **개인 편차 (Z-score) 등급** (`DEVIATION_THRESHOLDS`): mild 1.5σ /
  moderate 2.0σ / severe 3.0σ.
- **추세 유의 임계** (`TREND_THRESHOLD`): |기울기| ≥ 0.05.

---

## 8. MVP 범위 제외 항목 (Out of Scope)

1. 실시간 연속 모니터링 (현재는 1회성 128프레임 윈도우 분석).
2. 개별 환자 longitudinal 학습 — 개인 베이스라인은 참조값일 뿐 모델에 반영 X.
3. 스켈레톤 기반 3D 각도 운동학 (knee flexion angle 등) — base feature 미포함.
4. 영상/비디오 스트림 입력 — 센서 배열만 지원.
5. 질환 감별 진단 결과의 임상 사용 — `is_demo_mode=True` 유지 시 참고용.

---

## 9. 변경 이력

| 버전   | 날짜        | 변경                                             |
| ------ | ----------- | ------------------------------------------------ |
| 0.1.0  | 2026-04-17  | 초기 정의 (센서·13지표·4분류·11질환·6부상)       |
