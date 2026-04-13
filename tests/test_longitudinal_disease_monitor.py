"""1년 종단 질환 변화 감지 알고리즘 테스트.

검증 시나리오:
  1. 기준선 구축: 최소 세션 수, 1년 필터링, 통계 계산
  2. 변화점 감지: Z-score, CUSUM, 슬라이딩 윈도우
  3. 질환 시그니처 매칭:
     - 정상 보행 → 경고 없음
     - 파킨슨 시뮬레이션 → 파킨슨 경고
     - 뇌졸중 시뮬레이션 (급성) → 뇌출혈 경고
     - 근골격계 시뮬레이션 → 근골격계 경고
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.analysis.longitudinal_disease_monitor import (
    AdaptiveBaselineBuilder,
    ChangePointDetector,
    DiseaseSignatureMatcher,
    LongitudinalDiseaseMonitor,
    SessionRecord,
    analyze_longitudinal,
    DISEASE_SIGNATURES,
    ALERT_LEVELS,
)


# ── 도우미 함수 ─────────────────────────────────────────────────────

def _normal_gait_features(rng: np.random.RandomState) -> dict:
    """정상 보행 특성 생성."""
    return {
        "gait_speed": rng.normal(1.2, 0.1),
        "cadence": rng.normal(115, 8),
        "stride_regularity": rng.normal(0.85, 0.05),
        "step_symmetry": rng.normal(0.92, 0.03),
        "cop_sway": rng.normal(0.04, 0.01),
        "ml_variability": rng.normal(0.06, 0.015),
        "heel_pressure_ratio": rng.normal(0.32, 0.03),
        "forefoot_pressure_ratio": rng.normal(0.45, 0.04),
        "arch_index": rng.normal(0.25, 0.04),
        "pressure_asymmetry": rng.normal(0.05, 0.02),
        "acceleration_rms": rng.normal(1.5, 0.2),
        "acceleration_variability": rng.normal(0.15, 0.04),
        "trunk_sway": rng.normal(2.0, 0.4),
    }


def _generate_sessions(
    n_baseline: int,
    n_recent: int,
    baseline_factory,
    recent_factory,
    reference_date: datetime,
    rng: np.random.RandomState,
) -> list[SessionRecord]:
    """기준선 + 최근 기간의 세션 기록 생성.

    - baseline: reference_date로부터 365~30일 전 사이
    - recent: 최근 14일 이내
    """
    sessions = []
    # 기준선 세션 (30~300일 전에 분산)
    for i in range(n_baseline):
        days_ago = rng.randint(30, 300)
        ts = reference_date - timedelta(days=days_ago)
        sessions.append(SessionRecord(timestamp=ts, features=baseline_factory(rng)))

    # 최근 세션 (0~13일 이내)
    for i in range(n_recent):
        days_ago = rng.randint(0, 13)
        ts = reference_date - timedelta(days=days_ago)
        sessions.append(SessionRecord(timestamp=ts, features=recent_factory(rng)))

    return sessions


# ── 기준선 구축 테스트 ──────────────────────────────────────────────

class TestAdaptiveBaselineBuilder:
    def test_insufficient_sessions_returns_empty_baseline(self):
        builder = AdaptiveBaselineBuilder()
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)
        sessions = [
            SessionRecord(
                timestamp=ref - timedelta(days=i * 30),
                features=_normal_gait_features(rng),
            )
            for i in range(5)
        ]
        baseline = builder.build(sessions, "user1", ref)
        assert baseline.session_count == 5
        assert baseline.stats == {}

    def test_builds_baseline_with_min_sessions(self):
        builder = AdaptiveBaselineBuilder()
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)
        sessions = [
            SessionRecord(
                timestamp=ref - timedelta(days=i * 7),
                features=_normal_gait_features(rng),
            )
            for i in range(30)
        ]
        baseline = builder.build(sessions, "user1", ref)
        assert baseline.session_count == 30
        assert "gait_speed" in baseline.stats
        assert baseline.stats["gait_speed"]["n"] > 0
        assert baseline.stats["gait_speed"]["std"] > 0

    def test_filters_sessions_outside_365_days(self):
        builder = AdaptiveBaselineBuilder()
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)

        # 400일 전 세션 (필터링 대상)
        old = [
            SessionRecord(
                timestamp=ref - timedelta(days=400 + i),
                features=_normal_gait_features(rng),
            )
            for i in range(25)
        ]
        # 100일 이내 세션 (기준선 포함)
        recent = [
            SessionRecord(
                timestamp=ref - timedelta(days=i * 3),
                features=_normal_gait_features(rng),
            )
            for i in range(25)
        ]
        baseline = builder.build(old + recent, "user1", ref)
        assert baseline.session_count == 25

    def test_ewma_stats_computed(self):
        builder = AdaptiveBaselineBuilder()
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)
        sessions = [
            SessionRecord(
                timestamp=ref - timedelta(days=i * 5),
                features=_normal_gait_features(rng),
            )
            for i in range(30)
        ]
        baseline = builder.build(sessions, "user1", ref)
        assert "ewma_mean" in baseline.stats["gait_speed"]
        assert "ewma_std" in baseline.stats["gait_speed"]

    def test_outlier_removal_reduces_std(self):
        """외곽값이 평균에 미치는 영향이 제거되어야 함."""
        builder = AdaptiveBaselineBuilder()
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)

        sessions = []
        for i in range(40):
            feats = _normal_gait_features(rng)
            # 극단적 외곽값 하나 삽입
            if i == 20:
                feats["gait_speed"] = 10.0  # 비현실적 값
            sessions.append(SessionRecord(
                timestamp=ref - timedelta(days=i * 5),
                features=feats,
            ))

        baseline = builder.build(sessions, "user1", ref)
        # 외곽값 제거 후 평균이 정상 범위 내
        assert 1.0 < baseline.stats["gait_speed"]["mean"] < 1.5


# ── 변화점 감지 테스트 ──────────────────────────────────────────────

class TestChangePointDetector:
    def test_no_change_in_stable_pattern(self):
        """안정된 패턴에서는 변화 신호가 거의 없어야 함."""
        builder = AdaptiveBaselineBuilder()
        detector = ChangePointDetector()
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)

        sessions = _generate_sessions(
            n_baseline=50, n_recent=5,
            baseline_factory=_normal_gait_features,
            recent_factory=_normal_gait_features,
            reference_date=ref, rng=rng,
        )

        baseline = builder.build(sessions, "user1", ref)
        signals = detector.detect(sessions, baseline, ref)
        # 변화 있더라도 높은 점수는 없어야 함
        high_score_signals = [s for s in signals if s.aggregate_score > 0.5]
        assert len(high_score_signals) <= 1

    def test_detects_gait_speed_decrease(self):
        """보행 속도가 급격히 저하되면 감지되어야 함."""
        builder = AdaptiveBaselineBuilder()
        detector = ChangePointDetector()
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)

        def slow_gait(rng):
            f = _normal_gait_features(rng)
            f["gait_speed"] = rng.normal(0.6, 0.08)  # 현저히 저하
            return f

        sessions = _generate_sessions(
            n_baseline=50, n_recent=10,
            baseline_factory=_normal_gait_features,
            recent_factory=slow_gait,
            reference_date=ref, rng=rng,
        )

        baseline = builder.build(sessions, "user1", ref)
        signals = detector.detect(sessions, baseline, ref)

        # gait_speed가 하향 방향으로 감지되어야 함
        gs_signal = next((s for s in signals if s.feature == "gait_speed"), None)
        assert gs_signal is not None
        assert gs_signal.direction == "decrease"
        assert abs(gs_signal.z_score) > 1.5
        assert gs_signal.aggregate_score > 0.5

    def test_cusum_captures_gradual_drift(self):
        """점진적 drift가 CUSUM으로 감지되어야 함."""
        builder = AdaptiveBaselineBuilder()
        detector = ChangePointDetector()
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)

        # 최근 세션마다 점진적으로 cadence 감소
        sessions = []
        for i in range(50):
            f = _normal_gait_features(rng)
            ts = ref - timedelta(days=30 + i * 5)
            sessions.append(SessionRecord(timestamp=ts, features=f))

        for i in range(10):
            f = _normal_gait_features(rng)
            f["cadence"] -= i * 2  # 점진적 감소
            ts = ref - timedelta(days=13 - i)
            sessions.append(SessionRecord(timestamp=ts, features=f))

        baseline = builder.build(sessions, "user1", ref)
        signals = detector.detect(sessions, baseline, ref)
        cad_signal = next((s for s in signals if s.feature == "cadence"), None)
        # CUSUM 또는 sliding window가 drift를 잡아야 함
        assert cad_signal is not None


# ── 질환 시그니처 매칭 테스트 ──────────────────────────────────────

class TestDiseaseSignatureMatcher:
    def test_no_alert_for_stable_pattern(self):
        """정상 보행에서는 질환 경고가 없어야 함."""
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)

        sessions = _generate_sessions(
            n_baseline=50, n_recent=10,
            baseline_factory=_normal_gait_features,
            recent_factory=_normal_gait_features,
            reference_date=ref, rng=rng,
        )

        report = analyze_longitudinal(
            [s.to_dict() for s in sessions],
            user_id="user1",
            reference_date=ref,
        )

        # 정상 보행이므로 위험 수준 경고 없음
        severe_alerts = [
            a for a in report.disease_alerts
            if a.alert_level in ("경고", "위험")
        ]
        assert len(severe_alerts) == 0

    def test_parkinsons_signature_detected(self):
        """파킨슨 시그니처 시뮬레이션 시 경고 발생."""
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)

        def parkinsons_gait(rng):
            # 파킨슨 특징: 보행 속도↓, 규칙성↓, cadence↑, variability↑
            return {
                "gait_speed": rng.normal(0.7, 0.08),        # ↓
                "cadence": rng.normal(145, 10),              # ↑ festination
                "stride_regularity": rng.normal(0.50, 0.08), # ↓ (핵심)
                "step_symmetry": rng.normal(0.78, 0.06),     # ↓
                "cop_sway": rng.normal(0.07, 0.015),         # ↑
                "ml_variability": rng.normal(0.09, 0.02),
                "heel_pressure_ratio": rng.normal(0.30, 0.03),
                "forefoot_pressure_ratio": rng.normal(0.48, 0.04),
                "arch_index": rng.normal(0.24, 0.04),
                "pressure_asymmetry": rng.normal(0.08, 0.03),
                "acceleration_rms": rng.normal(1.0, 0.2),
                "acceleration_variability": rng.normal(0.38, 0.06),  # ↑
                "trunk_sway": rng.normal(2.8, 0.5),
            }

        sessions = _generate_sessions(
            n_baseline=50, n_recent=15,
            baseline_factory=_normal_gait_features,
            recent_factory=parkinsons_gait,
            reference_date=ref, rng=rng,
        )

        report = analyze_longitudinal(
            [s.to_dict() for s in sessions],
            user_id="user1",
            reference_date=ref,
        )

        # 파킨슨 경고가 감지되어야 함
        parkinsons = next(
            (a for a in report.disease_alerts if a.disease_id == "parkinsons"), None
        )
        assert parkinsons is not None
        assert parkinsons.alert_level in ("주의", "경고", "위험")
        assert len(parkinsons.matched_signals) >= 3

    def test_stroke_acute_signature_detected(self):
        """급성 뇌출혈/뇌경색 시그니처 감지."""
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)

        def stroke_gait(rng):
            return {
                "gait_speed": rng.normal(0.5, 0.08),         # ↓↓
                "cadence": rng.normal(85, 10),
                "stride_regularity": rng.normal(0.55, 0.08),
                "step_symmetry": rng.normal(0.55, 0.06),     # ↓↓ 비대칭 (핵심)
                "cop_sway": rng.normal(0.08, 0.02),
                "ml_variability": rng.normal(0.16, 0.03),    # ↑
                "heel_pressure_ratio": rng.normal(0.28, 0.03),
                "forefoot_pressure_ratio": rng.normal(0.50, 0.04),
                "arch_index": rng.normal(0.26, 0.04),
                "pressure_asymmetry": rng.normal(0.22, 0.04),  # ↑↑
                "acceleration_rms": rng.normal(1.1, 0.2),
                "acceleration_variability": rng.normal(0.25, 0.05),
                "trunk_sway": rng.normal(3.2, 0.6),
            }

        sessions = _generate_sessions(
            n_baseline=50, n_recent=10,
            baseline_factory=_normal_gait_features,
            recent_factory=stroke_gait,
            reference_date=ref, rng=rng,
        )

        report = analyze_longitudinal(
            [s.to_dict() for s in sessions],
            user_id="user1",
            reference_date=ref,
        )

        stroke = next(
            (a for a in report.disease_alerts
             if a.disease_id == "cerebral_hemorrhage"), None
        )
        assert stroke is not None
        assert stroke.alert_level in ("경고", "위험")
        assert stroke.progression_type == "acute"

    def test_musculoskeletal_signature_detected(self):
        """근골격계 이상 (통증 회피 보행) 시그니처 감지."""
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)

        def msk_gait(rng):
            return {
                "gait_speed": rng.normal(0.8, 0.08),         # ↓
                "cadence": rng.normal(100, 10),
                "stride_regularity": rng.normal(0.70, 0.07),
                "step_symmetry": rng.normal(0.76, 0.05),     # ↓
                "cop_sway": rng.normal(0.05, 0.015),
                "ml_variability": rng.normal(0.08, 0.02),
                "heel_pressure_ratio": rng.normal(0.20, 0.03),  # ↓
                "forefoot_pressure_ratio": rng.normal(0.60, 0.04), # ↑
                "arch_index": rng.normal(0.28, 0.04),
                "pressure_asymmetry": rng.normal(0.18, 0.04),  # ↑
                "acceleration_rms": rng.normal(1.2, 0.2),
                "acceleration_variability": rng.normal(0.20, 0.05),
                "trunk_sway": rng.normal(2.2, 0.5),
            }

        sessions = _generate_sessions(
            n_baseline=50, n_recent=12,
            baseline_factory=_normal_gait_features,
            recent_factory=msk_gait,
            reference_date=ref, rng=rng,
        )

        report = analyze_longitudinal(
            [s.to_dict() for s in sessions],
            user_id="user1",
            reference_date=ref,
        )

        msk = next(
            (a for a in report.disease_alerts
             if a.disease_id == "musculoskeletal"), None
        )
        assert msk is not None
        assert msk.alert_level in ("주의", "경고", "위험")


# ── 메인 모니터 통합 테스트 ────────────────────────────────────────

class TestLongitudinalDiseaseMonitor:
    def test_full_pipeline_insufficient_data(self):
        monitor = LongitudinalDiseaseMonitor()
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)
        for i in range(5):
            monitor.add_session(
                _normal_gait_features(rng),
                timestamp=ref - timedelta(days=i * 7),
            )
        report = monitor.analyze(user_id="user1", reference_date=ref)
        assert report.overall_health_trend == "양호"
        assert "데이터 부족" in report.summary_kr

    def test_full_pipeline_normal_gait(self):
        monitor = LongitudinalDiseaseMonitor()
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)
        for i in range(60):
            monitor.add_session(
                _normal_gait_features(rng),
                timestamp=ref - timedelta(days=i * 5),
            )
        report = monitor.analyze(user_id="user1", reference_date=ref)
        assert report.baseline.session_count == 60
        assert len(report.baseline.stats) > 0

    def test_save_load_sessions(self, tmp_path):
        monitor = LongitudinalDiseaseMonitor()
        rng = np.random.RandomState(42)
        ref = datetime(2026, 1, 1)
        for i in range(10):
            monitor.add_session(
                _normal_gait_features(rng),
                timestamp=ref - timedelta(days=i * 3),
            )

        save_path = tmp_path / "sessions.json"
        monitor.save_sessions(save_path)

        monitor2 = LongitudinalDiseaseMonitor()
        monitor2.load_sessions(save_path)
        assert len(monitor2.sessions) == 10


# ── 시그니처 정의 검증 ──────────────────────────────────────────────

class TestSignatureDefinitions:
    def test_all_diseases_have_required_fields(self):
        for disease_id, sig in DISEASE_SIGNATURES.items():
            assert "korean_name" in sig
            assert "progression" in sig
            assert sig["progression"] in ("gradual", "subacute", "acute")
            assert "feature_changes" in sig
            assert "min_features" in sig
            assert "referral" in sig
            assert "key_clinical_signs" in sig

    def test_feature_changes_have_valid_directions(self):
        for disease_id, sig in DISEASE_SIGNATURES.items():
            for feat, (direction, min_z, weight) in sig["feature_changes"].items():
                assert direction in ("increase", "decrease")
                assert 0 < min_z < 5
                assert 0 < weight <= 1.5

    def test_alert_levels_monotonic(self):
        """알림 단계가 임계값 순으로 단조 증가해야 함."""
        thresholds = [ALERT_LEVELS[lvl]["threshold"]
                      for lvl in ["관심", "주의", "경고", "위험"]]
        assert thresholds == sorted(thresholds)
