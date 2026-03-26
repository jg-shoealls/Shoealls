"""Tests for the personalized gait analysis system."""

import numpy as np
import pytest

from src.analysis.foot_zones import FootZoneAnalyzer, ZONE_DEFINITIONS
from src.analysis.gait_profile import PersonalGaitProfiler
from src.analysis.injury_risk import InjuryRiskEngine
from src.analysis.feedback import CorrektiveFeedbackGenerator
from src.analysis.trend_tracker import LongitudinalTrendTracker


def make_pressure_sequence(T=64, H=16, W=8, seed=42):
    """Generate synthetic pressure sequence."""
    rng = np.random.RandomState(seed)
    return rng.rand(T, 1, H, W).astype(np.float32)


def make_imu_sequence(C=6, T=128, seed=42):
    """Generate synthetic IMU sequence."""
    rng = np.random.RandomState(seed)
    return rng.randn(C, T).astype(np.float32)


class TestFootZoneAnalyzer:
    def test_zone_masks_cover_grid(self):
        analyzer = FootZoneAnalyzer(16, 8)
        combined = np.zeros((16, 8), dtype=bool)
        for mask in analyzer.zone_masks.values():
            combined |= mask
        assert combined.all(), "Zone masks should cover entire grid"

    def test_zone_masks_no_overlap(self):
        analyzer = FootZoneAnalyzer(16, 8)
        total_cells = sum(m.sum() for m in analyzer.zone_masks.values())
        assert total_cells == 16 * 8, "Zones should not overlap"

    def test_analyze_frame_output(self):
        analyzer = FootZoneAnalyzer(16, 8)
        pressure = np.random.rand(16, 8)
        result = analyzer.analyze_frame(pressure)

        assert len(result.zone_metrics) == len(ZONE_DEFINITIONS)
        assert 0 <= result.cop_x <= 1
        assert 0 <= result.cop_y <= 1
        assert -1 <= result.mediolateral_index <= 1
        assert -1 <= result.anteroposterior_index <= 1
        assert result.total_pressure > 0

    def test_analyze_frame_zeros(self):
        analyzer = FootZoneAnalyzer(16, 8)
        pressure = np.zeros((16, 8))
        result = analyzer.analyze_frame(pressure)

        assert result.cop_x == 0.5
        assert result.cop_y == 0.5
        assert result.total_pressure == 0.0

    def test_analyze_sequence(self):
        analyzer = FootZoneAnalyzer(16, 8)
        seq = make_pressure_sequence(T=32)
        result = analyzer.analyze_sequence(seq)

        assert result["num_frames"] == 32
        assert result["cop_trajectory"].shape == (32, 2)
        assert result["cop_sway"] >= 0
        assert len(result["zone_temporal"]) == len(ZONE_DEFINITIONS)

    def test_3d_input(self):
        """Test that (1, H, W) input works for analyze_frame."""
        analyzer = FootZoneAnalyzer(16, 8)
        pressure = np.random.rand(1, 16, 8)
        result = analyzer.analyze_frame(pressure)
        assert result.total_pressure > 0


class TestPersonalGaitProfiler:
    def test_extract_features_pressure_only(self):
        profiler = PersonalGaitProfiler(16, 8)
        seq = make_pressure_sequence(T=64)
        features = profiler.extract_session_features(seq)

        assert "ml_index" in features
        assert "ap_index" in features
        assert "cop_sway" in features
        assert "arch_index" in features
        # Should have zone features
        assert any(k.startswith("zone_") for k in features)

    def test_extract_features_with_imu(self):
        profiler = PersonalGaitProfiler(16, 8)
        seq = make_pressure_sequence(T=128)
        imu = make_imu_sequence(C=6, T=128)
        features = profiler.extract_session_features(seq, imu)

        assert "acceleration_rms" in features
        assert "step_symmetry" in features
        assert "cadence" in features
        assert "stride_regularity" in features

    def test_baseline_update(self):
        profiler = PersonalGaitProfiler(16, 8)

        for seed in range(5):
            seq = make_pressure_sequence(T=64, seed=seed)
            features = profiler.extract_session_features(seq)
            profiler.update_baseline(features)

        assert profiler.baseline is not None
        assert profiler.baseline.num_sessions == 5

    def test_deviation_detection(self):
        profiler = PersonalGaitProfiler(16, 8)

        # Build baseline from 5 similar sessions
        for seed in range(5):
            seq = make_pressure_sequence(T=64, seed=seed)
            features = profiler.extract_session_features(seq)
            profiler.update_baseline(features)

        # Compute deviation for a new session
        seq = make_pressure_sequence(T=64, seed=99)
        features = profiler.extract_session_features(seq)
        report = profiler.compute_deviations(features)

        assert hasattr(report, "deviations")
        assert hasattr(report, "alerts")
        assert 0 <= report.overall_deviation <= 1

    def test_single_session_no_alerts(self):
        profiler = PersonalGaitProfiler(16, 8)
        seq = make_pressure_sequence(T=64)
        features = profiler.extract_session_features(seq)
        profiler.update_baseline(features)

        report = profiler.compute_deviations(features)
        # With only 1 session, should produce no alerts
        assert len(report.alerts) == 0


class TestInjuryRiskEngine:
    def test_assess_risk_output(self):
        engine = InjuryRiskEngine(16, 8)
        seq = make_pressure_sequence(T=64)
        report = engine.assess_risk(seq)

        assert len(report.risks) == 6
        assert 0 <= report.overall_risk <= 1
        assert isinstance(report.top_risk, str)
        assert isinstance(report.summary_kr, str)

    def test_all_risks_bounded(self):
        engine = InjuryRiskEngine(16, 8)
        seq = make_pressure_sequence(T=64)
        report = engine.assess_risk(seq)

        for risk in report.risks:
            assert 0 <= risk.risk_score <= 1
            assert risk.severity in ("정상", "주의", "경고", "위험")
            assert len(risk.contributing_factors) > 0
            assert isinstance(risk.recommendation, str)

    def test_risk_names(self):
        engine = InjuryRiskEngine(16, 8)
        seq = make_pressure_sequence(T=64)
        report = engine.assess_risk(seq)

        names = {r.name for r in report.risks}
        expected = {
            "plantar_fasciitis", "metatarsal_stress", "ankle_sprain",
            "heel_spur", "flat_foot", "high_arch",
        }
        assert names == expected


class TestFeedbackGenerator:
    def test_generate_feedback(self):
        engine = InjuryRiskEngine(16, 8)
        seq = make_pressure_sequence(T=64)
        injury_report = engine.assess_risk(seq)

        gen = CorrektiveFeedbackGenerator()
        feedback = gen.generate(injury_report)

        assert isinstance(feedback.report_kr, str)
        assert feedback.overall_status in ("매우 양호", "양호", "개선 권장", "주의 필요")
        assert isinstance(feedback.encouragement, str)

    def test_feedback_with_deviations(self):
        profiler = PersonalGaitProfiler(16, 8)
        for seed in range(5):
            seq = make_pressure_sequence(T=64, seed=seed)
            features = profiler.extract_session_features(seq)
            profiler.update_baseline(features)

        seq = make_pressure_sequence(T=64, seed=99)
        features = profiler.extract_session_features(seq)
        deviation = profiler.compute_deviations(features)

        engine = InjuryRiskEngine(16, 8)
        injury_report = engine.assess_risk(seq)

        gen = CorrektiveFeedbackGenerator()
        feedback = gen.generate(injury_report, deviation, profiler.baseline)

        assert isinstance(feedback.report_kr, str)
        assert "맞춤형 보행 분석 피드백 리포트" in feedback.report_kr


class TestTrendTracker:
    def test_insufficient_sessions(self):
        tracker = LongitudinalTrendTracker()
        tracker.add_session({"ml_index": 0.1}, injury_risk=0.2)
        result = tracker.analyze_trends(min_sessions=3)
        assert result.sessions_analyzed == 1
        assert "최소" in result.report_kr

    def test_trend_analysis(self):
        tracker = LongitudinalTrendTracker()

        # Simulate improving cop_sway (decreasing)
        for i in range(5):
            features = {
                "cop_sway": 0.15 - i * 0.02,
                "stride_regularity": 0.5 + i * 0.05,
                "ml_index": 0.1,
            }
            tracker.add_session(features, injury_risk=0.3 - i * 0.05)

        result = tracker.analyze_trends(min_sessions=3)

        assert result.sessions_analyzed == 5
        assert "cop_sway" in result.metric_trends
        assert "stride_regularity" in result.metric_trends
        assert isinstance(result.report_kr, str)
        assert "트렌드" in result.report_kr

    def test_trend_directions(self):
        tracker = LongitudinalTrendTracker()

        # Clear improving trend for cop_sway (lower_better, decreasing = improving)
        for i in range(6):
            tracker.add_session(
                {"cop_sway": 0.20 - i * 0.03, "stride_regularity": 0.4 + i * 0.1},
                injury_risk=0.5 - i * 0.08,
            )

        result = tracker.analyze_trends(min_sessions=3)

        # cop_sway should be improving (lower_better, slope < 0)
        if "cop_sway" in result.metric_trends:
            assert result.metric_trends["cop_sway"]["slope"] < 0
