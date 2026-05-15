"""Tests for the personalized gait analysis system."""

import numpy as np
import pytest

from src.analysis.foot_zones import FootZoneAnalyzer, ZONE_DEFINITIONS
from src.analysis.gait_profile import PersonalGaitProfiler
from src.analysis.injury_risk import InjuryRiskEngine
from src.analysis.feedback import CorrektiveFeedbackGenerator
from src.analysis.trend_tracker import LongitudinalTrendTracker
from src.analysis.prodromal_biomarkers import ProdrOmalBiomarkerExtractor, ProdromalPanel


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

    def test_feedback_with_prodromal(self):
        engine = InjuryRiskEngine(16, 8)
        seq_p = make_pressure_sequence(T=64)
        injury_report = engine.assess_risk(seq_p)

        # Create abnormal prodromal panel
        extractor = ProdrOmalBiomarkerExtractor(fs=100.0)
        # Use random data that will likely be abnormal or just mock it
        imu = np.random.randn(6, 500).astype(np.float32)
        # Add a 5Hz tremor to trigger abnormality
        t = np.arange(500) / 100.0
        imu[0] += 0.5 * np.sin(2 * np.pi * 5.0 * t)

        pres = np.random.rand(500, 16, 8).astype(np.float32)
        panel = extractor.extract(imu, pres, external={"olfactory_screen_score": 0.4})

        gen = CorrektiveFeedbackGenerator()
        feedback = gen.generate(injury_report, prodromal_panel=panel)

        assert "전임상 신경계 신호 분석" in feedback.report_kr
        assert "위험 단계" in feedback.report_kr
        assert any(item.category == "medical" for item in feedback.items)
        assert any("후각" in item.title for item in feedback.items)


class TestTrendTracker:
    def test_weekly_summary(self):
        import pandas as pd
        analyzer = LongitudinalTrendTracker()
        df = pd.DataFrame({
            'fe_timestamp': pd.date_range(start='2026-05-01', periods=10, freq='D'),
            'speed': np.random.rand(10),
            'tilt': np.random.rand(10),
            'fe_event_value': np.random.rand(10)
        })
        summary = analyzer.get_weekly_summary(df)
        assert len(summary) <= 7
        if summary:
            assert "date" in summary[0]
            assert "speed" in summary[0]
            assert "stability" in summary[0]

    def test_anomalous_trends(self):
        import pandas as pd
        analyzer = LongitudinalTrendTracker()
        # Create a decreasing speed trend with enough samples
        speed = np.concatenate([np.ones(100), np.linspace(1.0, 0.5, 100)])
        df = pd.DataFrame({
            'speed': speed,
        })
        findings = analyzer.detect_anomalous_trends(df)
        assert len(findings) > 1 or len(findings) == 1
        assert "감소" in findings[0]
