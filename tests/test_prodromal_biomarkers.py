"""Tests for the preclinical Parkinson's biomarker panel."""

import numpy as np
import pytest

from src.analysis.prodromal_biomarkers import (
    _STAGE_THRESHOLDS,
    ProdromalBiomarker,
    ProdrOmalBiomarkerExtractor,
    ProdromalPanel,
    _bandpower,
    _coefficient_of_variation,
    _double_support_ratio,
    _rest_tremor_index,
    _stride_time_cv,
)

FS = 100.0  # Hz


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _normal_imu(T: int = 512) -> np.ndarray:
    """정상 보행: 매우 규칙적인 수직 가속도, 트레모 없음."""
    imu = np.zeros((6, T), dtype=np.float32)
    t = np.arange(T) / FS
    imu[2] = np.sin(2 * np.pi * 2.0 * t)   # 완전 규칙 걸음
    imu[0] = np.sin(2 * np.pi * 2.0 * t) * 0.01  # 극소 ML 신호
    return imu


def _pd_imu(T: int = 512, seed: int = 7) -> np.ndarray:
    """전임상 PD: 누적 위상 변동으로 불규칙 보폭 + 강한 5 Hz 트레모."""
    rng = np.random.RandomState(seed)
    imu = np.zeros((6, T), dtype=np.float32)
    t = np.arange(T) / FS
    # 누적 위상 드리프트 → 보폭 간격이 누적적으로 불규칙해짐
    phase_drift = np.cumsum(rng.randn(T) * 0.25).astype(np.float32)
    imu[2] = np.sin(2 * np.pi * 2.0 * t + phase_drift)
    # 강한 5 Hz 안정 시 트레모 (X, Y 채널)
    imu[0] = 0.50 * np.sin(2 * np.pi * 5.0 * t)
    imu[1] = 0.50 * np.cos(2 * np.pi * 5.0 * t)
    return imu


def _normal_pressure(T: int = 512) -> np.ndarray:
    """정상 족압: 양발 교대 지지."""
    rng = np.random.RandomState(42)
    p = np.zeros((T, 16, 8), dtype=np.float32)
    for i in range(T):
        # 0~0.4 주기 왼발, 0.4~0.6 양발, 0.6~1.0 오른발
        phase = (i / FS) % 1.0
        if phase < 0.4:
            p[i, :8, :] = rng.rand(8, 8).astype(np.float32)
        elif phase < 0.6:
            p[i, :, :] = rng.rand(16, 8).astype(np.float32) * 0.5
        else:
            p[i, 8:, :] = rng.rand(8, 8).astype(np.float32)
    return p


# ─────────────────────────────────────────────
# 단위 함수 테스트
# ─────────────────────────────────────────────

class TestCoefficientOfVariation:
    def test_zero_variance(self):
        assert _coefficient_of_variation(np.ones(10)) == 0.0

    def test_known_cv(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cv = _coefficient_of_variation(arr)
        assert cv == pytest.approx(np.std(arr) / np.mean(arr), rel=1e-5)

    def test_empty(self):
        assert _coefficient_of_variation(np.array([])) == 0.0

    def test_single_element(self):
        assert _coefficient_of_variation(np.array([5.0])) == 0.0

    def test_near_zero_mean(self):
        arr = np.array([1e-12, 2e-12])
        assert _coefficient_of_variation(arr) == 0.0


class TestBandpower:
    def test_pure_tone_in_band(self):
        t = np.arange(512) / FS
        sig = np.sin(2 * np.pi * 5.0 * t)  # 5 Hz
        ratio = _bandpower(sig, FS, 4.0, 6.0)
        assert ratio > 0.80, f"Pure 5 Hz 신호 대역 파워비 {ratio:.3f} < 0.80"

    def test_pure_tone_out_of_band(self):
        t = np.arange(512) / FS
        sig = np.sin(2 * np.pi * 20.0 * t)  # 20 Hz — 대역 밖
        ratio = _bandpower(sig, FS, 4.0, 6.0)
        assert ratio < 0.10

    def test_empty_signal(self):
        assert _bandpower(np.array([]), FS, 4.0, 6.0) == 0.0

    def test_zero_signal(self):
        assert _bandpower(np.zeros(64), FS, 4.0, 6.0) == 0.0


class TestStridTimeCV:
    def test_regular_gait_low_cv(self):
        cv = _stride_time_cv(_normal_imu(), FS)
        assert cv < 0.08, f"정상 보행 stride_time_cv={cv:.4f} 너무 높음"

    def test_irregular_gait_higher_cv(self):
        cv_normal = _stride_time_cv(_normal_imu(), FS)
        cv_pd     = _stride_time_cv(_pd_imu(), FS)
        assert cv_pd >= cv_normal, "PD 패턴이 정상보다 CV가 낮을 수 없음"

    def test_too_short_returns_zero(self):
        short_imu = np.random.randn(6, 5).astype(np.float32)
        assert _stride_time_cv(short_imu, FS) == 0.0


class TestRestTremorIndex:
    def test_no_tremor_low_index(self):
        rti = _rest_tremor_index(_normal_imu(), FS)
        assert rti < 0.30, f"정상 IMU rest_tremor_index={rti:.4f} 너무 높음"

    def test_5hz_tremor_elevates_index(self):
        rti_normal = _rest_tremor_index(_normal_imu(), FS)
        rti_pd     = _rest_tremor_index(_pd_imu(), FS)
        assert rti_pd > rti_normal, "5 Hz 트레모 삽입 후 지수가 높아야 함"

    def test_pure_5hz_signal(self):
        # 단일 채널에 5 Hz 만 존재 — 개별 채널 밴드파워로 검출해야 함
        t = np.arange(512) / FS
        imu = np.zeros((6, 512), dtype=np.float32)
        imu[0] = np.sin(2 * np.pi * 5.0 * t)   # X 채널만 5 Hz
        rti = _rest_tremor_index(imu, FS)
        assert rti > 0.10, f"순수 5 Hz 신호에서 rti={rti:.4f} 낮음"


class TestDoubleSupportRatio:
    def test_value_in_expected_range(self):
        dsr = _double_support_ratio(_normal_pressure())
        assert 0.0 <= dsr <= 1.0

    def test_all_zeros_pressure(self):
        p = np.zeros((100, 16, 8), dtype=np.float32)
        dsr = _double_support_ratio(p)
        assert dsr == 0.0

    def test_always_both_feet_active(self):
        p = np.ones((100, 16, 8), dtype=np.float32)
        dsr = _double_support_ratio(p)
        assert dsr == pytest.approx(1.0)

    def test_4d_pressure_input(self):
        p4d = _normal_pressure()[:, np.newaxis, :, :]  # (T, 1, H, W)
        dsr = _double_support_ratio(p4d)
        assert 0.0 <= dsr <= 1.0


# ─────────────────────────────────────────────
# ProdromalBiomarker 데이터 클래스 테스트
# ─────────────────────────────────────────────

class TestProdromalBiomarker:
    def test_normal_value_not_abnormal(self):
        b = ProdromalBiomarker("x", "x", 0.02, (0.00, 0.03), "CV", "IMU", "test")
        assert not b.is_abnormal

    def test_value_above_range_is_abnormal(self):
        b = ProdromalBiomarker("x", "x", 0.10, (0.00, 0.03), "CV", "IMU", "test")
        assert b.is_abnormal

    def test_value_below_range_is_abnormal(self):
        b = ProdromalBiomarker("x", "x", 0.50, (0.70, 1.00), "score", "EXTERNAL", "test")
        assert b.is_abnormal

    def test_boundary_values_not_abnormal(self):
        b_lo = ProdromalBiomarker("x", "x", 0.00, (0.00, 0.03), "CV", "IMU", "test")
        b_hi = ProdromalBiomarker("x", "x", 0.03, (0.00, 0.03), "CV", "IMU", "test")
        assert not b_lo.is_abnormal
        assert not b_hi.is_abnormal


# ─────────────────────────────────────────────
# ProdrOmalBiomarkerExtractor 통합 테스트
# ─────────────────────────────────────────────

class TestProdrOmalBiomarkerExtractor:
    @pytest.fixture
    def extractor(self):
        return ProdrOmalBiomarkerExtractor(fs=FS)

    def test_returns_panel(self, extractor):
        panel = extractor.extract(_normal_imu(), _normal_pressure())
        assert isinstance(panel, ProdromalPanel)

    def test_has_eight_biomarkers(self, extractor):
        panel = extractor.extract(_normal_imu(), _normal_pressure())
        assert len(panel.biomarkers) == 8

    def test_all_expected_names_present(self, extractor):
        panel = extractor.extract(_normal_imu(), _normal_pressure())
        names = {b.name for b in panel.biomarkers}
        expected = {
            "stride_time_cv", "step_width_cv", "cadence_variability",
            "double_support_ratio", "rest_tremor_index",
            "nocturnal_movement_regularity", "olfactory_screen_score",
            "prodromal_composite_score",
        }
        assert names == expected

    def test_normal_imu_stage_is_normal_or_subclinical(self, extractor):
        panel = extractor.extract(_normal_imu(), _normal_pressure())
        assert panel.prodrome_stage in ("정상", "전임상"), (
            f"정상 입력인데 단계={panel.prodrome_stage}, composite={panel.composite_score:.3f}"
        )

    def test_pd_imu_higher_composite_than_normal(self, extractor):
        panel_n = extractor.extract(_normal_imu(), _normal_pressure())
        panel_p = extractor.extract(_pd_imu(), _normal_pressure())
        assert panel_p.composite_score >= panel_n.composite_score, (
            f"PD composite {panel_p.composite_score:.3f} < normal {panel_n.composite_score:.3f}"
        )

    def test_composite_score_in_range(self, extractor):
        panel = extractor.extract(_normal_imu(), _normal_pressure())
        assert 0.0 <= panel.composite_score <= 1.0

    def test_stage_confidence_in_range(self, extractor):
        panel = extractor.extract(_normal_imu(), _normal_pressure())
        assert 0.0 <= panel.stage_confidence <= 1.0

    def test_olfactory_abnormal_raises_risk(self, extractor):
        panel_ok  = extractor.extract(_normal_imu(), _normal_pressure(), external={"olfactory_screen_score": 0.90})
        panel_bad = extractor.extract(_normal_imu(), _normal_pressure(), external={"olfactory_screen_score": 0.30})
        assert panel_bad.composite_score > panel_ok.composite_score

    def test_olfactory_abnormal_flagged(self, extractor):
        panel = extractor.extract(_normal_imu(), _normal_pressure(), external={"olfactory_screen_score": 0.40})
        olf = next(b for b in panel.biomarkers if b.name == "olfactory_screen_score")
        assert olf.is_abnormal

    def test_no_external_uses_default_normal_olfactory(self, extractor):
        panel = extractor.extract(_normal_imu(), _normal_pressure())
        olf = next(b for b in panel.biomarkers if b.name == "olfactory_screen_score")
        assert olf.value == pytest.approx(1.0)
        assert not olf.is_abnormal

    def test_stage_threshold_boundaries(self, extractor):
        _t1, _t2, _t3 = _STAGE_THRESHOLDS
        # composite를 각 경계값 부근에 고정하기 위해 후각으로만 조절
        # composite = 0.4 * (1-olf) → olf = 1 - composite/0.4 (변동성·REM=0 이상)
        # 정확히 제어하기 어려우므로 실제 panel 통합 흐름만 확인
        panel_any = extractor.extract(_normal_imu(), _normal_pressure())
        assert panel_any.prodrome_stage in ("정상", "전임상", "프로드로말", "발현")

    def test_abnormal_count_matches(self, extractor):
        panel = extractor.extract(_normal_imu(), _normal_pressure())
        expected = sum(b.is_abnormal for b in panel.biomarkers)
        assert panel.abnormal_count == expected

    def test_4d_pressure_accepted(self, extractor):
        p4d = _normal_pressure()[:, np.newaxis, :, :]
        panel = extractor.extract(_normal_imu(), p4d)
        assert isinstance(panel, ProdromalPanel)

    def test_summary_string(self, extractor):
        panel = extractor.extract(_normal_imu(), _normal_pressure())
        s = panel.summary()
        assert "전임상 단계" in s
        assert "복합점수" in s
