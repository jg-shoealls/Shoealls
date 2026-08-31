"""질환 특이적 보행 바이오마커 시스템 테스트"""
import pytest
import numpy as np

from src.analysis.disease_biomarkers import (
    DiseaseBiomarkerAnalyzer,
    SensorChannel,
    BiomarkerDefinition,
    DiseasePanel,
    BiomarkerMeasurement,
    DiseaseBiomarkerProfile,
    MultimodalBiomarkerReport,
)
from src.models.biomarker_net import (
    BiomarkerDiseaseNet,
    BiomarkerExtractionHead,
    DiseasePanelAttention,
    MultiScaleTemporalAnalyzer,
    BiomarkerLoss,
    BiomarkerNetConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_features():
    """Dict with all 13 base features at normal values."""
    return {
        "gait_speed": 1.2,
        "cadence": 110.0,
        "stride_regularity": 0.95,
        "step_symmetry": 0.98,
        "cop_sway": 0.008,
        "ml_variability": 0.012,
        "heel_pressure_ratio": 0.55,
        "forefoot_pressure_ratio": 0.30,
        "arch_index": 0.22,
        "pressure_asymmetry": 0.03,
        "acceleration_rms": 0.25,
        "acceleration_variability": 0.05,
        "trunk_sway": 2.0,
    }


@pytest.fixture
def parkinsons_features():
    """Features characteristic of Parkinson's disease."""
    return {
        "gait_speed": 0.7,
        "cadence": 85.0,
        "stride_regularity": 0.70,
        "step_symmetry": 0.80,
        "cop_sway": 0.025,
        "ml_variability": 0.035,
        "heel_pressure_ratio": 0.40,
        "forefoot_pressure_ratio": 0.45,
        "arch_index": 0.28,
        "pressure_asymmetry": 0.12,
        "acceleration_rms": 0.45,
        "acceleration_variability": 0.15,
        "trunk_sway": 5.5,
    }


@pytest.fixture
def stroke_features():
    """Features characteristic of stroke."""
    return {
        "gait_speed": 0.6,
        "cadence": 75.0,
        "stride_regularity": 0.60,
        "step_symmetry": 0.65,
        "cop_sway": 0.030,
        "ml_variability": 0.040,
        "heel_pressure_ratio": 0.35,
        "forefoot_pressure_ratio": 0.50,
        "arch_index": 0.25,
        "pressure_asymmetry": 0.25,
        "acceleration_rms": 0.50,
        "acceleration_variability": 0.18,
        "trunk_sway": 7.0,
    }


@pytest.fixture
def oa_features():
    """Features characteristic of osteoarthritis."""
    return {
        "gait_speed": 0.9,
        "cadence": 95.0,
        "stride_regularity": 0.82,
        "step_symmetry": 0.85,
        "cop_sway": 0.015,
        "ml_variability": 0.020,
        "heel_pressure_ratio": 0.60,
        "forefoot_pressure_ratio": 0.25,
        "arch_index": 0.24,
        "pressure_asymmetry": 0.10,
        "acceleration_rms": 0.30,
        "acceleration_variability": 0.08,
        "trunk_sway": 3.5,
    }


@pytest.fixture
def disc_features():
    """Features characteristic of disc herniation."""
    return {
        "gait_speed": 0.85,
        "cadence": 90.0,
        "stride_regularity": 0.78,
        "step_symmetry": 0.82,
        "cop_sway": 0.018,
        "ml_variability": 0.025,
        "heel_pressure_ratio": 0.50,
        "forefoot_pressure_ratio": 0.35,
        "arch_index": 0.23,
        "pressure_asymmetry": 0.08,
        "acceleration_rms": 0.35,
        "acceleration_variability": 0.10,
        "trunk_sway": 4.5,
    }


@pytest.fixture
def analyzer():
    """Instantiate the disease biomarker analyzer."""
    return DiseaseBiomarkerAnalyzer()


@pytest.fixture
def biomarker_net_config():
    """Configuration for BiomarkerDiseaseNet."""
    return BiomarkerNetConfig(
        embed_dim=128,
        num_biomarkers=59,
        biomarker_attention_heads=4,
        num_diseases=10,
        panel_hidden_dim=64,
        temporal_scales=[3, 7, 15, 31],
        dropout=0.3,
    )


@pytest.fixture
def biomarker_net(biomarker_net_config):
    """Instantiate BiomarkerDiseaseNet for testing."""
    return BiomarkerDiseaseNet(biomarker_net_config)


# ---------------------------------------------------------------------------
# TestBiomarkerDefinitions
# ---------------------------------------------------------------------------

class TestBiomarkerDefinitions:
    """Tests that verify the completeness and correctness of biomarker definitions."""

    def test_all_59_biomarkers_defined(self, analyzer):
        """Verify that the system defines exactly 59 biomarkers (13 base + 46 disease-specific)."""
        all_markers = analyzer.get_all_biomarker_definitions()
        assert len(all_markers) == 59, (
            f"Expected 59 biomarkers, got {len(all_markers)}"
        )

    def test_neurological_panel_markers(self, analyzer):
        """Verify neurological panel has 6 diseases with correct marker counts."""
        neuro = analyzer.get_panel_definitions("neurological")
        expected_diseases = {
            "parkinsons", "dementia", "cerebellar_ataxia",
            "multiple_sclerosis", "stroke", "cerebrovascular",
        }
        assert set(neuro.keys()) == expected_diseases
        # Marker counts per disease
        assert len(neuro["parkinsons"]) == 5
        assert len(neuro["dementia"]) == 5
        assert len(neuro["cerebellar_ataxia"]) == 5
        assert len(neuro["multiple_sclerosis"]) == 5
        assert len(neuro["stroke"]) == 4
        assert len(neuro["cerebrovascular"]) == 2

    def test_musculoskeletal_panel_markers(self, analyzer):
        """Verify musculoskeletal panel has 4 diseases with correct marker counts."""
        msk = analyzer.get_panel_definitions("musculoskeletal")
        expected_diseases = {
            "osteoarthritis", "rheumatoid_arthritis",
            "disc_herniation", "spinal_stenosis",
        }
        assert set(msk.keys()) == expected_diseases
        assert len(msk["osteoarthritis"]) == 5
        assert len(msk["rheumatoid_arthritis"]) == 4
        assert len(msk["disc_herniation"]) == 5
        assert len(msk["spinal_stenosis"]) == 5

    def test_biomarker_normal_ranges_valid(self, analyzer):
        """All biomarkers must have normal_range where low < high."""
        for marker in analyzer.get_all_biomarker_definitions():
            low, high = marker.normal_range
            assert low < high, (
                f"Biomarker '{marker.name}' has invalid normal_range: "
                f"[{low}, {high}]"
            )

    def test_sensor_channel_assignments(self, analyzer):
        """Each biomarker must have a valid primary sensor channel."""
        valid_sensors = {
            SensorChannel.IMU,
            SensorChannel.PRESSURE,
            SensorChannel.SKELETON,
            SensorChannel.FUSION,
        }
        for marker in analyzer.get_all_biomarker_definitions():
            assert marker.primary_sensor in valid_sensors, (
                f"Biomarker '{marker.name}' has invalid sensor: "
                f"{marker.primary_sensor}"
            )

    def test_sensitivity_specificity_bounds(self, analyzer):
        """Sensitivity and specificity must be in [0, 1]."""
        for marker in analyzer.get_all_biomarker_definitions():
            assert 0.0 <= marker.sensitivity <= 1.0, (
                f"Biomarker '{marker.name}' sensitivity out of bounds: "
                f"{marker.sensitivity}"
            )
            assert 0.0 <= marker.specificity <= 1.0, (
                f"Biomarker '{marker.name}' specificity out of bounds: "
                f"{marker.specificity}"
            )

    def test_no_duplicate_marker_ids(self, analyzer):
        """All biomarker names/IDs must be unique across the entire system."""
        all_markers = analyzer.get_all_biomarker_definitions()
        names = [m.name for m in all_markers]
        assert len(names) == len(set(names)), (
            f"Duplicate biomarker names found: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_korean_names_present(self, analyzer):
        """All disease-specific biomarkers must have Korean names."""
        for marker in analyzer.get_all_biomarker_definitions():
            if marker.korean is not None:
                assert len(marker.korean) > 0, (
                    f"Biomarker '{marker.name}' has an empty Korean name"
                )


# ---------------------------------------------------------------------------
# TestBiomarkerComputation
# ---------------------------------------------------------------------------

class TestBiomarkerComputation:
    """Tests for individual biomarker computation logic."""

    def test_compute_festination_index_normal(self, analyzer, normal_features):
        """Festination index should be low for normal gait."""
        result = analyzer.compute_biomarker("festination_index", normal_features)
        assert isinstance(result, BiomarkerMeasurement)
        assert result.value == pytest.approx(result.value, abs=1e-6)
        assert result.value < 0.15, (
            f"Normal gait festination_index should be < 0.15, got {result.value}"
        )

    def test_compute_festination_index_parkinsons(self, analyzer, parkinsons_features):
        """Festination index should be elevated for Parkinson's-like gait."""
        result = analyzer.compute_biomarker("festination_index", parkinsons_features)
        assert result.value > 0.15, (
            f"Parkinson's festination_index should be > 0.15, got {result.value}"
        )

    def test_compute_tremor_power_normal(self, analyzer, normal_features):
        """Tremor power ratio should be near zero for normal gait."""
        result = analyzer.compute_biomarker("tremor_power_ratio", normal_features)
        assert result.value == pytest.approx(0.0, abs=0.06), (
            f"Normal tremor_power_ratio should be near 0, got {result.value}"
        )

    def test_compute_circumduction_stroke(self, analyzer, stroke_features):
        """Circumduction angle should be elevated for stroke gait."""
        result = analyzer.compute_biomarker("circumduction_angle", stroke_features)
        assert result.value > 5.0, (
            f"Stroke circumduction_angle should be > 5.0, got {result.value}"
        )

    def test_compute_weight_bearing_asymmetry(self, analyzer, stroke_features):
        """Weight-bearing asymmetry should be pronounced for stroke."""
        result = analyzer.compute_biomarker(
            "weight_bearing_asymmetry", stroke_features
        )
        assert result.value < 0.90, (
            f"Stroke weight_bearing_asymmetry should be < 0.90, got {result.value}"
        )

    def test_compute_antalgic_score_oa(self, analyzer, oa_features):
        """Antalgic score should be elevated for osteoarthritis."""
        result = analyzer.compute_biomarker("antalgic_score", oa_features)
        assert result.value > 0.10, (
            f"OA antalgic_score should be > 0.10, got {result.value}"
        )

    def test_compute_trunk_lateral_shift_disc(self, analyzer, disc_features):
        """Trunk lateral shift should be elevated for disc herniation."""
        result = analyzer.compute_biomarker("trunk_lateral_shift", disc_features)
        assert result.value > 2.0, (
            f"Disc trunk_lateral_shift should be > 2.0 degrees, got {result.value}"
        )

    def test_compute_flexion_preference_stenosis(self, analyzer, disc_features):
        """Flexion preference index should be measurable from features."""
        result = analyzer.compute_biomarker(
            "flexion_preference_index", disc_features
        )
        assert isinstance(result, BiomarkerMeasurement)
        assert result.value >= 0.0

    def test_all_biomarkers_computable_from_features(self, analyzer, normal_features):
        """All 59 biomarkers should return a valid measurement from base features."""
        all_markers = analyzer.get_all_biomarker_definitions()
        for marker in all_markers:
            result = analyzer.compute_biomarker(marker.name, normal_features)
            assert isinstance(result, BiomarkerMeasurement), (
                f"Biomarker '{marker.name}' did not return a BiomarkerMeasurement"
            )
            assert np.isfinite(result.value), (
                f"Biomarker '{marker.name}' returned non-finite value: {result.value}"
            )

    def test_biomarker_values_bounded(self, analyzer, normal_features):
        """Computed biomarker values should be finite and within a reasonable range."""
        all_markers = analyzer.get_all_biomarker_definitions()
        for marker in all_markers:
            result = analyzer.compute_biomarker(marker.name, normal_features)
            assert -1e6 < result.value < 1e6, (
                f"Biomarker '{marker.name}' value {result.value} is out of "
                f"reasonable bounds"
            )


# ---------------------------------------------------------------------------
# TestDiseasePanelAnalysis
# ---------------------------------------------------------------------------

class TestDiseasePanelAnalysis:
    """Tests for disease panel scoring and differential diagnosis."""

    def test_parkinsons_panel_detection(self, analyzer, parkinsons_features):
        """Parkinson's panel should score highest for parkinsons-like gait."""
        profile = analyzer.analyze(parkinsons_features)
        assert isinstance(profile, DiseaseBiomarkerProfile)
        scores = profile.disease_scores
        assert "parkinsons" in scores
        assert scores["parkinsons"] > 0.5, (
            f"Expected parkinsons score > 0.5, got {scores['parkinsons']}"
        )

    def test_dementia_panel_detection(self, analyzer, normal_features):
        """Dementia panel should have a measurable score given any features."""
        dementia_features = dict(normal_features)
        dementia_features["gait_speed"] = 0.6
        dementia_features["stride_regularity"] = 0.65
        dementia_features["cadence"] = 80.0
        profile = analyzer.analyze(dementia_features)
        assert "dementia" in profile.disease_scores

    def test_ataxia_panel_detection(self, analyzer, normal_features):
        """Cerebellar ataxia panel should respond to lateral instability."""
        ataxia_features = dict(normal_features)
        ataxia_features["ml_variability"] = 0.08
        ataxia_features["cop_sway"] = 0.05
        ataxia_features["trunk_sway"] = 8.0
        profile = analyzer.analyze(ataxia_features)
        assert "cerebellar_ataxia" in profile.disease_scores
        assert profile.disease_scores["cerebellar_ataxia"] > 0.3

    def test_stroke_panel_detection(self, analyzer, stroke_features):
        """Stroke panel should score high for asymmetric gait patterns."""
        profile = analyzer.analyze(stroke_features)
        assert "stroke" in profile.disease_scores
        assert profile.disease_scores["stroke"] > 0.5, (
            f"Expected stroke score > 0.5, got {profile.disease_scores['stroke']}"
        )

    def test_oa_panel_detection(self, analyzer, oa_features):
        """Osteoarthritis panel should detect joint loading issues."""
        profile = analyzer.analyze(oa_features)
        assert "osteoarthritis" in profile.disease_scores
        assert profile.disease_scores["osteoarthritis"] > 0.3

    def test_disc_panel_detection(self, analyzer, disc_features):
        """Disc herniation panel should detect trunk shift and avoidance."""
        profile = analyzer.analyze(disc_features)
        assert "disc_herniation" in profile.disease_scores
        assert profile.disease_scores["disc_herniation"] > 0.3

    def test_normal_low_scores(self, analyzer, normal_features):
        """All disease panels should score low for normal gait."""
        profile = analyzer.analyze(normal_features)
        for disease, score in profile.disease_scores.items():
            assert score < 0.3, (
                f"Disease '{disease}' score {score} too high for normal gait"
            )

    def test_differential_diagnosis(self, analyzer, parkinsons_features):
        """Differential diagnosis should return ranked alternatives."""
        profile = analyzer.analyze(parkinsons_features)
        differential = profile.differential_diagnosis
        assert isinstance(differential, list)
        assert len(differential) <= 3
        if len(differential) > 1:
            # Scores should be in descending order
            scores = [d["score"] for d in differential]
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# TestMultimodalReport
# ---------------------------------------------------------------------------

class TestMultimodalReport:
    """Tests for multimodal biomarker report generation."""

    def test_report_generation(self, analyzer, parkinsons_features):
        """Report should be generated without errors."""
        profile = analyzer.analyze(parkinsons_features)
        report = analyzer.generate_report(profile)
        assert isinstance(report, MultimodalBiomarkerReport)
        assert report.summary is not None
        assert len(report.summary) > 0

    def test_report_contains_korean(self, analyzer, parkinsons_features):
        """Report should include Korean language content."""
        profile = analyzer.analyze(parkinsons_features)
        report = analyzer.generate_report(profile)
        # Check that at least some Korean characters are present
        has_korean = any("\uac00" <= ch <= "\ud7a3" for ch in report.summary)
        assert has_korean, "Report summary should contain Korean characters"

    def test_sensor_contribution_analysis(self, analyzer, stroke_features):
        """Report should include per-sensor contribution analysis."""
        profile = analyzer.analyze(stroke_features)
        report = analyzer.generate_report(profile)
        assert report.sensor_contributions is not None
        assert isinstance(report.sensor_contributions, dict)
        # At least IMU and pressure should be represented
        assert len(report.sensor_contributions) >= 2

    def test_overall_health_score_bounded(self, analyzer, normal_features):
        """Overall health score must be between 0 and 1."""
        profile = analyzer.analyze(normal_features)
        report = analyzer.generate_report(profile)
        assert 0.0 <= report.overall_health_score <= 1.0, (
            f"Health score {report.overall_health_score} out of [0, 1]"
        )

    def test_top_diseases_filtered(self, analyzer, parkinsons_features):
        """Report should list only diseases above the minimum threshold."""
        profile = analyzer.analyze(parkinsons_features)
        report = analyzer.generate_report(profile)
        min_score = 0.15
        for entry in report.top_diseases:
            assert entry["score"] >= min_score, (
                f"Disease '{entry['name']}' score {entry['score']} below "
                f"minimum threshold {min_score}"
            )


# ---------------------------------------------------------------------------
# TestBiomarkerNet
# ---------------------------------------------------------------------------

class TestBiomarkerNet:
    """Tests for the BiomarkerDiseaseNet neural network components."""

    def test_biomarker_extraction_head_shape(self, biomarker_net_config):
        """BiomarkerExtractionHead should output (batch, num_biomarkers)."""
        head = BiomarkerExtractionHead(
            embed_dim=biomarker_net_config.embed_dim,
            num_biomarkers=biomarker_net_config.num_biomarkers,
            num_heads=biomarker_net_config.biomarker_attention_heads,
        )
        batch_size = 4
        seq_len = 32
        x = np.random.randn(batch_size, seq_len, biomarker_net_config.embed_dim)
        import torch

        x_t = torch.tensor(x, dtype=torch.float32)
        out = head(x_t)
        assert out.shape == (batch_size, biomarker_net_config.num_biomarkers), (
            f"Expected shape ({batch_size}, {biomarker_net_config.num_biomarkers}), "
            f"got {out.shape}"
        )

    def test_disease_panel_attention_shape(self, biomarker_net_config):
        """DiseasePanelAttention should output (batch, num_diseases)."""
        panel = DiseasePanelAttention(
            num_biomarkers=biomarker_net_config.num_biomarkers,
            num_diseases=biomarker_net_config.num_diseases,
            hidden_dim=biomarker_net_config.panel_hidden_dim,
        )
        batch_size = 4
        import torch

        biomarkers = torch.randn(batch_size, biomarker_net_config.num_biomarkers)
        out = panel(biomarkers)
        assert out.shape == (batch_size, biomarker_net_config.num_diseases), (
            f"Expected shape ({batch_size}, {biomarker_net_config.num_diseases}), "
            f"got {out.shape}"
        )

    def test_multi_scale_temporal_shape(self, biomarker_net_config):
        """MultiScaleTemporalAnalyzer should preserve batch and feature dims."""
        temporal = MultiScaleTemporalAnalyzer(
            embed_dim=biomarker_net_config.embed_dim,
            scales=biomarker_net_config.temporal_scales,
        )
        batch_size = 4
        seq_len = 128
        import torch

        x = torch.randn(batch_size, seq_len, biomarker_net_config.embed_dim)
        out = temporal(x)
        assert out.shape[0] == batch_size
        assert out.shape[2] == biomarker_net_config.embed_dim, (
            f"Feature dim should be {biomarker_net_config.embed_dim}, "
            f"got {out.shape[2]}"
        )

    def test_full_model_forward_pass(self, biomarker_net):
        """Full model forward pass should produce disease scores and biomarkers."""
        batch_size = 2
        seq_len = 128
        import torch

        x = torch.randn(batch_size, seq_len, 128)
        output = biomarker_net(x)
        assert "disease_scores" in output
        assert "biomarker_values" in output
        assert output["disease_scores"].shape == (batch_size, 10)
        assert output["biomarker_values"].shape == (batch_size, 59)

    def test_biomarker_loss_computation(self, biomarker_net):
        """BiomarkerLoss should return a scalar loss value."""
        import torch

        criterion = BiomarkerLoss(
            disease_weight=1.0,
            biomarker_weight=0.5,
            routing_weight=0.1,
            consistency_weight=0.3,
        )
        batch_size = 4
        predictions = {
            "disease_scores": torch.randn(batch_size, 10),
            "biomarker_values": torch.randn(batch_size, 59),
        }
        targets = {
            "disease_labels": torch.randint(0, 10, (batch_size,)),
            "biomarker_targets": torch.randn(batch_size, 59),
        }
        loss = criterion(predictions, targets)
        assert loss.dim() == 0, "Loss should be a scalar tensor"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_interpretability_map(self, biomarker_net):
        """Model should produce an interpretability/attention map."""
        batch_size = 2
        seq_len = 128
        import torch

        x = torch.randn(batch_size, seq_len, 128)
        output = biomarker_net(x, return_attention=True)
        assert "attention_map" in output, (
            "Forward pass with return_attention=True should include 'attention_map'"
        )
        attn = output["attention_map"]
        # Attention map should be non-negative
        assert (attn >= 0).all(), "Attention weights should be non-negative"

    def test_confidence_calibration(self, biomarker_net):
        """Model should support temperature-scaled confidence calibration."""
        batch_size = 2
        seq_len = 128
        import torch

        x = torch.randn(batch_size, seq_len, 128)
        output = biomarker_net(x)
        scores = output["disease_scores"]
        # After softmax, scores should sum to ~1 per sample
        probs = torch.softmax(scores, dim=-1)
        row_sums = probs.sum(dim=-1)
        for i in range(batch_size):
            assert row_sums[i].item() == pytest.approx(1.0, abs=1e-5), (
                f"Softmax row sum for sample {i} is {row_sums[i].item()}, "
                f"expected ~1.0"
            )
        # Temperature scaling should change the distribution entropy
        if hasattr(biomarker_net, "temperature"):
            original_temp = biomarker_net.temperature.item()
            assert original_temp > 0, "Temperature must be positive"
