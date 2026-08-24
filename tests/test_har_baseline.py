"""Tests for HAR/IMU baseline components."""

import torch

from src.data.har_dataset import (
    prepare_har_window,
    synthetic_har_dataset,
)
from src.models.har_transformer import PatchTSTClassifier, freeze_encoder


def test_prepare_har_window_transposes_and_resizes():
    window = torch.randn(200, 6).numpy()
    prepared = prepare_har_window(window, sequence_length=128)
    assert prepared.shape == (6, 128)
    assert prepared.dtype.name == "float32"


def test_har_window_dataset_item_shape():
    dataset = synthetic_har_dataset(num_samples=12, channels=6, sequence_length=128, num_classes=3)
    item = dataset[0]
    assert item["sensor"].shape == (6, 128)
    assert item["label"].dtype == torch.long


def test_patchtst_classifier_forward():
    model = PatchTSTClassifier(in_channels=6, num_classes=4, sequence_length=128)
    logits = model(torch.randn(2, 6, 128))
    assert logits.shape == (2, 4)
    assert model.get_num_trainable_params() == model.get_num_params()


def test_freeze_encoder_leaves_classifier_trainable():
    model = PatchTSTClassifier(in_channels=6, num_classes=4, sequence_length=128)
    freeze_encoder(model)
    assert all(not param.requires_grad for name, param in model.named_parameters() if not name.startswith("classifier"))
    assert all(param.requires_grad for name, param in model.named_parameters() if name.startswith("classifier"))
