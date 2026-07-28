"""Datasets and adapters for HAR-style wearable time-series experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split


LABEL_COLUMNS = ("label", "labels", "target", "targets", "y", "class", "activity")
FEATURE_COLUMNS = ("x", "X", "features", "feature", "values", "series", "signal", "data")


@dataclass(frozen=True)
class HARDSpec:
    """Small descriptor used to keep cross-dataset experiments explicit."""

    name: str
    channels: int
    sequence_length: int
    num_classes: int
    sampling_hz: float | None = None


class HARWindowDataset(Dataset):
    """Windowed sensor dataset for human activity recognition.

    Samples are normalized to ``(channels, sequence_length)`` so they can be
    shared by PAMAP2, WISDM, UCI-HAR, and ShoeAlls local windows.
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        sequence_length: int | None = None,
        normalize: bool = True,
    ) -> None:
        if len(windows) != len(labels):
            raise ValueError("windows and labels must have the same length")
        if len(windows) == 0:
            raise ValueError("dataset is empty")

        self.sequence_length = sequence_length
        self.normalize = normalize
        self.windows = [
            prepare_har_window(window, sequence_length, normalize)
            for window in windows
        ]
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "sensor": torch.from_numpy(self.windows[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def prepare_har_window(
    window: np.ndarray,
    sequence_length: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Convert one sensor window to ``float32`` ``(channels, time)``."""
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)

    # Most public HAR sets store windows as (time, channels). If the opposite
    # is already true, keep it. This heuristic works for common C <= 64, T >= 20.
    if arr.shape[0] >= arr.shape[1]:
        arr = arr.T

    if sequence_length is not None:
        arr = resize_time_axis(arr, sequence_length)

    if normalize:
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True)
        arr = (arr - mean) / np.maximum(std, 1e-6)

    return arr.astype(np.float32, copy=False)


def resize_time_axis(window: np.ndarray, sequence_length: int) -> np.ndarray:
    """Pad, truncate, or linearly resample the time axis."""
    channels, current_length = window.shape
    if current_length == sequence_length:
        return window
    if current_length < 2:
        return np.repeat(window, sequence_length, axis=1)

    old_x = np.linspace(0.0, 1.0, current_length)
    new_x = np.linspace(0.0, 1.0, sequence_length)
    resized = np.empty((channels, sequence_length), dtype=np.float32)
    for channel in range(channels):
        resized[channel] = np.interp(new_x, old_x, window[channel])
    return resized


def split_dataset(
    dataset: Dataset,
    train_split: float,
    val_split: float,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Create reproducible train/val/test splits."""
    if train_split <= 0 or val_split < 0 or train_split + val_split >= 1:
        raise ValueError("splits must leave a positive test split")

    total = len(dataset)
    train_n = int(total * train_split)
    val_n = int(total * val_split)
    test_n = total - train_n - val_n
    return random_split(
        dataset,
        [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(seed),
    )


def load_har_npz(
    path: str | Path,
    sequence_length: int | None = None,
    normalize: bool = True,
) -> HARWindowDataset:
    """Load local ShoeAlls/HAR windows from an NPZ file.

    Expected arrays are ``windows``/``labels`` or ``x``/``y``.
    """
    data = np.load(path, allow_pickle=True)
    window_key = "windows" if "windows" in data else "x"
    label_key = "labels" if "labels" in data else "y"
    return HARWindowDataset(
        data[window_key],
        data[label_key],
        sequence_length=sequence_length,
        normalize=normalize,
    )


def load_hf_har_dataset(
    repo_id: str,
    split: str = "train",
    sequence_length: int | None = None,
    normalize: bool = True,
    max_samples: int | None = None,
) -> HARWindowDataset:
    """Load a HAR dataset from Hugging Face Hub via the optional datasets lib."""
    try:
        return _load_hf_har_arrays(
            repo_id=repo_id,
            sequence_length=sequence_length,
            normalize=normalize,
            max_samples=max_samples,
        )
    except Exception as exc:
        if exc.__class__.__name__ not in {
            "EntryNotFoundError",
            "RemoteEntryNotFoundError",
            "LocalEntryNotFoundError",
            "FileNotFoundError",
        }:
            raise
        pass

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the optional dependency with `pip install datasets` "
            "to load Hugging Face HAR datasets."
        ) from exc

    hf_dataset = load_dataset(repo_id, split=split)
    if max_samples is not None:
        hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

    windows: list[np.ndarray] = []
    labels: list[int] = []
    for row in hf_dataset:
        window, label = _extract_har_row(row)
        windows.append(window)
        labels.append(label)

    labels_arr = _encode_labels(np.asarray(labels))
    return HARWindowDataset(
        np.asarray(windows, dtype=object),
        labels_arr,
        sequence_length=sequence_length,
        normalize=normalize,
    )


def _load_hf_har_arrays(
    repo_id: str,
    sequence_length: int | None,
    normalize: bool,
    max_samples: int | None,
) -> HARWindowDataset:
    """Load MONSTER-style ``*_X.npy``/``*_y.npy`` files directly from Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install `huggingface_hub` or `datasets` to load HF HAR datasets."
        ) from exc

    dataset_name = repo_id.rstrip("/").split("/")[-1]
    x_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"{dataset_name}_X.npy",
    )
    y_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"{dataset_name}_y.npy",
    )
    windows = np.load(x_path, allow_pickle=True)
    labels = _encode_labels(np.asarray(np.load(y_path, allow_pickle=True)))
    if max_samples is not None:
        indices = _balanced_indices(labels, max_samples)
        windows = windows[indices]
        labels = labels[indices]
    return HARWindowDataset(
        windows,
        labels,
        sequence_length=sequence_length,
        normalize=normalize,
    )


def synthetic_har_dataset(
    num_samples: int = 96,
    channels: int = 6,
    sequence_length: int = 128,
    num_classes: int = 4,
    seed: int = 42,
) -> HARWindowDataset:
    """Generate a tiny deterministic HAR-like dataset for smoke tests."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, sequence_length, dtype=np.float32)
    windows = np.empty((num_samples, channels, sequence_length), dtype=np.float32)
    labels = np.arange(num_samples, dtype=np.int64) % num_classes

    for idx, label in enumerate(labels):
        freq = 1.0 + label
        base = np.sin(2 * np.pi * freq * t)
        channel_scale = np.linspace(0.6, 1.4, channels, dtype=np.float32)
        noise = rng.normal(0, 0.08, size=(channels, sequence_length))
        windows[idx] = channel_scale[:, None] * base[None, :] + noise

    return HARWindowDataset(windows, labels, sequence_length, normalize=True)


def _extract_har_row(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    label_key = _first_existing_key(row, LABEL_COLUMNS)
    feature_key = _first_existing_key(row, FEATURE_COLUMNS)

    if label_key is None:
        raise ValueError(f"Could not find a label column in HF row keys: {list(row)}")
    if feature_key is None:
        numbered_keys = sorted(
            (key for key in row if key.isdigit()),
            key=lambda value: int(value),
        )
        if numbered_keys:
            feature_value = [row[key] for key in numbered_keys]
        else:
            numeric_items = [
                value for key, value in row.items()
                if key != label_key and isinstance(value, (int, float, list, tuple, np.ndarray))
            ]
            if not numeric_items:
                raise ValueError(f"Could not find sensor features in HF row keys: {list(row)}")
            feature_value = numeric_items[0]
    else:
        feature_value = row[feature_key]

    return np.asarray(feature_value, dtype=np.float32), row[label_key]


def _first_existing_key(row: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in row:
            return key
    lower_map = {key.lower(): key for key in row}
    for key in candidates:
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return None


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    if np.issubdtype(labels.dtype, np.integer):
        unique = np.unique(labels)
        if np.array_equal(unique, np.arange(len(unique))):
            return labels.astype(np.int64)
    _, encoded = np.unique(labels, return_inverse=True)
    return encoded.astype(np.int64)


def _balanced_indices(labels: np.ndarray, max_samples: int) -> np.ndarray:
    """Select up to ``max_samples`` while covering classes before repeats."""
    rng = np.random.default_rng(42)
    selected: list[int] = []
    classes = np.unique(labels)
    per_class = max(1, max_samples // len(classes))

    class_indices: dict[int, np.ndarray] = {}
    for cls in classes:
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        class_indices[int(cls)] = idx
        selected.extend(idx[:per_class].tolist())

    if len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=False)
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)].tolist())

    selected_arr = np.asarray(selected[:max_samples], dtype=np.int64)
    rng.shuffle(selected_arr)
    return selected_arr
