"""CES.tsv/CES.csv loader and preprocessing utilities.

The source file is tab-separated and contains JSON strings in ``fe_raw_data``.
This module flattens those JSON fields and prepares numeric features for a
small time-series model. Outputs are monitoring signals, not medical diagnosis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


CES_REQUIRED_COLUMNS = {
    "fe_idx",
    "fe_device_idx",
    "fe_user_idx",
    "fe_event_type",
    "fe_event_value",
    "fe_raw_data",
    "fe_analysis_result",
    "fe_timestamp",
}

CES_FEATURE_COLUMNS = ["speed", "tilt", "fe_event_value", "event_type_encoded"]


@dataclass(frozen=True)
class CESPreprocessResult:
    dataframe: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    fall_flags: np.ndarray
    event_encoder: LabelEncoder
    label_encoder: LabelEncoder


class CESDataProcessor:
    """Load and preprocess Shoealls CES event logs.

    The expected input is a TSV-like file with ``sep="\\t"``. Values such as
    ``\\N`` and empty strings are treated as missing.
    """

    def __init__(self, file_path: str | Path, sep: str = "\t") -> None:
        self.file_path = Path(file_path)
        self.sep = sep
        self.event_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()
        self.df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"CES data file not found: {self.file_path}")

        df = pd.read_csv(
            self.file_path,
            sep=self.sep,
            na_values=["\\N", "", "null", "None"],
            keep_default_na=True,
        )
        missing = CES_REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"CES data is missing required columns: {sorted(missing)}")
        return df

    def preprocess(self) -> CESPreprocessResult:
        df = self.load().copy()

        raw = df["fe_raw_data"].apply(self._parse_raw_json)
        raw_df = pd.json_normalize(raw)
        df = pd.concat([df.reset_index(drop=True), raw_df.reset_index(drop=True)], axis=1)

        df["fe_timestamp"] = pd.to_datetime(df["fe_timestamp"], errors="coerce")
        df = df.dropna(subset=["fe_timestamp"]).sort_values("fe_timestamp").reset_index(drop=True)
        df = df.set_index("fe_timestamp", drop=False)

        df["foot_type"] = df.get("foot_type", "unknown").fillna("unknown").astype(str)
        df["label"] = df.get("label", "unknown").fillna("unknown").astype(str)
        df["fe_event_type"] = df["fe_event_type"].fillna("unknown").astype(str)

        for column in ["speed", "tilt", "confidence", "fe_event_value"]:
            df[column] = pd.to_numeric(df.get(column, 0.0), errors="coerce").fillna(0.0)

        df["event_type_encoded"] = self.event_encoder.fit_transform(df["fe_event_type"])
        df["label_encoded"] = self.label_encoder.fit_transform(df["label"])
        df["fall_flag"] = (df["fe_event_type"].str.lower() == "fall").astype(np.int64)

        df[CES_FEATURE_COLUMNS] = self._normalize_features(df[CES_FEATURE_COLUMNS])

        features = df[CES_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        labels = df["label_encoded"].to_numpy(dtype=np.int64)
        fall_flags = df["fall_flag"].to_numpy(dtype=np.int64)

        self.df = df
        return CESPreprocessResult(
            dataframe=df,
            features=features,
            labels=labels,
            fall_flags=fall_flags,
            event_encoder=self.event_encoder,
            label_encoder=self.label_encoder,
        )

    def load_and_preprocess(self) -> pd.DataFrame:
        """Backward-compatible helper used by older scripts."""
        return self.preprocess().dataframe

    def get_feature_matrix(self) -> np.ndarray:
        if self.df is None:
            self.preprocess()
        assert self.df is not None
        return self.df[CES_FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    def get_labels(self) -> np.ndarray:
        if self.df is None:
            self.preprocess()
        assert self.df is not None
        return self.df["label_encoded"].to_numpy(dtype=np.int64)

    def get_fall_flags(self) -> np.ndarray:
        if self.df is None:
            self.preprocess()
        assert self.df is not None
        return self.df["fall_flag"].to_numpy(dtype=np.int64)

    @staticmethod
    def _parse_raw_json(value: Any) -> dict[str, Any]:
        if pd.isna(value):
            return {
                "foot_type": "unknown",
                "label": "unknown",
                "speed": 0.0,
                "tilt": 0.0,
                "confidence": 0.0,
                "probabilities": [],
            }

        try:
            payload = json.loads(str(value))
        except (json.JSONDecodeError, TypeError):
            return {
                "foot_type": "unknown",
                "label": "parse_error",
                "speed": 0.0,
                "tilt": 0.0,
                "confidence": 0.0,
                "probabilities": [],
            }

        return {
            "foot_type": payload.get("foot_type", "unknown"),
            "label": payload.get("label", "unknown"),
            "speed": payload.get("speed", 0.0),
            "tilt": payload.get("tilt", 0.0),
            "confidence": payload.get("confidence", 0.0),
            "probabilities": payload.get("probabilities", []),
        }

    @staticmethod
    def _normalize_features(features: pd.DataFrame) -> pd.DataFrame:
        normalized = features.astype(float).copy()
        for column in normalized.columns:
            min_value = normalized[column].min()
            max_value = normalized[column].max()
            if np.isclose(max_value, min_value):
                normalized[column] = 0.0
            else:
                normalized[column] = (normalized[column] - min_value) / (max_value - min_value)
        return normalized
