"""
CES.csv 스키마 기반 데이터 전처리 및 PyTorch Dataset/DataLoader 생성 모듈.

실제 데이터 포맷:
  - TSV (탭 구분자)
  - Columns: fe_idx, fe_device_idx, fe_user_idx, fe_event_type,
             fe_event_value, fe_raw_data, fe_analysis_result, fe_timestamp
  - fe_event_type: 'acceleration' | 'shuffling' | 'fall'
  - fe_raw_data JSON: {foot_type, label, speed, tilt, confidence, probabilities}
  - label: 'waiting' | 'drag' | 'normal'
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ─── 인코딩 맵 ───────────────────────────────────────────────────────────────

EVENT_TYPE_MAP: dict[str, int] = {
    "acceleration": 0,
    "shuffling": 1,
    "fall": 2,
}

LABEL_MAP: dict[str, int] = {
    "normal": 0,
    "waiting": 1,
    "drag": 2,
}

FOOT_TYPE_MAP: dict[str, int] = {
    "left": 0,
    "right": 1,
}

# 역방향 맵 (추론 결과 디코딩용)
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
IDX_TO_EVENT: dict[int, str] = {v: k for k, v in EVENT_TYPE_MAP.items()}

# ─── 특성 컬럼 정의 ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    "speed",
    "tilt",
    "fe_event_value_norm",  # 정규화된 이벤트 값
    "event_type_enc",       # 인코딩된 이벤트 타입
    "foot_type_enc",        # 인코딩된 발 구분
    "confidence_norm",      # 정규화된 신뢰도
]

# ─── 데이터 전처리 클래스 ─────────────────────────────────────────────────────


@dataclass
class ProcessorConfig:
    """DataProcessor 설정"""
    sep: str = "\t"
    timestamp_col: str = "fe_timestamp"
    raw_data_col: str = "fe_raw_data"
    event_type_col: str = "fe_event_type"
    fillna_speed: float = 0.0
    fillna_tilt: float = 0.0
    fillna_confidence: int = 50
    event_value_scale: float = 100.0  # fe_event_value 정규화 스케일


class DataProcessor:
    """
    CES.csv 형식의 스마트 슈즈 데이터를 로드하고 전처리하는 클래스.

    주요 작업:
      1. TSV 로드 및 타임스탬프 인덱스 변환
      2. fe_raw_data JSON 플래튼(flatten)
      3. 범주형 변수 Label Encoding
      4. 결측값 및 파싱 오류 방어적 처리
    """

    def __init__(self, config: ProcessorConfig | None = None) -> None:
        self.cfg = config or ProcessorConfig()
        self._df: pd.DataFrame | None = None

    # ── 공개 메서드 ──────────────────────────────────────────────────────────

    def load(self, path: str | Path) -> "DataProcessor":
        """CSV/TSV 파일 로드 및 타임스탬프 인덱스 설정"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")

        df = pd.read_csv(path, sep=self.cfg.sep, low_memory=False)
        logger.info("로드 완료: %s행 × %s열", len(df), len(df.columns))

        # 타임스탬프 인덱스
        if self.cfg.timestamp_col in df.columns:
            df[self.cfg.timestamp_col] = pd.to_datetime(
                df[self.cfg.timestamp_col], errors="coerce"
            )
            df = df.set_index(self.cfg.timestamp_col).sort_index()
        else:
            logger.warning("타임스탬프 컬럼 '%s'를 찾을 수 없습니다.", self.cfg.timestamp_col)

        self._df = df
        return self

    def process(self) -> pd.DataFrame:
        """전처리 전체 파이프라인 실행 후 결과 DataFrame 반환"""
        if self._df is None:
            raise RuntimeError("load()를 먼저 호출하세요.")

        df = self._df.copy()
        df = self._flatten_raw_data(df)
        df = self._encode_categoricals(df)
        df = self._normalize_features(df)
        df = self._drop_incomplete(df)

        logger.info("전처리 완료: %s행, 컬럼=%s", len(df), df.columns.tolist())
        return df

    # ── 내부 처리 메서드 ─────────────────────────────────────────────────────

    def _parse_json_safe(self, raw: Any) -> dict[str, Any]:
        """JSON 파싱 실패 시 빈 딕셔너리 반환 (방어적 프로그래밍)"""
        if pd.isna(raw) or raw in ("\\N", "N", "", None):
            return {}
        try:
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, dict):
                return raw
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug("JSON 파싱 오류 (무시됨): %s | 원본: %.80s", e, str(raw))
        return {}

    def _flatten_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        fe_raw_data JSON 스트링을 파싱하여 개별 컬럼으로 분리.
        JSON 예시: {"foot_type":"right","label":"normal","speed":1.2,
                    "tilt":0.5,"confidence":87,"probabilities":[0.8,0.1,0.1]}
        """
        col = self.cfg.raw_data_col
        if col not in df.columns:
            logger.warning("컬럼 '%s' 없음. JSON 플래튼 건너뜀.", col)
            return df

        parsed = df[col].apply(self._parse_json_safe)

        df["foot_type"] = parsed.apply(lambda d: d.get("foot_type", "left"))
        df["label"] = parsed.apply(lambda d: d.get("label", "normal"))
        df["speed"] = pd.to_numeric(
            parsed.apply(lambda d: d.get("speed", self.cfg.fillna_speed)),
            errors="coerce",
        ).fillna(self.cfg.fillna_speed)
        df["tilt"] = pd.to_numeric(
            parsed.apply(lambda d: d.get("tilt", self.cfg.fillna_tilt)),
            errors="coerce",
        ).fillna(self.cfg.fillna_tilt)
        df["confidence"] = pd.to_numeric(
            parsed.apply(lambda d: d.get("confidence", self.cfg.fillna_confidence)),
            errors="coerce",
        ).fillna(self.cfg.fillna_confidence).astype(int)

        # probabilities 리스트 → 개별 컬럼 (없으면 균등 분포)
        n_classes = len(LABEL_MAP)
        default_probs = [1.0 / n_classes] * n_classes
        probs = parsed.apply(lambda d: d.get("probabilities", default_probs))
        for i, cls_name in enumerate(LABEL_MAP.keys()):
            df[f"prob_{cls_name}"] = probs.apply(
                lambda p, idx=i: float(p[idx]) if isinstance(p, list) and len(p) > idx else default_probs[idx]
            )

        logger.info("fe_raw_data 플래튼 완료: foot_type, label, speed, tilt, confidence, prob_*")
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """범주형 변수를 정수형으로 인코딩"""
        # fe_event_type 인코딩
        df["event_type_enc"] = (
            df[self.cfg.event_type_col]
            .map(EVENT_TYPE_MAP)
            .fillna(0)
            .astype(int)
        )

        # label 인코딩
        df["label_enc"] = df["label"].map(LABEL_MAP).fillna(0).astype(int)

        # foot_type 인코딩
        df["foot_type_enc"] = df["foot_type"].map(FOOT_TYPE_MAP).fillna(0).astype(int)

        # fall 즉각 알림 플래그 (룰 기반)
        df["is_fall"] = (df[self.cfg.event_type_col] == "fall").astype(int)

        # shuffling 플래그 (파킨슨형 보행 징후 관련 이벤트)
        df["is_shuffling"] = (df[self.cfg.event_type_col] == "shuffling").astype(int)

        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """연속형 특성 정규화 (0~1 범위)"""
        # fe_event_value 정규화
        if "fe_event_value" in df.columns:
            ev = pd.to_numeric(df["fe_event_value"], errors="coerce").fillna(0.0)
            scale = self.cfg.event_value_scale
            df["fe_event_value_norm"] = (ev / scale).clip(0.0, 1.0)
        else:
            df["fe_event_value_norm"] = 0.0

        # confidence 정규화 (0~100 → 0~1)
        df["confidence_norm"] = (df["confidence"] / 100.0).clip(0.0, 1.0)

        return df

    def _drop_incomplete(self, df: pd.DataFrame) -> pd.DataFrame:
        """핵심 특성 컬럼의 NaN이 있는 행 제거"""
        essential = ["speed", "tilt", "event_type_enc", "label_enc"]
        before = len(df)
        df = df.dropna(subset=[c for c in essential if c in df.columns])
        dropped = before - len(df)
        if dropped > 0:
            logger.warning("NaN으로 인해 %s행 제거됨", dropped)
        return df

    # ── 통계 요약 ─────────────────────────────────────────────────────────────

    def summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """데이터 통계 요약 반환"""
        return {
            "total_rows": len(df),
            "event_type_dist": df[self.cfg.event_type_col].value_counts().to_dict()
            if self.cfg.event_type_col in df.columns else {},
            "label_dist": df["label"].value_counts().to_dict() if "label" in df.columns else {},
            "fall_count": int(df["is_fall"].sum()) if "is_fall" in df.columns else 0,
            "shuffling_count": int(df["is_shuffling"].sum()) if "is_shuffling" in df.columns else 0,
            "speed_stats": df["speed"].describe().to_dict() if "speed" in df.columns else {},
            "tilt_stats": df["tilt"].describe().to_dict() if "tilt" in df.columns else {},
        }


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────


class GaitWindowDataset(Dataset):
    """
    슬라이딩 윈도우 기반 PyTorch Dataset.

    각 샘플: (window_tensor, label, is_fall_flag)
      - window_tensor: [window_size, n_features] float32
      - label: int (0=normal, 1=waiting, 2=drag)
      - is_fall_flag: int (0 or 1) — 룰 기반 즉각 알림용
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        stride: int = 10,
        feature_cols: list[str] | None = None,
        label_col: str = "label_enc",
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.feature_cols = feature_cols or FEATURE_COLS
        self.label_col = label_col

        # 필요한 컬럼만 추출 후 numpy 변환
        avail_features = [c for c in self.feature_cols if c in df.columns]
        if not avail_features:
            raise ValueError(f"특성 컬럼이 없습니다. 사용 가능: {df.columns.tolist()}")

        self.features = df[avail_features].values.astype(np.float32)
        self.labels = df[label_col].values.astype(np.int64) if label_col in df.columns else np.zeros(len(df), dtype=np.int64)
        self.is_fall = df["is_fall"].values.astype(np.int64) if "is_fall" in df.columns else np.zeros(len(df), dtype=np.int64)
        self.n_features = len(avail_features)

        # 유효 슬라이딩 윈도우 시작 인덱스 계산
        self.indices = list(range(0, len(self.features) - window_size + 1, stride))
        logger.info(
            "Dataset 생성: %s 샘플, window=%s, stride=%s, features=%s",
            len(self.indices), window_size, stride, avail_features,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.window_size

        window = torch.from_numpy(self.features[start:end])  # [window_size, n_features]
        label = torch.tensor(int(self.labels[end - 1]), dtype=torch.long)
        is_fall = torch.tensor(int(self.is_fall[start:end].max()), dtype=torch.long)

        return window, label, is_fall


# ─── DataLoader 팩토리 ────────────────────────────────────────────────────────


@dataclass
class DataLoaderConfig:
    window_size: int = 30
    stride: int = 10
    batch_size: int = 64
    num_workers: int = 0
    train_ratio: float = 0.8
    shuffle_train: bool = True
    feature_cols: list[str] = field(default_factory=lambda: FEATURE_COLS)


def build_dataloaders(
    df: pd.DataFrame,
    cfg: DataLoaderConfig | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    전처리된 DataFrame으로부터 train/val DataLoader 생성.

    Returns:
        train_loader, val_loader
    """
    cfg = cfg or DataLoaderConfig()

    n = len(df)
    split = int(n * cfg.train_ratio)
    df_train = df.iloc[:split]
    df_val = df.iloc[split:]

    # 클래스 불균형 보정 (낙상/끌림이 희소)
    train_dataset = GaitWindowDataset(df_train, cfg.window_size, cfg.stride, cfg.feature_cols)
    val_dataset = GaitWindowDataset(df_val, cfg.window_size, 1, cfg.feature_cols)

    train_labels = [int(train_dataset.labels[i + cfg.window_size - 1]) for i in train_dataset.indices]
    class_counts = np.bincount(train_labels, minlength=len(LABEL_MAP))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = torch.tensor([class_weights[l] for l in train_labels], dtype=torch.float)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "DataLoader 생성 완료 — train: %s 배치, val: %s 배치",
        len(train_loader), len(val_loader),
    )
    return train_loader, val_loader


# ─── 빠른 테스트용 entrypoint ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "CES.csv"
    processor = DataProcessor()
    processor.load(csv_path)
    df = processor.process()

    print("\n=== 데이터 통계 요약 ===")
    for k, v in processor.summary(df).items():
        print(f"  {k}: {v}")

    print("\n=== 샘플 데이터 (5행) ===")
    display_cols = ["speed", "tilt", "event_type_enc", "label_enc", "is_fall", "confidence_norm"]
    print(df[[c for c in display_cols if c in df.columns]].head())

    train_loader, val_loader = build_dataloaders(df)
    batch = next(iter(train_loader))
    window, label, is_fall = batch
    print(f"\nBatch 형태: window={window.shape}, label={label.shape}, is_fall={is_fall.shape}")
