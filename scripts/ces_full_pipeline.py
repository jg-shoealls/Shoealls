#!/usr/bin/env python3
"""
============================================================
 Shoealls CES 보행 패턴 분류 + 낙상 감지 완전 실행 파이프라인
============================================================

한 파일에서 즉시 실행 가능한 독립 스크립트입니다.
외부 모듈 의존 없음 — Python 표준 + pandas + torch 만 사용합니다.

실행 예시:
  python scripts/ces_full_pipeline.py --data CES.csv --epochs 20
  python scripts/ces_full_pipeline.py --data CES.csv --no-train
  python scripts/ces_full_pipeline.py --data CES.csv --epochs 30 --save model.pt

포함 기능:
  1. DataProcessor      — TSV 로드 / JSON 플래튼 / 라벨 인코딩 / 타임스탬프 인덱스
  2. GaitWindowDataset  — 슬라이딩 윈도우 시계열 PyTorch Dataset
  3. GaitLSTM           — BiLSTM 분류기 + 룰 기반 낙상 감지 통합
  4. 학습 / 검증 루프   — 체크포인트 저장 포함
"""

from __future__ import annotations

# ─── 표준 라이브러리 ──────────────────────────────────────────────────────────
import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ─── 서드파티 ─────────────────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError as e:
    print(f"[오류] 필수 패키지 미설치: {e}")
    print("설치 명령: pip install pandas numpy torch")
    sys.exit(1)

# ─── 로거 설정 ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ces_pipeline")


# ==============================================================================
# PART 1. DataProcessor — CES.csv 전처리
# ==============================================================================

# 인코딩 맵 (모델 입력용 정수 변환)
EVENT_TYPE_MAP: dict[str, int] = {
    "acceleration": 0,
    "shuffling":    1,
    "fall":         2,
}
LABEL_MAP: dict[str, int] = {
    "normal":  0,
    "waiting": 1,
    "drag":    2,
}
FOOT_TYPE_MAP: dict[str, int] = {
    "left":  0,
    "right": 1,
}

# 역방향 맵 (추론 결과 디코딩)
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
IDX_TO_EVENT: dict[int, str] = {v: k for k, v in EVENT_TYPE_MAP.items()}

# 슬라이딩 윈도우에 사용할 특성 컬럼 (순서 고정)
FEATURE_COLS: list[str] = [
    "speed",               # 보행 속도 (float)
    "tilt",                # 기울기 (float)
    "fe_event_value_norm", # 이벤트 값 정규화 (float)
    "event_type_enc",      # 이벤트 타입 인코딩 (int→float)
]


class DataProcessor:
    """
    CES.csv (TSV 포맷) 데이터 전처리기.

    처리 순서:
      1. TSV 파일 로드
      2. fe_timestamp → Datetime 인덱스
      3. fe_raw_data JSON 플래튼 (foot_type, label, speed, tilt, confidence, probabilities)
      4. 범주형 변수 Label Encoding (event_type, label, foot_type)
      5. 연속형 특성 정규화
      6. 결측값 / 파싱 오류 방어적 처리
    """

    def __init__(
        self,
        sep: str = "\t",
        timestamp_col: str = "fe_timestamp",
        raw_data_col: str = "fe_raw_data",
        event_type_col: str = "fe_event_type",
        event_value_scale: float = 100.0,
    ) -> None:
        self.sep = sep
        self.timestamp_col = timestamp_col
        self.raw_data_col = raw_data_col
        self.event_type_col = event_type_col
        self.event_value_scale = event_value_scale

    # ── 공개 메서드 ──────────────────────────────────────────────────────────

    def load_and_process(self, path: str | Path) -> pd.DataFrame:
        """
        파일 로드부터 전처리 완료까지 한 번에 실행.

        Returns:
            전처리 완료된 DataFrame
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"데이터 파일 없음: {path}")

        log.info("파일 로드 중: %s", path)
        df = pd.read_csv(path, sep=self.sep, low_memory=False)
        log.info("로드 완료: %d행 × %d열", len(df), len(df.columns))

        df = self._set_datetime_index(df)
        df = self._flatten_raw_data(df)
        df = self._encode_labels(df)
        df = self._normalize_features(df)
        df = self._drop_nulls(df)

        log.info("전처리 완료: %d행 남음", len(df))
        return df

    def print_summary(self, df: pd.DataFrame) -> None:
        """데이터 통계 요약 출력"""
        print("\n" + "=" * 60)
        print("  데이터 통계 요약")
        print("=" * 60)
        print(f"  총 행 수         : {len(df):,}")
        print(f"  타임스탬프 범위  : {df.index.min()} ~ {df.index.max()}" if hasattr(df.index, 'min') else "")
        if self.event_type_col in df.columns:
            print(f"  이벤트 타입 분포 : {df[self.event_type_col].value_counts().to_dict()}")
        if "label" in df.columns:
            print(f"  레이블 분포      : {df['label'].value_counts().to_dict()}")
        if "is_fall" in df.columns:
            print(f"  낙상 이벤트 수   : {df['is_fall'].sum():,}")
        if "is_shuffling" in df.columns:
            print(f"  끌림 이벤트 수   : {df['is_shuffling'].sum():,}")
        if "speed" in df.columns:
            print(f"  속도 통계        : mean={df['speed'].mean():.3f}  std={df['speed'].std():.3f}")
        if "tilt" in df.columns:
            print(f"  기울기 통계      : mean={df['tilt'].mean():.3f}  std={df['tilt'].std():.3f}")
        print("=" * 60 + "\n")

    # ── 내부 처리 메서드 ─────────────────────────────────────────────────────

    def _set_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """fe_timestamp 컬럼을 Datetime 인덱스로 변환"""
        if self.timestamp_col not in df.columns:
            log.warning("타임스탬프 컬럼 '%s' 없음 — 인덱스 변환 건너뜀", self.timestamp_col)
            return df
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors="coerce")
        df = df.set_index(self.timestamp_col).sort_index()
        nat_count = df.index.isna().sum()
        if nat_count > 0:
            log.warning("타임스탬프 파싱 실패 %d행 제거", nat_count)
            df = df[~df.index.isna()]
        return df

    def _parse_json_safe(self, raw: Any) -> dict[str, Any]:
        """JSON 파싱 실패 / NaN 등 비정상 입력 방어 처리"""
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return {}
        if raw in ("\\N", "N", "null", "None", ""):
            return {}
        try:
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, dict):
                return raw
        except (json.JSONDecodeError, ValueError):
            pass
        return {}

    def _flatten_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        fe_raw_data JSON 스트링을 파싱하여 개별 컬럼으로 분리(Flatten).

        JSON 예시:
          {"foot_type":"right","label":"normal","speed":1.2,"tilt":0.05,
           "confidence":87,"probabilities":[0.8,0.1,0.1]}
        """
        col = self.raw_data_col
        if col not in df.columns:
            log.warning("컬럼 '%s' 없음 — JSON 플래튼 건너뜀", col)
            return df

        log.info("fe_raw_data JSON 파싱 중...")
        parsed = df[col].apply(self._parse_json_safe)

        # 기본 필드 추출
        df["foot_type"]  = parsed.apply(lambda d: str(d.get("foot_type", "left")))
        df["label"]      = parsed.apply(lambda d: str(d.get("label", "normal")))
        df["speed"]      = pd.to_numeric(parsed.apply(lambda d: d.get("speed", 0.0)), errors="coerce").fillna(0.0)
        df["tilt"]       = pd.to_numeric(parsed.apply(lambda d: d.get("tilt", 0.0)), errors="coerce").fillna(0.0)
        df["confidence"] = pd.to_numeric(parsed.apply(lambda d: d.get("confidence", 50)), errors="coerce").fillna(50).astype(int)

        # probabilities 리스트 → 클래스별 개별 컬럼
        n_cls = len(LABEL_MAP)
        default_p = [1.0 / n_cls] * n_cls
        probs_series = parsed.apply(lambda d: d.get("probabilities", default_p))
        for i, cls_name in enumerate(LABEL_MAP.keys()):
            df[f"prob_{cls_name}"] = probs_series.apply(
                lambda p, idx=i: float(p[idx]) if isinstance(p, list) and len(p) > idx else default_p[idx]
            )

        parse_fail = (df["speed"] == 0.0).sum()
        log.info("JSON 플래튼 완료 (속도=0 의심 행: %d)", parse_fail)
        return df

    def _encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """범주형 변수를 정수형(Label Encoding)으로 변환"""
        # fe_event_type 인코딩
        df["event_type_enc"] = (
            df[self.event_type_col].map(EVENT_TYPE_MAP).fillna(0).astype(int)
        )
        # label 인코딩
        df["label_enc"] = df["label"].map(LABEL_MAP).fillna(0).astype(int)
        # foot_type 인코딩
        df["foot_type_enc"] = df["foot_type"].map(FOOT_TYPE_MAP).fillna(0).astype(int)

        # 즉각 알림용 불리언 플래그
        df["is_fall"]      = (df[self.event_type_col] == "fall").astype(int)
        df["is_shuffling"] = (df[self.event_type_col] == "shuffling").astype(int)

        log.info(
            "라벨 인코딩 완료: fall=%d  shuffling=%d  normal=%d",
            df["is_fall"].sum(), df["is_shuffling"].sum(),
            (df["event_type_enc"] == 0).sum(),
        )
        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """연속형 특성 0~1 범위로 정규화"""
        if "fe_event_value" in df.columns:
            ev = pd.to_numeric(df["fe_event_value"], errors="coerce").fillna(0.0)
            df["fe_event_value_norm"] = (ev / self.event_value_scale).clip(0.0, 1.0)
        else:
            df["fe_event_value_norm"] = 0.0

        df["confidence_norm"] = (df["confidence"] / 100.0).clip(0.0, 1.0)
        return df

    def _drop_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """필수 특성 컬럼의 NaN 행 제거"""
        essential = ["speed", "tilt", "event_type_enc", "label_enc"]
        before = len(df)
        df = df.dropna(subset=[c for c in essential if c in df.columns])
        removed = before - len(df)
        if removed > 0:
            log.warning("NaN으로 인해 %d행 제거", removed)
        return df


# ==============================================================================
# PART 2. GaitWindowDataset — 슬라이딩 윈도우 PyTorch Dataset
# ==============================================================================

class GaitWindowDataset(Dataset):
    """
    슬라이딩 윈도우 기반 시계열 PyTorch Dataset.

    각 샘플:
      x          : [window_size, n_features]  float32 텐서
      label      : int64  (0=normal, 1=waiting, 2=drag)
      fall_flag  : bool   (윈도우 내 fall 이벤트 존재 → 룰 기반 즉각 알림)

    슬라이딩 예시 (window=4, stride=2):
      indices: [0,1,2,3], [2,3,4,5], [4,5,6,7], ...
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        stride: int = 10,
        feature_cols: list[str] | None = None,
        label_col: str = "label_enc",
    ) -> None:
        if len(df) < window_size:
            raise ValueError(
                f"데이터 행 수({len(df)})가 window_size({window_size})보다 작습니다."
            )

        self.window_size = window_size
        self.stride = stride
        self.feature_cols = feature_cols or FEATURE_COLS

        # 사용 가능한 특성만 추출
        avail = [c for c in self.feature_cols if c in df.columns]
        if not avail:
            raise ValueError(f"사용 가능한 특성 컬럼 없음. DataFrame 컬럼: {df.columns.tolist()}")
        if len(avail) < len(self.feature_cols):
            missing = set(self.feature_cols) - set(avail)
            log.warning("특성 컬럼 일부 없음 (무시됨): %s", missing)
        self.feature_cols = avail

        self.features  = df[avail].values.astype(np.float32)          # [N, n_features]
        self.labels    = df[label_col].values.astype(np.int64) if label_col in df.columns \
                         else np.zeros(len(df), dtype=np.int64)
        self.fall_flags = df["is_fall"].values.astype(np.int64) if "is_fall" in df.columns \
                          else np.zeros(len(df), dtype=np.int64)

        # 유효 윈도우 시작 인덱스
        self.indices = list(range(0, len(self.features) - window_size + 1, stride))
        log.info(
            "Dataset 생성: %d 샘플 (window=%d, stride=%d, features=%s)",
            len(self.indices), window_size, stride, avail,
        )

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.indices[idx]
        end   = start + self.window_size
        return {
            "x":         torch.from_numpy(self.features[start:end]),        # [window, n_feat]
            "label":     torch.tensor(int(self.labels[end - 1]), dtype=torch.long),
            "fall_flag": torch.tensor(int(self.fall_flags[start:end].max()), dtype=torch.bool),
        }


def build_dataloaders(
    df: pd.DataFrame,
    window_size: int = 30,
    stride: int = 10,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, int]:
    """
    DataFrame → (train_loader, val_loader, n_features) 반환.

    클래스 불균형 보정: WeightedRandomSampler 사용.
    """
    # 시간 순서대로 train/val 분할 (시계열 누수 방지)
    split = int(len(df) * train_ratio)
    df_train = df.iloc[:split].copy()
    df_val   = df.iloc[split:].copy()

    train_ds = GaitWindowDataset(df_train, window_size, stride)
    val_ds   = GaitWindowDataset(df_val,   window_size, 1)    # val은 stride=1

    # 클래스 가중치 샘플러 (rare class 오버샘플링)
    train_labels = [int(train_ds.labels[i + window_size - 1]) for i in train_ds.indices]
    n_cls  = len(LABEL_MAP)
    counts = np.bincount(train_labels, minlength=n_cls)
    weights_cls = 1.0 / (counts.astype(float) + 1e-6)
    sample_w = torch.tensor([weights_cls[l] for l in train_labels], dtype=torch.float)
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    log.info(
        "DataLoader 생성: train=%d배치 / val=%d배치 (batch=%d)",
        len(train_loader), len(val_loader), batch_size,
    )
    return train_loader, val_loader, train_ds.n_features


# ==============================================================================
# PART 3. GaitLSTM — AI 모델 + 룰 기반 낙상 감지 통합
# ==============================================================================

class TemporalAttention(nn.Module):
    """LSTM 시퀀스 출력 기반 소프트 어텐션 (중요한 시간 구간 강조)"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: [batch, seq, hidden_dim]
        Returns:
            context:  [batch, hidden_dim]
            weights:  [batch, seq]  (해석 가능성용)
        """
        weights = F.softmax(self.score(lstm_out).squeeze(-1), dim=-1)  # [batch, seq]
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)         # [batch, hidden]
        return context, weights


class GaitLSTM(nn.Module):
    """
    보행 상태 분류 BiLSTM + Temporal Attention.

    ┌───────────────────────────────────────────────────────┐
    │  입력: [batch, window_size, n_features]               │
    │  → BatchNorm → BiLSTM → Temporal Attention            │
    │  → 분류 헤드 → logits [batch, n_classes]              │
    │  → 낙상 헤드 → fall_score [batch, 1]  (0~1)           │
    └───────────────────────────────────────────────────────┘

    predict_with_fall_rule():
      fe_event_type == 'fall' → 즉각 fall_alert=True  (룰 기반, AI 점수 무관)
      fall_score >= threshold → fall_alert=True       (AI 기반)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # 입력 정규화
        self.input_norm = nn.BatchNorm1d(input_dim)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # 어텐션
        self.attention = TemporalAttention(lstm_out_dim)

        # 보행 상태 분류 헤드 (normal / waiting / drag)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

        # 낙상 위험 점수 헤드 (AI 기반, 0~1)
        self.fall_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """가중치 초기화 (LSTM: Xavier/Orthogonal, Linear: 기본값)"""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq, n_features]
        Returns:
            logits      : [batch, n_classes]
            fall_score  : [batch, 1]
            attn_weights: [batch, seq]
        """
        # BatchNorm — [batch, n_features, seq] 형태로 변환 후 역변환
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)

        lstm_out, _ = self.lstm(x)                    # [batch, seq, hidden*2]
        context, attn_w = self.attention(lstm_out)     # [batch, hidden*2], [batch, seq]

        logits     = self.classifier(context)          # [batch, n_classes]
        fall_score = self.fall_head(context)           # [batch, 1]
        return logits, fall_score, attn_w

    def predict_with_fall_rule(
        self,
        x: torch.Tensor,
        fall_flag: bool | torch.Tensor = False,
        fall_threshold: float = 0.65,
    ) -> dict[str, Any]:
        """
        AI 분류 결과 + 룰 기반 낙상 감지 통합 추론.

        ┌─────────────────────────────────────────────────────┐
        │  fe_event_type == 'fall'  →  fall_alert = True      │
        │  (즉각 알림, AI 점수 계산과 무관하게 최우선 처리)    │
        │                                                      │
        │  fall_score >= threshold  →  fall_alert = True      │
        │  (AI 기반 낙상 위험 감지)                            │
        │                                                      │
        │  두 조건 모두 해당 없으면  →  fall_alert = False     │
        └─────────────────────────────────────────────────────┘

        Args:
            x:              [seq, n_features] 또는 [1, seq, n_features]
            fall_flag:      True = fe_event_type이 'fall'인 경우 (룰 기반)
            fall_threshold: AI 낙상 점수 임계값

        Returns:
            {
              "label":           str,         # 예측 클래스명
              "label_idx":       int,
              "probabilities":   list[float], # 클래스별 확률
              "fall_score_ai":   float,       # AI 낙상 위험 점수 (0~1)
              "fall_alert":      bool,        # 최종 낙상 알림 플래그
              "is_rule_trigger": bool,        # 룰 기반 감지 여부
              "is_ai_trigger":   bool,        # AI 기반 감지 여부
              "attention":       list[float], # 시간 어텐션 가중치
            }
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)               # [1, seq, features]
            logits, fall_t, attn_w = self.forward(x)
            probs = F.softmax(logits, dim=-1)[0]  # [n_classes]

        label_idx = int(probs.argmax().item())
        ai_score  = float(fall_t[0, 0].item())

        rule_trigger = (
            bool(fall_flag.item()) if isinstance(fall_flag, torch.Tensor) else bool(fall_flag)
        )
        ai_trigger = ai_score >= fall_threshold

        return {
            "label":           IDX_TO_LABEL.get(label_idx, "unknown"),
            "label_idx":       label_idx,
            "probabilities":   probs.cpu().tolist(),
            "fall_score_ai":   round(ai_score, 4),
            "fall_alert":      rule_trigger or ai_trigger,
            "is_rule_trigger": rule_trigger,
            "is_ai_trigger":   ai_trigger,
            "attention":       attn_w[0].cpu().tolist(),
        }


# ==============================================================================
# PART 4. 손실 함수 + 학습 / 검증 루프
# ==============================================================================

class CombinedLoss(nn.Module):
    """
    다중 태스크 손실:
      total = CrossEntropy(보행 분류) + lambda_fall × BCE(낙상 위험)

    낙상이 희소한 의료 데이터 특성 상 lambda_fall > 1로 상향 가중치.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        lambda_fall: float = 2.5,
    ) -> None:
        super().__init__()
        self.ce       = nn.CrossEntropyLoss(weight=class_weights)
        self.bce      = nn.BCELoss()
        self.lam_fall = lambda_fall

    def forward(
        self,
        logits:      torch.Tensor,  # [batch, n_classes]
        labels:      torch.Tensor,  # [batch]       정수
        fall_score:  torch.Tensor,  # [batch, 1]    sigmoid 출력
        fall_target: torch.Tensor,  # [batch]       0 or 1
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss  = self.ce(logits, labels)
        bce_loss = self.bce(fall_score.squeeze(-1), fall_target.float())
        total    = ce_loss + self.lam_fall * bce_loss
        return total, {
            "ce":    round(float(ce_loss.item()),  4),
            "bce":   round(float(bce_loss.item()), 4),
            "total": round(float(total.item()),    4),
        }


@dataclass
class EpochResult:
    epoch:        int
    loss:         float
    ce_loss:      float
    bce_loss:     float
    accuracy:     float
    fall_recall:  float   # 낙상 감지 재현율 (핵심 지표)
    val_accuracy: float = 0.0
    val_fall_recall: float = 0.0


def train_one_epoch(
    model:     GaitLSTM,
    loader:    DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> dict[str, float]:
    """1 에폭 학습 루프"""
    model.train()
    total_loss = ce_sum = bce_sum = 0.0
    correct = total = 0
    fall_tp = fall_total = 0

    for batch in loader:
        x          = batch["x"].to(device)            # [batch, seq, features]
        label      = batch["label"].to(device)         # [batch]
        fall_flag  = batch["fall_flag"].to(device)     # [batch] bool

        optimizer.zero_grad(set_to_none=True)
        logits, fall_score, _ = model(x)

        loss, breakdown = criterion(logits, label, fall_score, fall_flag.long())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += breakdown["total"]
        ce_sum     += breakdown["ce"]
        bce_sum    += breakdown["bce"]
        preds       = logits.argmax(dim=-1)
        correct    += int((preds == label).sum().item())
        total      += int(label.numel())

        # 낙상 재현율 계산
        predicted_fall = (fall_score.squeeze(-1) >= 0.65)
        fall_total += int(fall_flag.sum().item())
        fall_tp    += int((predicted_fall & fall_flag).sum().item())

    n = max(len(loader), 1)
    return {
        "loss":        round(total_loss / n, 4),
        "ce":          round(ce_sum     / n, 4),
        "bce":         round(bce_sum    / n, 4),
        "accuracy":    round(correct    / max(total, 1), 4),
        "fall_recall": round(fall_tp    / max(fall_total, 1), 4),
    }


@torch.no_grad()
def evaluate(
    model:  GaitLSTM,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """검증 루프"""
    model.eval()
    correct = total = fall_tp = fall_total = 0

    for batch in loader:
        x         = batch["x"].to(device)
        label     = batch["label"].to(device)
        fall_flag = batch["fall_flag"].to(device)

        logits, fall_score, _ = model(x)
        preds = logits.argmax(dim=-1)
        correct    += int((preds == label).sum().item())
        total      += int(label.numel())
        predicted_fall = (fall_score.squeeze(-1) >= 0.65)
        fall_total += int(fall_flag.sum().item())
        fall_tp    += int((predicted_fall & fall_flag).sum().item())

    return {
        "accuracy":    round(correct / max(total, 1), 4),
        "fall_recall": round(fall_tp / max(fall_total, 1), 4),
    }


def train(
    model:        GaitLSTM,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int = 30,
    lr:           float = 3e-4,
    device:       torch.device = torch.device("cpu"),
    save_path:    str = "",
) -> list[EpochResult]:
    """
    전체 학습 루프.

    Args:
        model:        GaitLSTM 인스턴스
        train_loader: 학습 DataLoader
        val_loader:   검증 DataLoader
        epochs:       총 에폭 수
        lr:           학습률
        device:       torch.device
        save_path:    체크포인트 저장 경로 (빈 문자열 = 저장 안 함)

    Returns:
        EpochResult 리스트 (학습 이력)
    """
    criterion = CombinedLoss(lambda_fall=2.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state   = None
    history: list[EpochResult] = []

    log.info("학습 시작: epochs=%d  lr=%.2e  device=%s", epochs, lr, device)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m   = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        lr_now  = scheduler.get_last_lr()[0]

        result = EpochResult(
            epoch        = epoch,
            loss         = train_m["loss"],
            ce_loss      = train_m["ce"],
            bce_loss     = train_m["bce"],
            accuracy     = train_m["accuracy"],
            fall_recall  = train_m["fall_recall"],
            val_accuracy     = val_m["accuracy"],
            val_fall_recall  = val_m["fall_recall"],
        )
        history.append(result)

        # 콘솔 출력
        marker = " ◀ best" if val_m["accuracy"] > best_val_acc else ""
        log.info(
            "Epoch %3d/%d | loss=%.4f (ce=%.4f bce=%.4f) | "
            "acc=%.4f fall_rec=%.4f | val_acc=%.4f val_fall=%.4f | "
            "lr=%.1e  %.1fs%s",
            epoch, epochs,
            result.loss, result.ce_loss, result.bce_loss,
            result.accuracy, result.fall_recall,
            result.val_accuracy, result.val_fall_recall,
            lr_now, elapsed, marker,
        )

        # 베스트 모델 체크포인트
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 베스트 가중치 복원
    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("베스트 모델 복원 (val_acc=%.4f)", best_val_acc)

    # 체크포인트 저장
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": best_state,
            "val_accuracy":     best_val_acc,
            "n_features":       model.lstm.input_size,
            "hidden_dim":       model.lstm.hidden_size,
            "num_classes":      model.classifier[-1].out_features,
            "feature_cols":     FEATURE_COLS,
            "label_map":        LABEL_MAP,
            "history":          [vars(r) for r in history],
        }, save_path)
        log.info("체크포인트 저장: %s", save_path)

    return history


# ==============================================================================
# PART 5. 추론 데모 — predict_with_fall_rule 실사용 예시
# ==============================================================================

def demo_inference(model: GaitLSTM, val_loader: DataLoader, device: torch.device) -> None:
    """
    학습된 모델로 실제 추론 + 낙상 감지 데모.
    룰 기반(fall_flag)과 AI 기반(fall_score)의 결합 결과를 출력합니다.
    """
    print("\n" + "=" * 60)
    print("  추론 데모 (최대 5 샘플)")
    print("=" * 60)

    model.eval()
    shown = 0
    for batch in val_loader:
        if shown >= 5:
            break
        x          = batch["x"].to(device)
        fall_flags = batch["fall_flag"]

        for i in range(min(len(x), 5 - shown)):
            result = model.predict_with_fall_rule(
                x[i],
                fall_flag=fall_flags[i],
                fall_threshold=0.65,
            )
            alert_str = "🚨 낙상 알림!" if result["fall_alert"] else "정상"
            print(
                f"  [{shown + 1}] 예측={result['label']:8s} | "
                f"신뢰도={max(result['probabilities']):.3f} | "
                f"AI낙상점수={result['fall_score_ai']:.3f} | "
                f"룰트리거={result['is_rule_trigger']} | "
                f"AI트리거={result['is_ai_trigger']} | "
                f"{alert_str}"
            )
            shown += 1

    print("=" * 60)


# ==============================================================================
# PART 6. CLI 진입점
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shoealls CES 보행 분류 + 낙상 감지 파이프라인",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",         default="CES.csv",           help="CES.csv 경로")
    p.add_argument("--epochs",       type=int,   default=20,      help="학습 에폭 수")
    p.add_argument("--batch-size",   type=int,   default=64,      help="배치 크기")
    p.add_argument("--window-size",  type=int,   default=30,      help="슬라이딩 윈도우 크기")
    p.add_argument("--stride",       type=int,   default=10,      help="윈도우 스트라이드")
    p.add_argument("--hidden-dim",   type=int,   default=128,     help="LSTM hidden 크기")
    p.add_argument("--lr",           type=float, default=3e-4,    help="학습률")
    p.add_argument("--train-ratio",  type=float, default=0.8,     help="학습/검증 분할 비율")
    p.add_argument("--device",       default="auto",              help="cpu / cuda / auto")
    p.add_argument("--save",         default="",                  help="체크포인트 저장 경로")
    p.add_argument("--no-train",     action="store_true",         help="데이터 확인만 (학습 건너뜀)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 디바이스 ──────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("사용 디바이스: %s", device)

    # 재현성
    torch.manual_seed(42)
    np.random.seed(42)

    # ── 1. 데이터 전처리 ──────────────────────────────────────────────────────
    processor = DataProcessor()
    df = processor.load_and_process(args.data)
    processor.print_summary(df)

    if args.no_train:
        print("--no-train 모드: 데이터 확인 완료 후 종료합니다.")
        return

    # ── 2. DataLoader 생성 ────────────────────────────────────────────────────
    train_loader, val_loader, n_features = build_dataloaders(
        df,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
    )

    # ── 3. 모델 초기화 ────────────────────────────────────────────────────────
    model = GaitLSTM(
        input_dim=n_features,
        hidden_dim=args.hidden_dim,
        num_classes=len(LABEL_MAP),
        bidirectional=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "모델 초기화: GaitLSTM | 파라미터=%d | 특성=%d | 클래스=%d",
        n_params, n_features, len(LABEL_MAP),
    )

    # ── 4. 학습 ───────────────────────────────────────────────────────────────
    history = train(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=args.save,
    )

    # ── 5. 최종 결과 출력 ─────────────────────────────────────────────────────
    best = max(history, key=lambda r: r.val_accuracy)
    print("\n" + "=" * 60)
    print("  학습 완료 요약")
    print("=" * 60)
    print(f"  최고 검증 정확도  : {best.val_accuracy:.4f}  (epoch {best.epoch})")
    print(f"  최고 낙상 재현율  : {max(r.val_fall_recall for r in history):.4f}")
    print(f"  최종 학습 손실    : {history[-1].loss:.4f}")
    if args.save:
        print(f"  체크포인트 저장   : {args.save}")
    print("=" * 60)

    # ── 6. 추론 데모 ──────────────────────────────────────────────────────────
    demo_inference(model, val_loader, device)


if __name__ == "__main__":
    main()
