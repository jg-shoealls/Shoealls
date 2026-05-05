"""공유 matplotlib 설정 및 시각화 유틸리티.

한글 폰트 설정, 공통 색상 팔레트, 레이블 매핑, 공통 축 스타일 함수를
이 모듈 하나에 모아 여러 시각화 모듈에서 재사용합니다.

import 시 matplotlib Agg 백엔드 설정과 rcParams 이 자동으로 적용됩니다.
"""

import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after use())
from matplotlib import font_manager as fm

# ── 한글 폰트 설정 ────────────────────────────────────────────────────
_system_os = platform.system()
if _system_os == "Windows":
    font_family = "Malgun Gothic"
elif _system_os == "Darwin":
    font_family = "AppleGothic"
else:
    font_family = "NanumGothic"

FONT_PROP       = fm.FontProperties(family=font_family, weight="bold")
FONT_PROP_LIGHT = fm.FontProperties(family=font_family, weight="normal")
plt.rcParams["font.family"]        = font_family
plt.rcParams["axes.unicode_minus"] = False

# ── 색상 팔레트 ──────────────────────────────────────────────────────
C_PRIMARY  = "#1B3A5C"
C_ACCENT   = "#E8792B"
C_SUCCESS  = "#2E8B57"
C_DANGER   = "#C0392B"
C_WARNING  = "#F39C12"
C_LIGHT_BG = "#F7F9FC"
C_INFO     = "#2196F3"

# ── 한글 레이블 매핑 ──────────────────────────────────────────────────
CLASS_KR: dict[str, str] = {
    "normal":        "정상 보행",
    "antalgic":      "절뚝거림(Antalgic)",
    "ataxic":        "운동실조(Ataxic)",
    "parkinsonian":  "파킨슨(Parkinsonian)",
}

METRIC_KR: dict[str, str] = {
    "cop_sway":           "체중심 흔들림",
    "stride_regularity":  "보폭 규칙성",
    "step_symmetry":      "보행 대칭성",
    "ml_index":           "좌우 압력 지수",
    "injury_risk":        "부상 위험도",
    "overall_deviation":  "개인 기준 편차",
}


def set_ax_style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """공통 축 스타일 적용."""
    if title:
        ax.set_title(title, fontproperties=FONT_PROP, fontsize=13, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=FONT_PROP_LIGHT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=FONT_PROP_LIGHT, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15, linewidth=0.5)
