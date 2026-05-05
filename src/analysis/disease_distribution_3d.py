"""3차원 질병 분포 시각화 모듈.

보행 바이오마커 공간에서 11개 질환 클러스터를 3D로 시각화합니다:
  - PCA 기반 3D 축 축소: 13개 보행 특성 → 3개 주성분
  - 임상 축 기반 3D: 이동성(mobility) × 안정성(stability) × 대칭성(symmetry)
  - 인터랙티브 HTML 시각화 (Plotly)
  - 정적 리포트 이미지 (Matplotlib)
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np

from .disease_classifier import DISEASE_LABELS, FEATURE_NAMES, _DISEASE_PROFILES
from .common import get_feature_korean

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 질환 카테고리 및 색상
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DISEASE_CATEGORIES = {
    "신경계": [1, 4, 6],          # 파킨슨, 소뇌실조, 치매
    "뇌혈관계": [2, 7, 8],        # 뇌졸중, 뇌출혈, 뇌경색
    "근골격계": [5, 9, 10],       # 골관절염, 디스크, 류마티스
    "대사/신경": [3],             # 당뇨 신경병증
    "정상": [0],                  # 정상 보행
}

CATEGORY_COLORS = {
    "정상": "#2E8B57",
    "신경계": "#E74C3C",
    "뇌혈관계": "#3498DB",
    "근골격계": "#F39C12",
    "대사/신경": "#9B59B6",
}

CATEGORY_MARKERS = {
    "정상": "o",
    "신경계": "^",
    "뇌혈관계": "s",
    "근골격계": "D",
    "대사/신경": "P",
}

# 임상 축 정의: 보행 특성 → 임상 지표 매핑
CLINICAL_AXES = {
    "이동성 (Mobility)": {
        "features": ["gait_speed", "cadence", "stride_regularity"],
        "weights": [0.5, 0.3, 0.2],
        "description": "보행 속도·보행률·보폭 규칙성 복합 지표",
    },
    "안정성 (Stability)": {
        "features": ["cop_sway", "ml_variability", "trunk_sway"],
        "weights": [0.35, 0.35, 0.3],
        "description": "체중심 흔들림·좌우 변동성·체간 흔들림 복합 지표",
    },
    "대칭성 (Symmetry)": {
        "features": ["step_symmetry", "pressure_asymmetry", "heel_pressure_ratio"],
        "weights": [0.45, 0.35, 0.2],
        "description": "좌우 대칭성·압력 비대칭·뒤꿈치 하중 복합 지표",
    },
}


@dataclass
class DiseaseCluster:
    """단일 질환의 3D 분포 데이터."""
    disease_id: int
    name_en: str
    name_kr: str
    category: str
    samples: np.ndarray    # (n_samples, n_features)
    center: np.ndarray     # (n_features,)
    color: str
    marker: str


def _get_category(disease_id: int) -> str:
    """질환 ID로 카테고리 반환."""
    for cat, ids in DISEASE_CATEGORIES.items():
        if disease_id in ids:
            return cat
    return "정상"


def generate_disease_samples(
    n_samples_per_disease: int = 80,
    random_state: int = 42,
) -> list[DiseaseCluster]:
    """질환별 합성 보행 데이터를 생성합니다.

    Args:
        n_samples_per_disease: 질환당 생성할 샘플 수.
        random_state: 랜덤 시드.

    Returns:
        DiseaseCluster 리스트 (11개 질환).
    """
    rng = np.random.RandomState(random_state)
    clusters = []

    for disease_id, profile in _DISEASE_PROFILES.items():
        mean = np.array(profile["mean"])
        std = np.array(profile["std"])
        samples = rng.normal(mean, std, size=(n_samples_per_disease, len(mean)))
        name_en, name_kr = DISEASE_LABELS[disease_id]
        category = _get_category(disease_id)

        clusters.append(DiseaseCluster(
            disease_id=disease_id,
            name_en=name_en,
            name_kr=name_kr,
            category=category,
            samples=samples,
            center=mean,
            color=CATEGORY_COLORS[category],
            marker=CATEGORY_MARKERS[category],
        ))

    return clusters


_FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
# 안정성 축은 값이 높을수록 불안정하므로 점수를 반전한다
_STABILITY_AXIS_INDEX = 1


def _normalize_clinical_axes(clusters: list[DiseaseCluster]) -> dict:
    """모든 클러스터를 합쳐 전역 정규화 파라미터를 계산합니다.

    Returns:
        dict: feature별 (min, max) 튜플.
    """
    all_samples = np.vstack([c.samples for c in clusters])
    return {
        name: (all_samples[:, i].min(), all_samples[:, i].max())
        for i, name in enumerate(FEATURE_NAMES)
    }


def _project_to_clinical_axes(
    samples: np.ndarray,
    stats: dict,
) -> np.ndarray:
    """(n, 13) 샘플을 (n, 3) 임상 축 점수로 투영 (전역 정규화)."""
    n = samples.shape[0] if samples.ndim == 2 else 1
    arr = samples.reshape(n, -1)
    scores = np.zeros((n, 3))
    for ax_i, ax_def in enumerate(CLINICAL_AXES.values()):
        for feat, weight in zip(ax_def["features"], ax_def["weights"]):
            idx = _FEATURE_IDX[feat]
            col = arr[:, idx]
            fmin, fmax = stats[feat]
            if fmax - fmin > 1e-8:
                normalized = (col - fmin) / (fmax - fmin)
            else:
                normalized = np.zeros_like(col)
            if ax_i == _STABILITY_AXIS_INDEX:
                normalized = 1.0 - normalized
            scores[:, ax_i] += weight * normalized
    return scores


def compute_clinical_scores(
    clusters: list[DiseaseCluster],
) -> list[tuple[DiseaseCluster, np.ndarray]]:
    """모든 클러스터를 임상 축 점수로 변환 (전역 정규화).

    Returns:
        [(cluster, scores_3d), ...] 리스트. scores_3d는 (n, 3).
    """
    stats = _normalize_clinical_axes(clusters)
    return [(c, _project_to_clinical_axes(c.samples, stats)) for c in clusters]


def _fit_pca_3d(clusters: list[DiseaseCluster]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """클러스터를 표준화하고 SVD 기반 3D PCA를 적합합니다.

    Returns:
        (mean, std, components(3×F), explained_ratio(3,))
    """
    all_samples = np.vstack([c.samples for c in clusters])
    mean = all_samples.mean(axis=0)
    std = all_samples.std(axis=0) + 1e-8
    _, S, Vt = np.linalg.svd((all_samples - mean) / std, full_matrices=False)
    components = Vt[:3]
    explained = (S[:3] ** 2) / (S ** 2).sum()
    return mean, std, components, explained


def _project_to_pca(samples: np.ndarray, mean: np.ndarray, std: np.ndarray, components: np.ndarray) -> np.ndarray:
    """샘플을 표준화 후 PCA 공간으로 투영."""
    arr = np.atleast_2d(samples)
    return ((arr - mean) / std) @ components.T


def compute_pca_3d(
    clusters: list[DiseaseCluster],
) -> tuple[list[tuple[DiseaseCluster, np.ndarray]], np.ndarray, np.ndarray]:
    """PCA로 13개 특성을 3차원으로 축소합니다.

    Returns:
        ([(cluster, coords_3d), ...], components, explained_variance_ratio)
    """
    mean, std, components, explained = _fit_pca_3d(clusters)
    results = [
        (c, _project_to_pca(c.samples, mean, std, components))
        for c in clusters
    ]
    return results, components, explained


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Matplotlib 정적 시각화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_disease_distribution_3d(
    save_path: Path,
    n_samples: int = 80,
    mode: str = "clinical",
    elev: float = 25,
    azim: float = 135,
    figsize: tuple = (16, 12),
    random_state: int = 42,
) -> Path:
    """질환 분포 3D 산점도를 생성합니다 (Matplotlib).

    Args:
        save_path: 저장 경로 (.png).
        n_samples: 질환당 샘플 수.
        mode: "clinical" (임상 축) 또는 "pca" (PCA 축).
        elev: 3D 뷰 elevation 각도.
        azim: 3D 뷰 azimuth 각도.
        figsize: 그림 크기.
        random_state: 랜덤 시드.

    Returns:
        저장된 파일 경로.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    clusters = generate_disease_samples(n_samples, random_state)

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

    # ── 메인 3D 산점도 ──
    ax3d = fig.add_subplot(gs[0, :], projection="3d")

    if mode == "pca":
        data, components, explained = compute_pca_3d(clusters)
        xlabel = f"PC1 ({explained[0]*100:.1f}%)"
        ylabel = f"PC2 ({explained[1]*100:.1f}%)"
        zlabel = f"PC3 ({explained[2]*100:.1f}%)"
        title = "질환별 보행 특성 분포 (PCA 3D)"
    else:
        data = compute_clinical_scores(clusters)
        xlabel = "이동성 (Mobility)"
        ylabel = "안정성 (Stability)"
        zlabel = "대칭성 (Symmetry)"
        title = "질환별 보행 특성 분포 (임상 축 3D)"

    # 카테고리별로 그룹핑하여 플롯
    for cluster, coords in data:
        ax3d.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=cluster.color,
            marker=cluster.marker,
            s=25,
            alpha=0.45,
            edgecolors="white",
            linewidths=0.3,
            label=f"{cluster.name_kr}",
        )

    # 클러스터 중심 강조
    for cluster, coords in data:
        center = coords.mean(axis=0)
        ax3d.scatter(
            [center[0]], [center[1]], [center[2]],
            c=cluster.color,
            marker=cluster.marker,
            s=200,
            alpha=1.0,
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
        )
        ax3d.text(
            center[0], center[1], center[2] + 0.03,
            cluster.name_kr,
            fontsize=7,
            ha="center",
            va="bottom",
            fontweight="bold",
            color=cluster.color,
        )

    ax3d.set_xlabel(xlabel, fontsize=10, labelpad=10)
    ax3d.set_ylabel(ylabel, fontsize=10, labelpad=10)
    ax3d.set_zlabel(zlabel, fontsize=10, labelpad=10)
    ax3d.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.tick_params(labelsize=8)

    # 범례 (카테고리별)
    from matplotlib.lines import Line2D
    legend_elements = []
    for cat, color in CATEGORY_COLORS.items():
        marker = CATEGORY_MARKERS[cat]
        legend_elements.append(
            Line2D([0], [0], marker=marker, color="w", markerfacecolor=color,
                   markersize=10, label=cat, markeredgecolor="black", markeredgewidth=0.5)
        )
    ax3d.legend(
        handles=legend_elements, loc="upper left",
        fontsize=9, framealpha=0.8, title="질환 카테고리",
        title_fontsize=10,
    )

    # ── 하단 좌측: 2D 투영 (XY 평면) ──
    ax_xy = fig.add_subplot(gs[1, 0])
    for cluster, coords in data:
        ax_xy.scatter(
            coords[:, 0], coords[:, 1],
            c=cluster.color, marker=cluster.marker,
            s=15, alpha=0.4, edgecolors="white", linewidths=0.2,
        )
        center = coords.mean(axis=0)
        ax_xy.annotate(
            cluster.name_kr, (center[0], center[1]),
            fontsize=6, ha="center", fontweight="bold", color=cluster.color,
        )
    ax_xy.set_xlabel(xlabel, fontsize=9)
    ax_xy.set_ylabel(ylabel, fontsize=9)
    ax_xy.set_title(f"2D 투영: {xlabel} vs {ylabel}", fontsize=10)
    ax_xy.grid(True, alpha=0.15)
    ax_xy.tick_params(labelsize=8)

    # ── 하단 우측: 2D 투영 (XZ 평면) ──
    ax_xz = fig.add_subplot(gs[1, 1])
    for cluster, coords in data:
        ax_xz.scatter(
            coords[:, 0], coords[:, 2],
            c=cluster.color, marker=cluster.marker,
            s=15, alpha=0.4, edgecolors="white", linewidths=0.2,
        )
        center = coords.mean(axis=0)
        ax_xz.annotate(
            cluster.name_kr, (center[0], center[2]),
            fontsize=6, ha="center", fontweight="bold", color=cluster.color,
        )
    ax_xz.set_xlabel(xlabel, fontsize=9)
    ax_xz.set_ylabel(zlabel, fontsize=9)
    ax_xz.set_title(f"2D 투영: {xlabel} vs {zlabel}", fontsize=10)
    ax_xz.grid(True, alpha=0.15)
    ax_xz.tick_params(labelsize=8)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plotly 인터랙티브 시각화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_disease_distribution_3d_interactive(
    save_path: Path,
    n_samples: int = 80,
    mode: str = "clinical",
    random_state: int = 42,
) -> Path:
    """질환 분포 3D 인터랙티브 시각화를 생성합니다 (Plotly HTML).

    마우스 드래그로 회전, 줌, 호버 정보 확인이 가능합니다.

    Args:
        save_path: 저장 경로 (.html).
        n_samples: 질환당 샘플 수.
        mode: "clinical" 또는 "pca".
        random_state: 랜덤 시드.

    Returns:
        저장된 HTML 파일 경로.
    """
    import plotly.graph_objects as go

    clusters = generate_disease_samples(n_samples, random_state)

    if mode == "pca":
        data, components, explained = compute_pca_3d(clusters)
        xlabel = f"PC1 ({explained[0]*100:.1f}%)"
        ylabel = f"PC2 ({explained[1]*100:.1f}%)"
        zlabel = f"PC3 ({explained[2]*100:.1f}%)"
        title = "질환별 보행 특성 3D 분포 (PCA)"
    else:
        data = compute_clinical_scores(clusters)
        xlabel = "이동성 (Mobility)"
        ylabel = "안정성 (Stability)"
        zlabel = "대칭성 (Symmetry)"
        title = "질환별 보행 특성 3D 분포 (임상 축)"

    fig = go.Figure()

    # 샘플 포인트
    for cluster, coords in data:
        hover_text = [
            f"<b>{cluster.name_kr}</b> ({cluster.category})<br>"
            f"{xlabel}: {coords[i, 0]:.3f}<br>"
            f"{ylabel}: {coords[i, 1]:.3f}<br>"
            f"{zlabel}: {coords[i, 2]:.3f}"
            for i in range(coords.shape[0])
        ]

        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            name=f"{cluster.name_kr} ({cluster.category})",
            marker=dict(
                size=3,
                color=cluster.color,
                opacity=0.5,
                line=dict(width=0.3, color="white"),
            ),
            hovertext=hover_text,
            hoverinfo="text",
            legendgroup=cluster.category,
        ))

        # 클러스터 중심
        center = coords.mean(axis=0)
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode="markers+text",
            name=f"{cluster.name_kr} (중심)",
            marker=dict(
                size=10,
                color=cluster.color,
                opacity=1.0,
                line=dict(width=2, color="black"),
                symbol="diamond",
            ),
            text=[cluster.name_kr],
            textposition="top center",
            textfont=dict(size=10, color=cluster.color),
            hovertext=f"<b>{cluster.name_kr}</b><br>카테고리: {cluster.category}<br>"
                      f"중심: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})",
            hoverinfo="text",
            legendgroup=cluster.category,
            showlegend=False,
        ))

    # 정상 영역 표시 (반투명 타원체)
    normal_cluster = next(c for c, _ in data if c.disease_id == 0)
    normal_coords = next(coords for c, coords in data if c.disease_id == 0)
    nc = normal_coords.mean(axis=0)
    ns = normal_coords.std(axis=0) * 2  # 2σ 영역

    # 반투명 타원체를 위한 메쉬 생성
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = nc[0] + ns[0] * np.outer(np.cos(u), np.sin(v))
    y_sphere = nc[1] + ns[1] * np.outer(np.sin(u), np.sin(v))
    z_sphere = nc[2] + ns[2] * np.outer(np.ones_like(u), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.08,
        colorscale=[[0, "#2E8B57"], [1, "#2E8B57"]],
        showscale=False,
        name="정상 범위 (2σ)",
        hoverinfo="name",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br>"
                 "<span style='font-size:12px;color:gray;'>"
                 "마우스 드래그로 회전 | 스크롤로 줌 | 호버로 상세 정보</span>",
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel,
            xaxis=dict(backgroundcolor="#f7f9fc", gridcolor="#ddd"),
            yaxis=dict(backgroundcolor="#f0f4f8", gridcolor="#ddd"),
            zaxis=dict(backgroundcolor="#f7f9fc", gridcolor="#ddd"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        legend=dict(
            groupclick="togglegroup",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ddd",
            borderwidth=1,
        ),
        width=1100,
        height=800,
        margin=dict(l=10, r=10, t=80, b=10),
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path), include_plotlyjs="cdn")
    return save_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 개인 환자 위치 표시
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_patient_in_disease_space(
    patient_features: dict[str, float],
    save_path: Path,
    mode: str = "clinical",
    n_samples: int = 80,
    random_state: int = 42,
) -> Path:
    """환자의 보행 데이터를 질환 분포 3D 공간에 표시합니다 (Plotly HTML).

    Args:
        patient_features: 13개 보행 특성 딕셔너리.
        save_path: 저장 경로 (.html).
        mode: "clinical" 또는 "pca".
        n_samples: 배경 질환 샘플 수.
        random_state: 랜덤 시드.

    Returns:
        저장된 HTML 파일 경로.
    """
    import plotly.graph_objects as go

    clusters = generate_disease_samples(n_samples, random_state)

    # 환자 특성 벡터 구성
    patient_vec = np.array([[patient_features.get(f, 0.0) for f in FEATURE_NAMES]])

    if mode == "pca":
        mean, std, components, explained = _fit_pca_3d(clusters)
        data = [
            (c, _project_to_pca(c.samples, mean, std, components))
            for c in clusters
        ]
        patient_3d = _project_to_pca(patient_vec, mean, std, components)

        xlabel = f"PC1 ({explained[0]*100:.1f}%)"
        ylabel = f"PC2 ({explained[1]*100:.1f}%)"
        zlabel = f"PC3 ({explained[2]*100:.1f}%)"
        title = "환자 위치 - 질환 분포 3D (PCA)"
    else:
        data = compute_clinical_scores(clusters)
        stats = _normalize_clinical_axes(clusters)
        patient_3d = _project_to_clinical_axes(patient_vec, stats)

        xlabel = "이동성 (Mobility)"
        ylabel = "안정성 (Stability)"
        zlabel = "대칭성 (Symmetry)"
        title = "환자 위치 - 질환 분포 3D (임상 축)"

    fig = go.Figure()

    # 질환 클러스터 (배경)
    for cluster, coords in data:
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            name=cluster.name_kr,
            marker=dict(size=2.5, color=cluster.color, opacity=0.3),
            legendgroup=cluster.category,
            hoverinfo="name",
        ))

        center = coords.mean(axis=0)
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode="text",
            text=[cluster.name_kr],
            textfont=dict(size=9, color=cluster.color),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=cluster.category,
        ))

    # 환자 위치 (강조)
    # 가장 가까운 질환 찾기
    distances = {}
    for cluster, coords in data:
        center = coords.mean(axis=0)
        dist = np.linalg.norm(patient_3d[0] - center)
        distances[cluster.name_kr] = dist
    nearest = min(distances, key=distances.get)
    nearest_dist = distances[nearest]

    fig.add_trace(go.Scatter3d(
        x=[patient_3d[0, 0]], y=[patient_3d[0, 1]], z=[patient_3d[0, 2]],
        mode="markers+text",
        name="현재 환자",
        marker=dict(
            size=14,
            color="#FF1744",
            opacity=1.0,
            line=dict(width=3, color="black"),
            symbol="cross",
        ),
        text=["현재 환자"],
        textposition="top center",
        textfont=dict(size=12, color="#FF1744", family="Arial Black"),
        hovertext=f"<b>현재 환자</b><br>"
                  f"가장 유사한 질환: {nearest} (거리: {nearest_dist:.3f})<br>"
                  f"{xlabel}: {patient_3d[0, 0]:.3f}<br>"
                  f"{ylabel}: {patient_3d[0, 1]:.3f}<br>"
                  f"{zlabel}: {patient_3d[0, 2]:.3f}",
        hoverinfo="text",
    ))

    # 환자→가장 가까운 질환 중심 연결선
    nearest_center = next(
        coords.mean(axis=0) for c, coords in data if c.name_kr == nearest
    )
    fig.add_trace(go.Scatter3d(
        x=[patient_3d[0, 0], nearest_center[0]],
        y=[patient_3d[0, 1], nearest_center[1]],
        z=[patient_3d[0, 2], nearest_center[2]],
        mode="lines",
        name=f"→ {nearest}",
        line=dict(color="#FF1744", width=3, dash="dash"),
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br>"
                 f"<span style='font-size:12px;color:#FF1744;'>"
                 f"가장 유사한 질환: {nearest}</span>",
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        width=1100,
        height=800,
        margin=dict(l=10, r=10, t=80, b=10),
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path), include_plotlyjs="cdn")
    return save_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데모 / CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_all_visualizations(
    output_dir: str = "output/disease_3d",
    n_samples: int = 80,
) -> dict[str, Path]:
    """모든 3D 질환 분포 시각화를 생성합니다.

    Returns:
        생성된 파일 경로 딕셔너리.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. 임상 축 정적 이미지
    results["clinical_png"] = plot_disease_distribution_3d(
        out / "disease_distribution_clinical.png",
        n_samples=n_samples, mode="clinical",
    )

    # 2. PCA 축 정적 이미지
    results["pca_png"] = plot_disease_distribution_3d(
        out / "disease_distribution_pca.png",
        n_samples=n_samples, mode="pca",
    )

    # 3. 임상 축 인터랙티브 HTML
    results["clinical_html"] = plot_disease_distribution_3d_interactive(
        out / "disease_distribution_clinical.html",
        n_samples=n_samples, mode="clinical",
    )

    # 4. PCA 인터랙티브 HTML
    results["pca_html"] = plot_disease_distribution_3d_interactive(
        out / "disease_distribution_pca.html",
        n_samples=n_samples, mode="pca",
    )

    # 5. 샘플 환자 위치 표시
    sample_patient = {
        "gait_speed": 0.75, "cadence": 130, "stride_regularity": 0.55,
        "step_symmetry": 0.82, "cop_sway": 0.055, "ml_variability": 0.085,
        "heel_pressure_ratio": 0.31, "forefoot_pressure_ratio": 0.47,
        "arch_index": 0.24, "pressure_asymmetry": 0.07, "acceleration_rms": 1.1,
        "acceleration_variability": 0.30, "trunk_sway": 2.7,
    }
    results["patient_html"] = plot_patient_in_disease_space(
        sample_patient,
        out / "patient_in_disease_space.html",
        mode="clinical", n_samples=n_samples,
    )

    return results


if __name__ == "__main__":
    results = generate_all_visualizations()
    print("=== 3D 질환 분포 시각화 생성 완료 ===")
    for key, path in results.items():
        print(f"  {key}: {path}")
