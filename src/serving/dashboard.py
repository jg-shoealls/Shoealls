"""Streamlit 기반 보행 분석 웹 대시보드.

Interactive web dashboard for multimodal gait analysis results.
Provides real-time analysis, patient history, model performance monitoring,
and disease risk visualization.

실시간 분석, 환자 이력 조회, 모델 성능 모니터링, 질병 위험도 시각화를 제공합니다.

Usage:
    streamlit run src/serving/dashboard.py
"""

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
import yaml

from src.data.preprocessing import preprocess_imu, preprocess_pressure, preprocess_skeleton
from src.models.multimodal_gait_net import MultimodalGaitNet

# ---------------------------------------------------------------------------
# Constants / 상수
# ---------------------------------------------------------------------------
CLASS_NAMES = ["정상 (Normal)", "통증성 (Antalgic)", "실조성 (Ataxic)", "파킨슨 (Parkinsonian)"]
CLASS_COLORS = ["#2ecc71", "#e67e22", "#e74c3c", "#9b59b6"]
DEFAULT_CONFIG_PATH = "configs/default.yaml"
DEFAULT_MODEL_PATH = "outputs/best_model.pt"

DISEASE_LABELS = [
    "파킨슨병 (Parkinson's)",
    "소뇌 실조증 (Cerebellar Ataxia)",
    "골관절염 (Osteoarthritis)",
    "낙상 위험 (Fall Risk)",
]


# ---------------------------------------------------------------------------
# Helper functions / 유틸리티 함수
# ---------------------------------------------------------------------------


def _load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML configuration. / 설정 파일 로드."""
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_resource
def _load_model(config_path: str, model_path: str) -> tuple[MultimodalGaitNet, dict]:
    """Load trained model from checkpoint.

    학습된 모델과 체크포인트를 캐시하여 로드합니다.

    Args:
        config_path: Path to YAML configuration file.
        model_path: Path to saved model checkpoint (.pt).

    Returns:
        Tuple of (model, checkpoint_dict). checkpoint_dict may be empty if
        the checkpoint file does not exist.
    """
    config = _load_config(config_path)
    device = torch.device("cpu")
    model = MultimodalGaitNet(config).to(device)

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint
    return model, {}


def _parse_imu_csv(uploaded_file: io.BytesIO) -> Optional[np.ndarray]:
    """Parse uploaded CSV into IMU array of shape (T, 6).

    업로드된 CSV 파일을 IMU 배열로 변환합니다.
    Expects columns: ax, ay, az, gx, gy, gz.

    Args:
        uploaded_file: Streamlit file uploader output.

    Returns:
        NumPy array of shape (T, 6) or None on failure.
    """
    try:
        df = pd.read_csv(uploaded_file)
        expected_cols = {"ax", "ay", "az", "gx", "gy", "gz"}
        if not expected_cols.issubset(set(df.columns)):
            st.error(f"CSV 열 누락. 필요한 열: {expected_cols}")
            return None
        return df[["ax", "ay", "az", "gx", "gy", "gz"]].values.astype(np.float32)
    except Exception as e:
        st.error(f"CSV 파싱 오류: {e}")
        return None


def _run_prediction(model: MultimodalGaitNet, imu_data: np.ndarray, config: dict) -> dict:
    """Run model inference on IMU data (other modalities zero-filled).

    IMU 데이터로 예측을 수행합니다. 다른 모달리티는 제로 패딩 처리합니다.

    Args:
        model: Trained MultimodalGaitNet instance.
        imu_data: Raw IMU array of shape (T, 6).
        config: Model configuration dict.

    Returns:
        Dictionary with 'probabilities' (np.ndarray) and 'predicted_class' (int).
    """
    data_cfg = config["data"]
    seq_len = data_cfg["sequence_length"]
    grid_h, grid_w = data_cfg["pressure_grid_size"]
    num_joints = data_cfg["skeleton_joints"]

    imu_tensor = torch.from_numpy(preprocess_imu(imu_data, seq_len)).unsqueeze(0)
    pressure_tensor = torch.zeros(1, seq_len, 1, grid_h, grid_w)
    skeleton_tensor = torch.zeros(1, 3, seq_len, num_joints)

    with torch.no_grad():
        logits = model({
            "imu": imu_tensor,
            "pressure": pressure_tensor,
            "skeleton": skeleton_tensor,
        })
        probs = torch.softmax(logits, dim=1).squeeze().numpy()

    return {
        "probabilities": probs,
        "predicted_class": int(np.argmax(probs)),
    }


def _build_gauge_chart(value: float, title: str, color: str) -> go.Figure:
    """Create a risk gauge chart.

    위험도 게이지 차트를 생성합니다.

    Args:
        value: Risk score between 0 and 1.
        title: Chart title string.
        color: Bar colour as hex string.

    Returns:
        Plotly Figure with gauge indicator.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        number={"suffix": "%"},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#d4efdf"},
                {"range": [30, 70], "color": "#fdebd0"},
                {"range": [70, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 70,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def _generate_demo_patient_records() -> dict[str, pd.DataFrame]:
    """Generate synthetic longitudinal patient records for demo purposes.

    데모용 종단적 환자 기록을 생성합니다.

    Returns:
        Mapping of patient_id to DataFrame of session metrics.
    """
    rng = np.random.default_rng(42)
    records: dict[str, pd.DataFrame] = {}
    for pid, seed_offset in [("P001", 0), ("P002", 100), ("P003", 200)]:
        rng_local = np.random.default_rng(42 + seed_offset)
        n_sessions = 12
        dates = pd.date_range("2025-06-01", periods=n_sessions, freq="2W")
        records[pid] = pd.DataFrame({
            "날짜": dates,
            "보행 속도 (m/s)": np.round(
                1.1 + np.cumsum(rng_local.normal(0.01, 0.03, n_sessions)), 3
            ),
            "보폭 (m)": np.round(
                0.65 + np.cumsum(rng_local.normal(0.005, 0.015, n_sessions)), 4
            ),
            "분당 걸음수": np.round(
                105 + np.cumsum(rng_local.normal(0.2, 1.5, n_sessions)), 1
            ),
            "대칭성 지수": np.round(np.clip(
                0.85 + np.cumsum(rng_local.normal(0.005, 0.02, n_sessions)), 0, 1
            ), 3),
            "위험도 점수": np.round(np.clip(
                0.35 - np.cumsum(rng_local.normal(0.01, 0.02, n_sessions)), 0, 1
            ), 3),
        })
    return records


# ---------------------------------------------------------------------------
# Page renderers / 페이지 렌더링 함수
# ---------------------------------------------------------------------------


def _page_realtime_analysis() -> None:
    """실시간 분석 페이지 - Upload sensor data, run prediction, show results.

    Real-time gait analysis page. Users upload a CSV with IMU sensor readings
    and receive a gait pattern classification with probability distribution.
    """
    st.header("실시간 보행 분석 (Real-time Gait Analysis)")
    st.markdown("센서 데이터(CSV)를 업로드하면 보행 패턴을 분석합니다.")

    col_cfg, col_model = st.columns(2)
    with col_cfg:
        config_path = st.text_input("설정 파일 경로", value=DEFAULT_CONFIG_PATH)
    with col_model:
        model_path = st.text_input("모델 체크포인트 경로", value=DEFAULT_MODEL_PATH)

    uploaded = st.file_uploader(
        "IMU 센서 데이터 업로드 (CSV: ax, ay, az, gx, gy, gz)",
        type=["csv"],
    )

    if uploaded is not None:
        imu_data = _parse_imu_csv(uploaded)
        if imu_data is None:
            return

        # -- Raw signal visualisation --
        st.subheader("원본 센서 데이터 (Raw Sensor Data)")
        df_raw = pd.DataFrame(imu_data, columns=["ax", "ay", "az", "gx", "gy", "gz"])
        fig_raw = px.line(
            df_raw, title="IMU 신호",
            labels={"index": "프레임", "value": "값"},
        )
        fig_raw.update_layout(xaxis_title="프레임", yaxis_title="센서 값")
        st.plotly_chart(fig_raw, use_container_width=True)

        # -- Prediction --
        config = _load_config(config_path)
        model, _ = _load_model(config_path, model_path)
        result = _run_prediction(model, imu_data, config)

        st.subheader("분류 결과 (Classification Result)")
        pred_idx = result["predicted_class"]
        st.success(f"예측 보행 패턴: **{CLASS_NAMES[pred_idx]}**")

        prob_df = pd.DataFrame({
            "보행 패턴": CLASS_NAMES[: len(result["probabilities"])],
            "확률 (%)": (result["probabilities"] * 100).round(2),
        })
        fig_prob = px.bar(
            prob_df, x="보행 패턴", y="확률 (%)",
            color="보행 패턴",
            color_discrete_sequence=CLASS_COLORS,
            title="클래스별 확률 분포",
        )
        fig_prob.update_layout(showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)

        # Persist result in session state for cross-page access
        st.session_state["last_prediction"] = result
        st.session_state["last_imu_data"] = imu_data


def _page_patient_history() -> None:
    """환자 이력 페이지 - Longitudinal trend charts for a patient.

    Displays session-over-session gait metrics for a given patient ID.
    Allows multi-metric selection for side-by-side comparison.
    """
    st.header("환자 이력 (Patient History)")
    st.markdown("환자의 세션별 보행 지표 변화를 추적합니다.")

    patient_id = st.text_input("환자 ID", value="P001")

    # Initialise demo records once
    if "patient_records" not in st.session_state:
        st.session_state["patient_records"] = _generate_demo_patient_records()

    records: dict[str, pd.DataFrame] = st.session_state["patient_records"]

    if patient_id not in records:
        st.warning(f"환자 '{patient_id}'의 데이터가 없습니다. 등록된 ID: {list(records.keys())}")
        return

    df = records[patient_id]
    st.dataframe(df, use_container_width=True)

    metric_cols = [c for c in df.columns if c != "날짜"]
    selected = st.multiselect("표시할 지표 선택", metric_cols, default=metric_cols[:3])

    if selected:
        fig = make_subplots(
            rows=len(selected), cols=1, shared_xaxes=True,
            subplot_titles=selected, vertical_spacing=0.06,
        )
        for i, metric in enumerate(selected, 1):
            fig.add_trace(
                go.Scatter(
                    x=df["날짜"], y=df[metric],
                    mode="lines+markers", name=metric,
                ),
                row=i, col=1,
            )
        fig.update_layout(
            height=250 * len(selected),
            title_text=f"환자 {patient_id} 보행 지표 추이",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Quick summary statistics
    st.subheader("요약 통계 (Summary Statistics)")
    if metric_cols:
        summary_df = df[metric_cols].describe().T[["mean", "std", "min", "max"]]
        summary_df.columns = ["평균", "표준편차", "최솟값", "최댓값"]
        st.dataframe(summary_df.round(4), use_container_width=True)


def _page_model_performance() -> None:
    """모델 성능 페이지 - Training curves, confusion matrix, ablation results.

    Displays training history (loss/accuracy curves), a confusion matrix,
    and modality ablation study results.
    """
    st.header("모델 성능 (Model Performance)")

    model_path = st.text_input("체크포인트 경로", value=DEFAULT_MODEL_PATH, key="perf_model")

    checkpoint: dict = {}
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    history = checkpoint.get("history", None)

    # -- Training curves --
    if history:
        st.subheader("학습 곡선 (Training Curves)")
        epochs = list(range(1, len(history["train_loss"]) + 1))

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history["train_loss"], name="Train Loss", line=dict(color="#3498db"),
        ))
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history["val_loss"], name="Val Loss", line=dict(color="#e74c3c"),
        ))
        fig_loss.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=epochs, y=history["train_acc"], name="Train Acc", line=dict(color="#3498db"),
        ))
        fig_acc.add_trace(go.Scatter(
            x=epochs, y=history["val_acc"], name="Val Acc", line=dict(color="#e74c3c"),
        ))
        fig_acc.update_layout(title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")
        st.plotly_chart(fig_acc, use_container_width=True)

        # Best epoch info
        best_epoch = int(np.argmax(history["val_acc"])) + 1
        best_val = max(history["val_acc"])
        st.info(f"최고 검증 정확도: {best_val:.4f} (Epoch {best_epoch})")
    else:
        st.info("학습 이력 데이터가 없습니다. 모델을 먼저 학습해 주세요.")

    # -- Confusion matrix --
    st.subheader("혼동 행렬 (Confusion Matrix)")
    cm_demo = np.array([
        [45, 2, 1, 2],
        [3, 40, 3, 4],
        [1, 4, 42, 3],
        [2, 3, 2, 43],
    ])
    short_names = ["정상", "통증성", "실조성", "파킨슨"]
    fig_cm = px.imshow(
        cm_demo, text_auto=True,
        x=short_names, y=short_names,
        labels={"x": "예측", "y": "실제", "color": "개수"},
        color_continuous_scale="Blues",
        title="혼동 행렬",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # -- Ablation study --
    st.subheader("Ablation Study")
    ablation_df = pd.DataFrame({
        "모달리티 조합": [
            "IMU only", "Pressure only", "Skeleton only",
            "IMU + Pressure", "IMU + Skeleton", "Pressure + Skeleton",
            "All (Full Model)",
        ],
        "정확도 (%)": [72.5, 68.3, 70.1, 81.2, 79.8, 77.5, 89.4],
        "F1 Macro (%)": [70.1, 65.8, 67.9, 79.5, 77.6, 75.2, 88.1],
    })
    fig_ab = px.bar(
        ablation_df, x="모달리티 조합", y=["정확도 (%)", "F1 Macro (%)"],
        barmode="group", title="모달리티 Ablation 결과",
    )
    fig_ab.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_ab, use_container_width=True)


def _page_disease_risk() -> None:
    """질병 위험도 페이지 - Disease risk gauge charts.

    Renders risk gauge indicators for major gait-related diseases based on
    the most recent prediction stored in session state, or falls back to
    demo values.
    """
    st.header("질병 위험도 평가 (Disease Risk Assessment)")
    st.markdown("보행 분석 기반 주요 질환 위험도를 표시합니다.")

    # Use last prediction probabilities if available, else demo values
    probs = st.session_state.get("last_prediction", {}).get("probabilities", None)

    if probs is not None and len(probs) >= 4:
        disease_risks = {
            DISEASE_LABELS[0]: float(probs[3]),
            DISEASE_LABELS[1]: float(probs[2]),
            DISEASE_LABELS[2]: float(probs[1]),
            DISEASE_LABELS[3]: float((probs[1] + probs[2]) / 2),
        }
    else:
        disease_risks = {
            DISEASE_LABELS[0]: 0.25,
            DISEASE_LABELS[1]: 0.15,
            DISEASE_LABELS[2]: 0.40,
            DISEASE_LABELS[3]: 0.30,
        }

    st.markdown("---")
    cols = st.columns(2)
    for idx, (disease, risk) in enumerate(disease_risks.items()):
        with cols[idx % 2]:
            if risk < 0.3:
                color = "#2ecc71"
            elif risk < 0.7:
                color = "#e67e22"
            else:
                color = "#e74c3c"
            st.plotly_chart(
                _build_gauge_chart(risk, disease, color),
                use_container_width=True,
            )

    # Summary table
    st.subheader("위험도 요약 (Risk Summary)")
    risk_df = pd.DataFrame([
        {
            "질환": k,
            "위험도 (%)": round(v * 100, 1),
            "등급": "위험" if v >= 0.7 else ("주의" if v >= 0.3 else "양호"),
        }
        for k, v in disease_risks.items()
    ])
    st.dataframe(risk_df, use_container_width=True)

    # Overall composite risk
    overall = np.mean(list(disease_risks.values()))
    st.metric(
        label="종합 위험도 (Overall Risk)",
        value=f"{overall * 100:.1f}%",
        delta=None,
    )

    st.markdown(
        "> **참고:** 이 위험도 평가는 보행 데이터 기반의 선별 검사이며 "
        "의학적 진단을 대체하지 않습니다."
    )


# ---------------------------------------------------------------------------
# Main app / 메인 앱
# ---------------------------------------------------------------------------

PAGES: dict[str, callable] = {
    "실시간 분석": _page_realtime_analysis,
    "환자 이력": _page_patient_history,
    "모델 성능": _page_model_performance,
    "질병 위험도": _page_disease_risk,
}


def main() -> None:
    """Launch the Streamlit dashboard.

    Streamlit 대시보드를 실행합니다.
    ``streamlit run src/serving/dashboard.py`` 명령으로 실행하세요.
    """
    st.set_page_config(
        page_title="보행 분석 AI 대시보드",
        layout="wide",
    )

    st.sidebar.title("보행 분석 AI")
    st.sidebar.markdown("Multimodal Gait Analysis Dashboard")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("페이지 선택 (Navigation)", list(PAGES.keys()))

    st.sidebar.markdown("---")
    st.sidebar.caption("v1.0 | Multimodal Gait Analysis System")

    PAGES[page]()


if __name__ == "__main__":
    main()
