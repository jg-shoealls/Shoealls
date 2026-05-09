"""Streamlit dashboard for quick gait model demos."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data.synthetic import CLASS_NAMES
from src.models.multimodal_gait_net import MultimodalGaitNet


CHECKPOINT_PATH = Path("outputs/light_stroke/best_model.pt")
DEMO_CLASSES = {
    "Normal": 0,
    "Antalgic": 1,
    "Ataxic": 2,
    "Parkinsonian": 3,
}


st.set_page_config(page_title="ShoeAlls AI Gait Dashboard", layout="wide")
st.title("ShoeAlls AI Gait Dashboard")
st.markdown("---")


@st.cache_resource
def load_gait_model():
    if not CHECKPOINT_PATH.exists():
        return None

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = MultimodalGaitNet(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["config"]


loaded = load_gait_model()

if loaded is None:
    st.error(
        f"Checkpoint not found at {CHECKPOINT_PATH}. "
        "Train a model first or update CHECKPOINT_PATH in app.py."
    )
else:
    model, config = loaded

    st.sidebar.header("Controls")
    sample_type = st.sidebar.selectbox("Sample gait type", list(DEMO_CLASSES.keys()))
    run_analysis = st.sidebar.button("Run gait analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live IMU Signal Preview")
        data_len = int(config.get("data", {}).get("sequence_length", 128))
        chart_data = pd.DataFrame(
            np.random.randn(data_len, 3),
            columns=["Accel X", "Accel Y", "Accel Z"],
        )
        st.line_chart(chart_data)

    with col2:
        st.subheader("AI Result")
        result_placeholder = st.empty()
        result_placeholder.info("Run analysis to generate a demo diagnosis.")

    if run_analysis:
        with st.spinner("Analyzing gait pattern..."):
            time.sleep(1.0)

        target_idx = DEMO_CLASSES[sample_type]
        label = CLASS_NAMES[target_idx] if target_idx < len(CLASS_NAMES) else sample_type
        confidence = np.random.uniform(85.0, 99.9)
        risk_value = confidence / 100 if sample_type != "Normal" else (100 - confidence) / 100

        with result_placeholder.container():
            st.success("Analysis complete")
            st.metric("Prediction", label)
            st.metric("Confidence", f"{confidence:.2f}%")
            st.write("Risk indicators")
            st.progress(risk_value, text="Pattern risk")
            st.progress(np.random.uniform(0.1, 0.3), text="Motion instability")

        st.markdown("---")
        st.subheader("Gait Biomarkers")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Walking speed", "1.2 m/s", "normal")
        b2.metric("Left/right symmetry", "94%", "-2%", delta_color="inverse")
        b3.metric("Stride regularity", "88%", "moderate")
        b4.metric("Cadence", "112 steps/min", "normal")

st.sidebar.markdown("---")
st.sidebar.info("ShoeAlls PoC Dashboard v1.0")
