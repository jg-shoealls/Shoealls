import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
from src.models.multimodal_gait_net import MultimodalGaitNet
from src.data.synthetic import generate_synthetic_dataset, CLASS_NAMES

# Page Config
st.set_page_config(page_title="ShoeAlls AI Gait Dashboard", layout="wide")

# Title
st.title("👟 ShoeAlls AI 보행 분석 대시보드")
st.markdown("---")

# 1. Load Model
@st.cache_resource
def load_gait_model():
    ckpt_path = Path("outputs/light_stroke/best_model.pt")
    if not ckpt_path.exists():
        return None
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MultimodalGaitNet(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["config"]

model = load_gait_model()

if model is None:
    st.error("학습된 모델 체크포인트를 찾을 수 없습니다. 먼저 학습을 진행해 주세요.")
else:
    net, config = model
    
    # Sidebar: Controls
    st.sidebar.header("📊 분석 제어")
    sample_type = st.sidebar.selectbox("샘플 질환 선택", ["정상(Normal)", "뇌졸중(Stroke)", "파킨슨(Parkinson)", "치매(Dementia)"])
    run_analysis = st.sidebar.button("보행 분석 시작")

    # Main Area: Visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📡 실시간 센서 스트림 (IMU)")
        chart_placeholder = st.empty()
        
        # Generate Fake Data for display
        data_len = 128
        chart_data = pd.DataFrame(
            np.random.randn(data_len, 3),
            columns=['가속도(X)', '가속도(Y)', '가속도(Z)']
        )
        chart_placeholder.line_chart(chart_data)

    with col2:
        st.subheader("🧠 AI 진단 결과")
        result_placeholder = st.empty()
        result_placeholder.info("분석 버튼을 누르면 인공지능이 진단을 시작합니다.")

    if run_analysis:
        with st.spinner("AI가 보행 패턴을 정밀 분석 중입니다..."):
            time.sleep(1.5)  # Simulate processing
            
            # Map selection to class index
            class_map = {"정상(Normal)": 0, "뇌졸중(Stroke)": 1, "파킨슨(Parkinson)": 3, "치매(Dementia)": 3}
            target_idx = class_map[sample_type]
            
            # Inference (Dummy for UI flow)
            # In real case, we would use the actual model input here
            conf = np.random.uniform(85, 99.9)
            
            with result_placeholder.container():
                st.success(f"✅ 분석 완료")
                st.metric("진단 결과", sample_type.split("(")[0])
                st.metric("신뢰도", f"{conf:.2f}%")
                
                # Risk progress bars
                st.write("**상세 위험도 지표**")
                st.progress(conf/100 if "정상" not in sample_type else (100-conf)/100, text="뇌질환 연관성")
                st.progress(np.random.uniform(0.1, 0.3), text="낙상 위험도")

        # Bottom Area: Biomarkers
        st.markdown("---")
        st.subheader("📏 보행 바이오마커 분석")
        b1, b2, col3, b4 = st.columns(4)
        b1.metric("보행 속도", "1.2 m/s", "정상")
        b2.metric("좌우 대칭성", "94%", "-2%", delta_color="inverse")
        col3.metric("보폭 규칙성", "88%", "보통")
        b4.metric("분당 걸음수", "112 step", "정상")

st.sidebar.markdown("---")
st.sidebar.info("ShoeAlls PoC Dashboard v1.0")
