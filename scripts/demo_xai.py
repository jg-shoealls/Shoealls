"""XAI 데모: Attention 시각화, Grad-CAM, 모달리티 기여도 분석.

사용법:
    python -m scripts.demo_xai
    python scripts/demo_xai.py
"""

import sys
from pathlib import Path

import torch
import yaml

# 프로젝트 루트를 경로에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.synthetic import generate_synthetic_dataset
from src.data.preprocessing import preprocess_imu, preprocess_pressure, preprocess_skeleton
from src.models.reasoning_engine import GaitReasoningEngine
from src.validation.xai_visualize import (
    plot_cross_modal_attention,
    compute_pressure_gradcam,
    plot_pressure_gradcam,
    compute_zone_importance,
    plot_modality_contribution,
    plot_xai_dashboard,
)


CLASS_NAMES_KR = ["정상 보행", "절뚝거림", "운동실조", "파킨슨"]


def main():
    # 설정 로드
    config_path = ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 출력 디렉토리
    output_dir = ROOT / "outputs" / "xai"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 엔진 초기화
    engine = GaitReasoningEngine(config)
    engine.eval()

    # 합성 데이터 생성
    data = generate_synthetic_dataset(num_samples_per_class=1, seed=42)
    class_names = data["class_names"]

    print("=" * 70)
    print("  XAI (설명 가능한 AI) 시각화 데모")
    print("=" * 70)

    for i in range(len(data["labels"])):
        label = data["labels"][i]
        name = class_names[label]
        name_kr = CLASS_NAMES_KR[label]

        print(f"\n{'─' * 70}")
        print(f"  샘플 {i + 1}: {name_kr} (class {label}: {name})")
        print(f"{'─' * 70}")

        # 전처리
        imu = preprocess_imu(data["imu"][i], 128)
        pressure = preprocess_pressure(data["pressure"][i], 128, (16, 8))
        skeleton = preprocess_skeleton(data["skeleton"][i], 128, 17)

        batch = {
            "imu": torch.from_numpy(imu).unsqueeze(0),
            "pressure": torch.from_numpy(pressure).unsqueeze(0),
            "skeleton": torch.from_numpy(skeleton).unsqueeze(0),
        }

        # 추론 실행
        result = engine.reason(batch)
        pred = result["prediction"][0].item()
        probs = result["calibrated_probs"][0].cpu().numpy()

        print(f"  예측: {CLASS_NAMES_KR[pred]} ({probs[pred]:.1%})")
        print(f"  불확실성: {result['uncertainty'][0].item():.1%}")

        # ── 1. Cross-modal attention 히트맵 ──
        cross_attn = result["evidence"]["cross_attn_weights"].cpu().numpy()
        attn_path = output_dir / f"cross_attention_{name}.png"
        plot_cross_modal_attention(
            cross_attn, attn_path,
            title=f"교차 모달 Attention — {name_kr}",
        )
        print(f"  [1] Cross-modal attention 저장: {attn_path}")

        # Attention 분석 요약
        avg_attn = cross_attn[0].mean(axis=0)
        for m_idx, m_name in enumerate(["IMU", "Pressure", "Skeleton"]):
            top_key = avg_attn[m_idx].argmax()
            top_names = ["IMU", "Pressure", "Skeleton"]
            print(f"      {m_name} → 가장 많이 참조: {top_names[top_key]} ({avg_attn[m_idx, top_key]:.2f})")

        # ── 2. Grad-CAM 압력 센서 ──
        gradcam_path = output_dir / f"gradcam_pressure_{name}.png"
        try:
            gradcam = compute_pressure_gradcam(engine, batch, target_class=pred)
            plot_pressure_gradcam(
                gradcam, batch["pressure"].numpy(), gradcam_path,
                title=f"Grad-CAM 압력 센서 — {name_kr}",
            )
            print(f"  [2] Grad-CAM 저장: {gradcam_path}")

            # 영역별 중요도 요약
            zone_scores = compute_zone_importance(gradcam)
            ranked = sorted(zone_scores.items(), key=lambda x: x[1]["importance"], reverse=True)
            print("      영역별 중요도 (상위 3):")
            for zone_name, info in ranked[:3]:
                print(f"        {info['label']}: {info['importance']:.3f} (peak: {info['peak']:.3f})")
        except Exception as e:
            print(f"  [2] Grad-CAM 실패: {e}")

        # ── 3. 모달리티 기여도 분석 ──
        contrib_path = output_dir / f"modality_contribution_{name}.png"
        plot_modality_contribution(
            result["evidence"]["modality_weights"].cpu().numpy(),
            result["evidence"]["cross_support"].cpu().numpy(),
            result["anomaly_results"],
            contrib_path,
            title=f"모달리티별 기여도 — {name_kr}",
        )
        print(f"  [3] 모달리티 기여도 저장: {contrib_path}")

        weights = result["evidence"]["modality_weights"][0].cpu().numpy()
        for m_idx, m_name in enumerate(CLASS_NAMES_KR[:3] if len(CLASS_NAMES_KR) >= 3 else ["IMU", "Pressure", "Skeleton"]):
            m_names = ["IMU (관성센서)", "족저압 센서", "스켈레톤"]
            print(f"      {m_names[m_idx]}: {weights[m_idx]:.1%}")

        # ── 4. 종합 XAI 대시보드 ──
        dashboard_path = output_dir / f"xai_dashboard_{name}.png"
        plot_xai_dashboard(result, batch, engine, dashboard_path)
        print(f"  [4] XAI 대시보드 저장: {dashboard_path}")

    print(f"\n{'=' * 70}")
    print(f"  모든 XAI 시각화 완료! 출력 디렉토리: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
