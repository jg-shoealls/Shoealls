"""Latent Space Analysis Module.

학습된 모델의 잠재 공간(Latent Space)을 시각화하여
정상 보행과 질환 보행의 분리 정도를 분석합니다.
"""

import torch
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from pathlib import Path

class LatentVisualizer:
    """모델의 특징 벡터를 추출하고 3D 시각화를 생성합니다."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def get_embeddings(self, dataloader):
        """데이터로더에서 모든 샘플의 특징 벡터와 레이블을 추출합니다."""
        embeddings = []
        labels = []
        
        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            target = batch.pop("label")
            
            # 특징 추출 (분류기 직전 레이어)
            feat = self.model.extract_features(batch)
            
            embeddings.append(feat.cpu().numpy())
            labels.append(target.cpu().numpy())
            
        return np.concatenate(embeddings), np.concatenate(labels)

    def visualize_3d(self, embeddings, labels, method="tsne", title="Latent Space 3D"):
        """3D 산점도를 생성합니다."""
        if method == "tsne":
            reducer = TSNE(n_components=3, perplexity=30, random_state=42)
        else:
            reducer = PCA(n_components=3)
            
        coords = reducer.fit_transform(embeddings)
        
        # 클래스 이름 매핑
        class_names = ["Healthy Control", "Antalgic", "Ataxic", "Parkinsonian"]
        colors = ["#2E8B57", "#3498DB", "#F39C12", "#E74C3C"]
        
        fig = go.Figure()
        
        for i, name in enumerate(class_names):
            mask = (labels == i)
            if not np.any(mask): continue
            
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                name=name,
                marker=dict(
                    size=5,
                    color=colors[i],
                    opacity=0.8,
                    line=dict(width=0.5, color="white")
                )
            ))
            
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Comp 1",
                yaxis_title="Comp 2",
                zaxis_title="Comp 3"
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig

def run_latent_analysis(model_path, data_dir, output_path="latent_space.html"):
    """전체 분석 파이프라인 실행."""
    from src.data.weargait_adapter import WearGaitPDAdapter
    from src.utils.model_manager import model_manager
    
    # 1. 모델 및 설정 로드
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    
    from src.models.multimodal_gait_net import MultimodalGaitNet
    model = MultimodalGaitNet(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 2. 데이터 로드
    adapter = WearGaitPDAdapter(data_dir)
    dataset = adapter.to_dataset(sequence_length=config["data"]["sequence_length"])
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 3. 임베딩 추출 및 시각화
    visualizer = LatentVisualizer(model)
    embeddings, labels = visualizer.get_embeddings(loader)
    
    fig = visualizer.visualize_3d(embeddings, labels, method="tsne", title="WearGait-PD Latent Space (t-SNE)")
    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")
    return fig

if __name__ == "__main__":
    # 로컬 테스트용
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()
    
    run_latent_analysis(args.checkpoint, args.data_dir)
