import torch
from src.models.fusion import CrossModalAttentionFusion

def test_fusion():
    fusion = CrossModalAttentionFusion(embed_dim=16, num_heads=2, num_modalities=2)
    features = [
        torch.randn(2, 5, 16),
        torch.randn(2, 3, 16)
    ]
    out = fusion(features)
    assert out.shape == (2, 16)
    print("CrossModalAttentionFusion test passed!")

if __name__ == "__main__":
    test_fusion()
