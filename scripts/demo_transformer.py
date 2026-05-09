"""Demo showing Transformer architecture for gait analysis."""

import torch
import torch.nn as nn
import time
import numpy as np

class GaitTransformer(nn.Module):
    """A pure Transformer-based model for multimodal gait analysis."""
    def __init__(self, imu_dim=6, embed_dim=64, num_heads=4, num_layers=2, num_classes=11):
        super().__init__()
        # 1. Linear Projection: Convert raw sensor data to embedding space
        self.imu_proj = nn.Linear(imu_dim, embed_dim)
        
        # 2. Positional Encoding: Give Transformer a sense of time/order
        self.pos_embedding = nn.Parameter(torch.randn(1, 128, embed_dim))
        
        # 3. Transformer Encoder: The heart of the model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Global Average Pooling & Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, time=128, channels=6)
        x = self.imu_proj(x)                  # (B, T, E)
        x = x + self.pos_embedding            # Add positional info
        
        # Self-Attention happens here!
        x = self.transformer(x)               # (B, T, E)
        
        # Take the mean across time dimension
        x = x.mean(dim=1)                     # (B, E)
        return self.classifier(x)

def run_transformer_demo():
    print("=" * 60)
    print("TRANSFORMER GAIT ANALYSIS DEMO")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitTransformer().to(device)
    
    # Calculate parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Type: Pure Transformer")
    print(f"Parameters: {params:,}")
    print(f"Device:     {device}")
    print("-" * 60)

    # Fake IMU data (Batch=1, Time=128, Channels=6)
    dummy_imu = torch.randn(1, 128, 6).to(device)
    
    print("1. Input Data: IMU Sensor Stream (Acc + Gyro)")
    print(f"   Shape: {dummy_imu.shape} (1 Sample, 128 Timesteps, 6 Sensors)")
    
    print("\n2. Processing with Multi-Head Self-Attention...")
    t0 = time.time()
    with torch.no_grad():
        logits = model(dummy_imu)
        probs = torch.softmax(logits, dim=1)
    elapsed = (time.time() - t0) * 1000
    
    print(f"\n3. Analysis Result:")
    top_prob, top_class = torch.max(probs, dim=1)
    print(f"   Predicted Disease Class: {top_class.item()}")
    print(f"   Confidence:             {top_prob.item():.1%}")
    print(f"   Inference Time:         {elapsed:.2f} ms")
    
    print("\n" + "=" * 60)
    print("KEY TRANSFORMER FEATURES DEMONSTRATED:")
    print("✔ Self-Attention: AI looks at the whole gait cycle at once")
    print("✔ Parallelism: Faster than LSTM because it doesn't process one-by-one")
    print("✔ Scalability: Can be much larger and smarter than old CNN/RNN models")
    print("=" * 60)

if __name__ == "__main__":
    run_transformer_demo()
