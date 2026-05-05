"""Optimize and quantize the gait model for mobile/edge deployment."""

import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from src.models.multimodal_gait_net import MultimodalGaitNet

def quantize_model(checkpoint_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load original model
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = MultimodalGaitNet(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    orig_size = os.path.getsize(checkpoint_path) / 1024
    print(f"Original model size: {orig_size:.2f} KB")
    
    # 2. Dynamic Quantization (INT8)
    print("Applying dynamic quantization (INT8)...")
    # Quantize Linear and LSTM layers which are the bottlenecks
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.LSTM, nn.Conv1d, nn.Conv2d}, 
        dtype=torch.qint8
    )
    
    # 3. Save Quantized Model
    q_path = output_dir / "model_quantized.pt"
    torch.save(quantized_model.state_dict(), q_path)
    q_size = os.path.getsize(q_path) / 1024
    print(f"Quantized model size: {q_size:.2f} KB (Reduction: {(1 - q_size/orig_size)*100:.1f}%)")
    
    # 4. Export to TorchScript for mobile
    print("Exporting to TorchScript (Tracing)...")
    # Create dummy input for tracing
    dummy_input = {
        "imu": torch.randn(1, 6, 128),
        "pressure": torch.randn(1, 128, 1, 16, 8),
        "skeleton": torch.randn(1, 3, 128, 17)
    }
    
    try:
        # Scripting is safer for models with control flow like LSTMs in some torch versions
        traced_model = torch.jit.trace(model, (dummy_input,))
        ts_path = output_dir / "model_mobile.pt"
        traced_model.save(ts_path)
        print(f"Mobile TorchScript model saved to {ts_path}")
        
        # Test latency
        start = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        cpu_time = (time.time() - start) / 100
        
        start = time.time()
        for _ in range(100):
            _ = traced_model(dummy_input)
        ts_time = (time.time() - start) / 100
        
        print(f"Inference Latency (CPU): {cpu_time*1000:.2f} ms")
        print(f"Inference Latency (TorchScript): {ts_time*1000:.2f} ms")
        print(f"Speedup: {cpu_time/ts_time:.2f}x")
        
    except Exception as e:
        print(f"TorchScript export failed: {e}")

if __name__ == "__main__":
    # Quantize the best lightweight model we have (Stroke model performed well)
    ckpt = "outputs/light_stroke/best_model.pt"
    if os.path.exists(ckpt):
        quantize_model(ckpt, "outputs/mobile_optimized")
    else:
        print("Checkpoint not found. Run training first.")
