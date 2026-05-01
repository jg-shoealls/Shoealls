"""Compare latency between FP32 lightweight model and INT8 quantized model."""

import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from src.models.multimodal_gait_net import MultimodalGaitNet

def measure_latency():
    ckpt_path = Path("outputs/light_stroke/best_model.pt")
    if not ckpt_path.exists():
        print("Error: Base model checkpoint not found.")
        return

    # 1. Load Original Model (FP32)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_fp32 = MultimodalGaitNet(config)
    model_fp32.load_state_dict(checkpoint["model_state_dict"])
    model_fp32.eval()

    # 2. Create Quantized Model (INT8)
    # Note: We apply dynamic quantization to the loaded FP32 model
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, 
        {nn.Linear, nn.LSTM}, 
        dtype=torch.qint8
    )

    # 3. Prepare Dummy Input
    dummy_input = {
        "imu": torch.randn(1, 6, 128),
        "pressure": torch.randn(1, 128, 1, 16, 8),
        "skeleton": torch.randn(1, 3, 128, 17)
    }

    # 4. Warm-up
    print("Warming up models...")
    for _ in range(50):
        _ = model_fp32(dummy_input)
        _ = model_int8(dummy_input)

    # 5. Measure Latency
    num_iters = 500
    print(f"Measuring latency over {num_iters} iterations...")

    # FP32 Latency
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model_fp32(dummy_input)
    end = time.perf_counter()
    fp32_time = (end - start) / num_iters * 1000  # ms

    # INT8 Latency
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model_int8(dummy_input)
    end = time.perf_counter()
    int8_time = (end - start) / num_iters * 1000  # ms

    # 6. Results
    print("\n" + "=" * 45)
    print(f"{'Metric':<25} | {'FP32':<8} | {'INT8':<8}")
    print("-" * 45)
    print(f"{'Avg Latency (ms)':<25} | {fp32_time:>8.2f} | {int8_time:>8.2f}")
    print(f"{'Throughput (inf/sec)':<25} | {1000/fp32_time:>8.1f} | {1000/int8_time:>8.1f}")
    print("-" * 45)
    
    speedup = fp32_time / int8_time
    print(f"Speedup: {speedup:.2f}x")
    print(f"Latency Reduction: {(1 - 1/speedup)*100:.1f}%")
    print("=" * 45)

    # 7. Consistency Check
    with torch.no_grad():
        out_fp32 = model_fp32(dummy_input)
        out_int8 = model_int8(dummy_input)
        cos_sim = torch.nn.functional.cosine_similarity(out_fp32, out_int8).item()
        print(f"Output Consistency (Cosine Similarity): {cos_sim:.4f}")

if __name__ == "__main__":
    measure_latency()
