"""Model export utilities for deployment.

배포를 위한 모델 변환 유틸리티.
ONNX, TorchScript, FP16/INT8 양자화 변환 및 추론 속도 벤치마크를 제공합니다.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.models.multimodal_gait_net import MultimodalGaitNet

logger = logging.getLogger(__name__)


def _load_model_from_checkpoint(
    config: dict,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> MultimodalGaitNet:
    """설정과 체크포인트로부터 모델을 로드합니다.

    Load model from config dict and optional checkpoint file.
    """
    model = MultimodalGaitNet(config)
    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(
            checkpoint_path, map_location=torch.device(device), weights_only=True,
        )
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint: %s", checkpoint_path)

    model.to(device)
    model.eval()
    return model


def _create_dummy_inputs(
    config: dict,
    batch_size: int = 1,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """벤치마크 및 트레이싱을 위한 더미 입력 텐서를 생성합니다.

    Create dummy input tensors matching the model's expected shapes.
    """
    data_cfg = config["data"]
    seq_len = data_cfg["sequence_length"]
    imu_channels = data_cfg["imu_channels"]
    grid_h, grid_w = data_cfg["pressure_grid_size"]
    num_joints = data_cfg["skeleton_joints"]
    skeleton_dims = data_cfg["skeleton_dims"]

    return {
        "imu": torch.randn(batch_size, imu_channels, seq_len, device=device),
        "pressure": torch.randn(batch_size, seq_len, 1, grid_h, grid_w, device=device),
        "skeleton": torch.randn(batch_size, skeleton_dims, seq_len, num_joints, device=device),
    }


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model: MultimodalGaitNet,
    config: dict,
    output_path: str,
    opset_version: int = 17,
) -> Path:
    """모델을 ONNX 형식으로 내보냅니다.

    Export the model to ONNX format with dynamic batch-size axis.

    Args:
        model: Trained MultimodalGaitNet in eval mode.
        config: Full YAML configuration dictionary.
        output_path: Destination file path for the .onnx file.
        opset_version: ONNX opset version (default 17).

    Returns:
        Path to the saved ONNX model.
    """
    import onnx

    dummy = _create_dummy_inputs(config, batch_size=1, device="cpu")
    model = model.cpu().eval()

    # Flatten dict inputs into a tuple for torch.onnx.export
    class _Wrapper(nn.Module):
        def __init__(self, net: MultimodalGaitNet):
            super().__init__()
            self.net = net

        def forward(
            self,
            imu: torch.Tensor,
            pressure: torch.Tensor,
            skeleton: torch.Tensor,
        ) -> torch.Tensor:
            return self.net({"imu": imu, "pressure": pressure, "skeleton": skeleton})

    wrapper = _Wrapper(model)

    dynamic_axes = {
        "imu": {0: "batch"},
        "pressure": {0: "batch"},
        "skeleton": {0: "batch"},
        "output": {0: "batch"},
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy["imu"], dummy["pressure"], dummy["skeleton"]),
        str(out),
        input_names=["imu", "pressure", "skeleton"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Validate
    onnx_model = onnx.load(str(out))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX export complete: %s (%.2f MB)", out, out.stat().st_size / 1e6)
    return out


# ---------------------------------------------------------------------------
# TorchScript export
# ---------------------------------------------------------------------------

def export_torchscript_traced(
    model: MultimodalGaitNet,
    config: dict,
    output_path: str,
) -> Path:
    """TorchScript traced 모델로 내보냅니다.

    Export using torch.jit.trace for static-graph deployment.

    Args:
        model: Trained model in eval mode.
        config: Full YAML config dict.
        output_path: Destination .pt file path.

    Returns:
        Path to the saved TorchScript model.
    """

    class _Wrapper(nn.Module):
        """Wrapper that accepts positional tensor args for tracing."""

        def __init__(self, net: MultimodalGaitNet):
            super().__init__()
            self.net = net

        def forward(
            self,
            imu: torch.Tensor,
            pressure: torch.Tensor,
            skeleton: torch.Tensor,
        ) -> torch.Tensor:
            return self.net({"imu": imu, "pressure": pressure, "skeleton": skeleton})

    model = model.cpu().eval()
    wrapper = _Wrapper(model)
    dummy = _create_dummy_inputs(config, batch_size=1, device="cpu")

    traced = torch.jit.trace(
        wrapper,
        (dummy["imu"], dummy["pressure"], dummy["skeleton"]),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    logger.info("TorchScript (traced) export: %s (%.2f MB)", out, out.stat().st_size / 1e6)
    return out


def export_torchscript_scripted(
    model: MultimodalGaitNet,
    config: dict,
    output_path: str,
) -> Path:
    """TorchScript scripted 모델로 내보냅니다.

    Export using torch.jit.script for models with dynamic control flow.

    Args:
        model: Trained model in eval mode.
        config: Full YAML config dict.
        output_path: Destination .pt file path.

    Returns:
        Path to the saved TorchScript model.
    """
    model = model.cpu().eval()

    scripted = torch.jit.script(model)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out))
    logger.info("TorchScript (scripted) export: %s (%.2f MB)", out, out.stat().st_size / 1e6)
    return out


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_fp16(
    model: MultimodalGaitNet,
    config: dict,
    output_path: str,
) -> Path:
    """FP16 반정밀도 양자화를 수행합니다.

    Convert model parameters to float16 for reduced memory and faster inference
    on GPUs that support half-precision.

    Args:
        model: Trained model in eval mode.
        config: Full YAML config dict.
        output_path: Destination .pt file path.

    Returns:
        Path to the saved FP16 model checkpoint.
    """
    model = model.cpu().eval()
    model_fp16 = model.half()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_fp16.state_dict(), str(out))

    original_size = sum(p.numel() * p.element_size() for p in model.float().parameters())
    fp16_size = sum(p.numel() * p.element_size() for p in model_fp16.parameters())
    logger.info(
        "FP16 quantization: %.2f MB -> %.2f MB (%.1f%% reduction)",
        original_size / 1e6, fp16_size / 1e6,
        (1 - fp16_size / original_size) * 100,
    )
    return out


def quantize_int8(
    model: MultimodalGaitNet,
    config: dict,
    output_path: str,
) -> Path:
    """INT8 사후 학습 양자화를 수행합니다.

    Apply post-training dynamic quantization (INT8) to Linear and LSTM layers.

    Args:
        model: Trained model in eval mode.
        config: Full YAML config dict.
        output_path: Destination .pt file path.

    Returns:
        Path to the saved INT8-quantized model.
    """
    model = model.cpu().eval().float()

    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized.state_dict(), str(out))

    logger.info("INT8 dynamic quantization complete: %s", out)
    return out


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_inference(
    config: dict,
    checkpoint_path: Optional[str] = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    batch_size: int = 1,
    device: str = "cpu",
) -> dict[str, float]:
    """다양한 형식의 추론 속도를 비교 벤치마크합니다.

    Benchmark inference latency for the original PyTorch model,
    FP16 variant, and INT8 variant.

    Args:
        config: Full YAML config dict.
        checkpoint_path: Optional checkpoint to load weights from.
        num_warmup: Number of warm-up iterations.
        num_runs: Number of timed iterations.
        batch_size: Input batch size.
        device: Target device ('cpu' or 'cuda').

    Returns:
        Dictionary mapping format name to mean latency in milliseconds.
    """
    results: dict[str, float] = {}

    model = _load_model_from_checkpoint(config, checkpoint_path, device=device)
    dummy = _create_dummy_inputs(config, batch_size=batch_size, device=device)

    def _time_model(mdl: nn.Module, inputs: dict, label: str) -> float:
        for _ in range(num_warmup):
            with torch.no_grad():
                mdl(inputs)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(num_runs):
            with torch.no_grad():
                mdl(inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / num_runs * 1000  # ms

        results[label] = round(elapsed, 3)
        logger.info("%s: %.3f ms/inference", label, elapsed)
        return elapsed

    # FP32
    _time_model(model, dummy, "fp32")

    # FP16 (GPU only)
    if device == "cuda":
        model_fp16 = model.half()
        dummy_fp16 = {k: v.half() for k, v in dummy.items()}
        _time_model(model_fp16, dummy_fp16, "fp16_cuda")

    # INT8 dynamic (CPU)
    model_cpu = _load_model_from_checkpoint(config, checkpoint_path, device="cpu")
    dummy_cpu = _create_dummy_inputs(config, batch_size=batch_size, device="cpu")

    model_int8 = torch.ao.quantization.quantize_dynamic(
        model_cpu, {nn.Linear, nn.LSTM}, dtype=torch.qint8,
    )
    _time_model(model_int8, dummy_cpu, "int8_cpu")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """모델 내보내기 CLI 진입점. / CLI entrypoint for model export."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Export MultimodalGaitNet for deployment",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="exports",
        help="Output directory for exported models",
    )
    parser.add_argument(
        "--formats", nargs="+",
        default=["onnx", "traced", "scripted", "fp16", "int8"],
        choices=["onnx", "traced", "scripted", "fp16", "int8"],
        help="Export formats to produce",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run inference speed benchmark after export",
    )
    parser.add_argument(
        "--benchmark-runs", type=int, default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for benchmark",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = _load_model_from_checkpoint(config, args.checkpoint)
    out_dir = Path(args.output_dir)

    if "onnx" in args.formats:
        export_onnx(model, config, str(out_dir / "model.onnx"))

    if "traced" in args.formats:
        export_torchscript_traced(model, config, str(out_dir / "model_traced.pt"))

    if "scripted" in args.formats:
        try:
            export_torchscript_scripted(model, config, str(out_dir / "model_scripted.pt"))
        except Exception as e:
            logger.warning("TorchScript scripting failed (expected for some architectures): %s", e)

    if "fp16" in args.formats:
        quantize_fp16(model, config, str(out_dir / "model_fp16.pt"))

    if "int8" in args.formats:
        quantize_int8(model, config, str(out_dir / "model_int8.pt"))

    if args.benchmark:
        print("\n--- Inference Benchmark ---")
        results = benchmark_inference(
            config,
            args.checkpoint,
            num_runs=args.benchmark_runs,
            device=args.device,
        )
        for fmt, ms in results.items():
            print(f"  {fmt:12s}: {ms:8.3f} ms")


if __name__ == "__main__":
    main()
