#!/usr/bin/env python3
"""Experiment manager for Shoealls R&D and OpenClaw integration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "experiments"


def list_configs() -> list[str]:
    configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    print("Available configs:")
    for config in configs:
        print(f"  {config.stem}")
    return [config.stem for config in configs]


def show_status() -> None:
    if not OUTPUTS_DIR.exists():
        print("No experiments found.")
        return

    exps = sorted(OUTPUTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not exps:
        print("No experiments found.")
        return

    print(f"  {'Name':<38} {'Kind':<14} {'Status':<10} {'Elapsed'}")
    print("  " + "-" * 78)
    for exp in exps[:10]:
        meta_path = exp / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        elapsed = f"{meta.get('elapsed_seconds', '?')}s"
        name = meta.get("name", exp.name)
        kind = meta.get("kind", "generic")
        status = meta.get("status", "?")
        print(f"  {name:<38} {kind:<14} {status:<10} {elapsed}")


def run_experiment(
    config_name: str,
    experiment_name: str | None = None,
    pipeline: bool = True,
    kind: str = "generic",
) -> int:
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = experiment_name or f"{config_name}_{timestamp}"
    output_dir = OUTPUTS_DIR / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "config": config_name,
        "kind": kind,
        "name": exp_name,
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }
    _write_meta(output_dir, meta)

    print(f"Experiment : {exp_name}")
    print(f"Kind       : {kind}")
    print(f"Config     : {config_path}")
    print(f"Output     : {output_dir}")
    print("=" * 60)

    project_root = Path(__file__).parent.parent
    start = time.time()
    env = None

    if kind == "weargait_loso":
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "train_weargait_loso.py"),
            str(config_path),
            str(output_dir),
        ]
        env = {"PYTHONPATH": str(project_root)}
    elif pipeline:
        cmd = [sys.executable, str(project_root / "run_pipeline.py")]
    else:
        cmd = [
            sys.executable,
            "-m",
            "src.training.train",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]

    run_env = None
    if env:
        import os

        run_env = os.environ.copy()
        run_env.update(env)

    result = subprocess.run(cmd, cwd=str(project_root), env=run_env)
    elapsed = round(time.time() - start, 1)
    status = "success" if result.returncode == 0 else "failed"

    meta.update(
        {
            "finished_at": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "return_code": result.returncode,
            "status": status,
        }
    )
    _write_meta(output_dir, meta)

    print(f"\nExperiment {status}: {exp_name} ({elapsed}s)")
    return result.returncode


def batch_run(config_names: list[str], kind: str = "generic") -> list[dict]:
    results = []
    for config_name in config_names:
        print(f"\n{'=' * 60}")
        print(f"Running: {config_name}")
        print(f"{'=' * 60}")
        rc = run_experiment(config_name, pipeline=False, kind=kind)
        results.append({"config": config_name, "status": "success" if rc == 0 else "failed"})

    print("\n" + "=" * 60)
    print("Batch Results:")
    for result in results:
        print(f"  {result['config']:<30} -> {result['status']}")
    return results


def _write_meta(output_dir: Path, meta: dict) -> None:
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shoealls Experiment Manager")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available configs")
    sub.add_parser("status", help="Show recent experiment status")

    run_p = sub.add_parser("run", help="Run a single experiment")
    run_p.add_argument("config", help="Config name without .yaml")
    run_p.add_argument("--name", help="Custom experiment name")
    run_p.add_argument("--train-only", action="store_true", help="Training only, skip full pipeline")
    run_p.add_argument(
        "--kind",
        choices=["generic", "weargait_loso"],
        default="generic",
        help="Experiment runner kind",
    )

    batch_p = sub.add_parser("batch", help="Run multiple experiments sequentially")
    batch_p.add_argument("configs", nargs="+", help="Config names")
    batch_p.add_argument(
        "--kind",
        choices=["generic", "weargait_loso"],
        default="generic",
        help="Experiment runner kind",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_configs()
    elif args.command == "status":
        show_status()
    elif args.command == "run":
        sys.exit(
            run_experiment(
                args.config,
                args.name,
                pipeline=not args.train_only,
                kind=args.kind,
            )
        )
    elif args.command == "batch":
        batch_run(args.configs, kind=args.kind)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
