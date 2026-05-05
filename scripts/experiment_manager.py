#!/usr/bin/env python3
"""Experiment manager for Shoealls R&D — OpenClaw integration entry point."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "experiments"


def list_configs():
    configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    print("Available configs:")
    for c in configs:
        print(f"  {c.stem}")
    return [c.stem for c in configs]


def show_status():
    if not OUTPUTS_DIR.exists():
        print("No experiments found.")
        return

    exps = sorted(OUTPUTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not exps:
        print("No experiments found.")
        return

    print(f"  {'Name':<38} {'Status':<10} {'Elapsed'}")
    print("  " + "-" * 60)
    for exp in exps[:10]:
        meta_path = exp / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            elapsed = f"{meta.get('elapsed_seconds', '?')}s"
            name = meta.get("name", exp.name)
            status = meta.get("status", "?")
            print(f"  {name:<38} {status:<10} {elapsed}")


def run_experiment(config_name: str, experiment_name: str = None, pipeline: bool = True) -> int:
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
        "name": exp_name,
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Experiment : {exp_name}")
    print(f"Config     : {config_path}")
    print(f"Output     : {output_dir}")
    print("=" * 60)

    project_root = Path(__file__).parent.parent
    start = time.time()

    if pipeline:
        cmd = [sys.executable, str(project_root / "run_pipeline.py")]
    else:
        cmd = [
            sys.executable, "-m", "src.training.train",
            "--config", str(config_path),
            "--output-dir", str(output_dir),
        ]

    result = subprocess.run(cmd, cwd=str(project_root))
    elapsed = round(time.time() - start, 1)
    status = "success" if result.returncode == 0 else "failed"

    meta.update({
        "finished_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "return_code": result.returncode,
        "status": status,
    })
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nExperiment {status}: {exp_name} ({elapsed}s)")
    return result.returncode


def batch_run(config_names: list) -> list:
    results = []
    for config_name in config_names:
        print(f"\n{'='*60}")
        print(f"Running: {config_name}")
        print(f"{'='*60}")
        rc = run_experiment(config_name, pipeline=False)
        results.append({"config": config_name, "status": "success" if rc == 0 else "failed"})

    print("\n" + "=" * 60)
    print("Batch Results:")
    for r in results:
        print(f"  {r['config']:<30} → {r['status']}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Shoealls Experiment Manager (OpenClaw)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available configs")
    sub.add_parser("status", help="Show recent experiment status")

    run_p = sub.add_parser("run", help="Run a single experiment")
    run_p.add_argument("config", help="Config name (without .yaml)")
    run_p.add_argument("--name", help="Custom experiment name")
    run_p.add_argument("--train-only", action="store_true", help="Training only, skip full pipeline")

    batch_p = sub.add_parser("batch", help="Run multiple experiments sequentially")
    batch_p.add_argument("configs", nargs="+", help="Config names")

    args = parser.parse_args()

    if args.command == "list":
        list_configs()
    elif args.command == "status":
        show_status()
    elif args.command == "run":
        sys.exit(run_experiment(args.config, args.name, pipeline=not args.train_only))
    elif args.command == "batch":
        batch_run(args.configs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
