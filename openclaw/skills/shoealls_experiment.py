#!/usr/bin/env python3
"""OpenClaw AgentSkill: Shoealls experiment runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MANAGER = PROJECT_ROOT / "scripts" / "experiment_manager.py"


def run(params: dict) -> str:
    """OpenClaw AgentSkill entry point."""
    command = params.get("command", "list")

    if command == "list":
        return _run_manager(["list"])

    if command == "status":
        return _run_manager(["status"])

    if command == "run":
        config = params.get("config", "default")
        cmd = ["run", config]
        if params.get("name"):
            cmd += ["--name", params["name"]]
        if params.get("train_only"):
            cmd += ["--train-only"]
        if params.get("kind"):
            cmd += ["--kind", params["kind"]]
        return _run_manager(cmd)

    if command == "batch":
        configs = params.get("configs", ["default"])
        cmd = ["batch", *configs]
        if params.get("kind"):
            cmd += ["--kind", params["kind"]]
        return _run_manager(cmd)

    return "Unknown command. Use: list | status | run | batch"


def _run_manager(args: list[str]) -> str:
    result = subprocess.run(
        [sys.executable, str(MANAGER), *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return result.stdout + (result.stderr if result.returncode != 0 else "")


if __name__ == "__main__":
    params = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(run(params))
