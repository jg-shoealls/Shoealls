#!/usr/bin/env python3
"""
OpenClaw AgentSkill: Shoealls Experiment Runner

OpenClaw이 이 스킬을 호출하여 슈올즈 보행 분석 실험을 자동 실행합니다.

Supported commands (via params dict):
    list   - 사용 가능한 config 목록
    status - 최근 실험 상태 조회
    run    - 단일 실험 실행 (config, name, train_only 옵션)
    batch  - 복수 실험 순차 실행 (configs 리스트)
"""

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
        result = subprocess.run(
            [sys.executable, str(MANAGER), "list"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        return result.stdout or result.stderr

    if command == "status":
        result = subprocess.run(
            [sys.executable, str(MANAGER), "status"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        return result.stdout or result.stderr

    if command == "run":
        config = params.get("config", "default")
        cmd = [sys.executable, str(MANAGER), "run", config]
        if params.get("name"):
            cmd += ["--name", params["name"]]
        if params.get("train_only"):
            cmd += ["--train-only"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        return result.stdout + (result.stderr if result.returncode != 0 else "")

    if command == "batch":
        configs = params.get("configs", ["default"])
        cmd = [sys.executable, str(MANAGER), "batch"] + configs
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        return result.stdout + (result.stderr if result.returncode != 0 else "")

    return f"Unknown command: '{command}'. Use: list | status | run | batch"


if __name__ == "__main__":
    # Direct CLI: python shoealls_experiment.py '{"command": "list"}'
    params = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(run(params))
