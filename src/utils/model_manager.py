import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

logger = logging.getLogger(__name__)

class ModelManager:
    """슈올즈 AI 모델 버전 관리 및 레지스트리 시스템."""

    def __init__(self, registry_dir: str = "outputs/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.registry_dir / "model_manifest.json"
        self._load_manifest()

    def _load_manifest(self):
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {"models": {}, "aliases": {"latest": None, "production": None}}

    def _save_manifest(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def save_model(
        self,
        model_state: Dict,
        config: Dict,
        metrics: Dict,
        version: str,
        model_type: str = "reasoning",
        alias: Optional[str] = None
    ) -> str:
        """모델 가중치와 메타데이터를 저장하고 레지스트리에 등록합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_type}_v{version}_{timestamp}"
        model_path = self.registry_dir / f"{model_id}.pt"

        # 메타데이터 결합
        checkpoint = {
            "model_id": model_id,
            "model_type": model_type,
            "version": version,
            "timestamp": timestamp,
            "config": config,
            "metrics": metrics,
            "model_state_dict": model_state,
        }

        # 파일 저장
        torch.save(checkpoint, model_path)
        
        # 레지스트리 업데이트
        self.manifest["models"][model_id] = {
            "path": str(model_path),
            "version": version,
            "type": model_type,
            "metrics": metrics,
            "timestamp": timestamp
        }
        
        # 기본 별칭 설정
        self.manifest["aliases"]["latest"] = model_id
        if alias:
            self.manifest["aliases"][alias] = model_id

        self._save_manifest()
        logger.info(f"Model saved and registered: {model_id}")
        return model_id

    def get_model_path(self, alias_or_id: str) -> Optional[Path]:
        """별칭(latest, production) 또는 모델 ID로 파일 경로를 반환합니다."""
        model_id = self.manifest["aliases"].get(alias_or_id) or alias_or_id
        if model_id in self.manifest["models"]:
            path = Path(self.manifest["models"][model_id]["path"])
            if path.exists():
                return path
        return None

    def list_models(self) -> List[Dict]:
        """등록된 모든 모델 목록을 반환합니다."""
        models = []
        for mid, info in self.manifest["models"].items():
            models.append({"id": mid, **info})
        return sorted(models, key=lambda x: x["timestamp"], reverse=True)

    def set_alias(self, model_id: str, alias: str):
        """특정 모델에 별칭(예: production)을 부여합니다."""
        if model_id in self.manifest["models"]:
            self.manifest["aliases"][alias] = model_id
            self._save_manifest()
            logger.info(f"Alias '{alias}' set to model: {model_id}")
        else:
            raise ValueError(f"Model ID {model_id} not found in manifest")

    def load_checkpoint(self, alias_or_id: str, device: str = "cpu") -> Dict:
        """체크포인트 전체를 로드합니다."""
        path = self.get_model_path(alias_or_id)
        if not path:
            raise FileNotFoundError(f"Model not found for: {alias_or_id}")
        return torch.load(path, map_location=device, weights_only=True)

# 싱글톤 인스턴스
model_manager = ModelManager()
