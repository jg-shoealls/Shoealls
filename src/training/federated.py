"""연합 학습 (Federated Learning) 지원 모듈.

FedAvg 알고리즘 기반으로 여러 클라이언트(병원/기관)에서 데이터를 공유하지 않고
모델을 협력 학습할 수 있는 프레임워크.

Federated learning support module using FedAvg algorithm.
Enables collaborative model training across multiple clients (hospitals)
without sharing raw patient data.
"""

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


class FederatedServer:
    """연합 학습 서버: 글로벌 모델 관리 및 가중치 집계.

    Federated learning server that manages the global model
    and aggregates client updates using FedAvg.
    """

    def __init__(self, global_model: nn.Module, device: str = "cpu"):
        self.global_model = global_model.to(device)
        self.device = torch.device(device)
        self.round_history = []

    def get_global_weights(self) -> dict:
        """글로벌 모델 가중치 반환."""
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(
        self,
        client_weights: list[dict],
        client_sizes: list[int],
    ) -> dict:
        """FedAvg: 클라이언트 모델 가중치를 데이터 크기 기반으로 가중 평균.

        Args:
            client_weights: 각 클라이언트의 state_dict 리스트.
            client_sizes: 각 클라이언트의 데이터 크기.

        Returns:
            집계된 글로벌 state_dict.
        """
        total_size = sum(client_sizes)
        weights_ratio = [s / total_size for s in client_sizes]

        aggregated = {}
        for key in client_weights[0].keys():
            aggregated[key] = sum(
                w[key].float() * ratio
                for w, ratio in zip(client_weights, weights_ratio)
            )

        self.global_model.load_state_dict(aggregated)
        return aggregated

    def evaluate(self, test_loader: DataLoader) -> dict:
        """글로벌 모델 평가."""
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("label")
                logits = self.global_model(batch)
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        return {"accuracy": acc, "loss": avg_loss, "num_samples": total}


class FederatedClient:
    """연합 학습 클라이언트: 로컬 데이터로 학습 후 가중치 전송.

    Federated learning client that trains on local data
    and sends model updates to the server.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        device: str = "cpu",
        dp_noise_scale: float = 0.0,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = torch.device(device)
        self.dp_noise_scale = dp_noise_scale
        self.train_history = []

    def receive_global_weights(self, global_weights: dict):
        """서버로부터 글로벌 가중치 수신."""
        self.model.load_state_dict(copy.deepcopy(global_weights))

    def local_train(
        self,
        epochs: int = 5,
        lr: float = 0.001,
        weight_decay: float = 0.01,
    ) -> dict:
        """로컬 데이터로 모델 학습.

        Args:
            epochs: 로컬 학습 에포크 수.
            lr: 학습률.
            weight_decay: 가중치 감쇠.

        Returns:
            학습 결과 딕셔너리.
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(epochs):
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("label")

                logits = self.model(batch)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()

                # 차분 프라이버시: 그래디언트에 노이즈 추가
                if self.dp_noise_scale > 0:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * self.dp_noise_scale
                            param.grad.add_(noise)

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_correct += (logits.argmax(dim=-1) == labels).sum().item()
                total_samples += labels.size(0)

        result = {
            "client_id": self.client_id,
            "loss": total_loss / total_samples if total_samples > 0 else 0.0,
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
            "num_samples": total_samples,
        }
        self.train_history.append(result)
        return result

    def get_model_weights(self) -> dict:
        """학습된 로컬 모델 가중치 반환."""
        return copy.deepcopy(self.model.state_dict())

    @property
    def data_size(self) -> int:
        return len(self.train_loader.dataset)


def simulate_federated_learning(
    model_class,
    config: dict,
    dataset,
    num_clients: int = 4,
    num_rounds: int = 10,
    local_epochs: int = 3,
    lr: float = 0.001,
    dp_noise_scale: float = 0.0,
    test_loader: Optional[DataLoader] = None,
    device: str = "cpu",
) -> dict:
    """연합 학습 시뮬레이션.

    데이터셋을 num_clients개로 분할하고 FedAvg로 학습을 시뮬레이션합니다.

    Args:
        model_class: 모델 클래스 (config를 인자로 받는 생성자).
        config: 모델 설정.
        dataset: 전체 데이터셋.
        num_clients: 클라이언트 수.
        num_rounds: 연합 학습 라운드 수.
        local_epochs: 클라이언트당 로컬 학습 에포크.
        lr: 학습률.
        dp_noise_scale: 차분 프라이버시 노이즈 스케일 (0이면 비활성).
        test_loader: 글로벌 모델 평가용 테스트 로더.
        device: 디바이스.

    Returns:
        학습 이력 딕셔너리.
    """
    # 데이터 분할: 각 클라이언트에 비중복 할당
    total = len(dataset)
    indices = np.random.permutation(total).tolist()
    split_sizes = [total // num_clients] * num_clients
    for i in range(total % num_clients):
        split_sizes[i] += 1

    client_indices = []
    offset = 0
    for size in split_sizes:
        client_indices.append(indices[offset:offset + size])
        offset += size

    # 서버 초기화
    global_model = model_class(config)
    server = FederatedServer(global_model, device=device)

    # 클라이언트 초기화
    clients = []
    for i in range(num_clients):
        client_dataset = Subset(dataset, client_indices[i])
        client_loader = DataLoader(client_dataset, batch_size=16, shuffle=True)
        client_model = model_class(config)
        client = FederatedClient(
            client_id=f"client_{i}",
            model=client_model,
            train_loader=client_loader,
            device=device,
            dp_noise_scale=dp_noise_scale,
        )
        clients.append(client)

    # 연합 학습 실행
    history = {"rounds": [], "global_accuracy": [], "global_loss": []}

    for round_idx in range(num_rounds):
        global_weights = server.get_global_weights()

        # 각 클라이언트 로컬 학습
        client_results = []
        client_weights = []
        client_sizes = []

        for client in clients:
            client.receive_global_weights(global_weights)
            result = client.local_train(epochs=local_epochs, lr=lr)
            client_results.append(result)
            client_weights.append(client.get_model_weights())
            client_sizes.append(client.data_size)

        # 서버에서 집계
        server.aggregate(client_weights, client_sizes)

        # 글로벌 모델 평가
        round_info = {"round": round_idx + 1, "client_results": client_results}
        if test_loader is not None:
            eval_result = server.evaluate(test_loader)
            round_info["global_eval"] = eval_result
            history["global_accuracy"].append(eval_result["accuracy"])
            history["global_loss"].append(eval_result["loss"])

        history["rounds"].append(round_info)

        # 로그 출력
        avg_client_acc = np.mean([r["accuracy"] for r in client_results])
        global_acc_str = ""
        if "global_eval" in round_info:
            global_acc_str = f" | Global Acc: {round_info['global_eval']['accuracy']:.4f}"
        print(
            f"Round {round_idx + 1}/{num_rounds} | "
            f"Avg Client Acc: {avg_client_acc:.4f}{global_acc_str}"
        )

    return {"server": server, "clients": clients, "history": history}
