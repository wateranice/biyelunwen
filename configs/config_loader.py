from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Union

import yaml


DeviceSpec = Literal["auto", "cuda", "cpu"]


@dataclass
class ExperimentConfig:
    seed: int
    data_root: str
    dataset: str
    eval_batch_size: int
    device: str
    global_rounds: int
    lam: float
    aggregation_weight_lr: float
    shrink_sum: float
    proxy_batch_size: int
    num_clients: int
    batch_size: int
    local_epochs: int
    learning_rate: float
    momentum: float
    dirichlet_beta: float
    proxy_ratio: float
    partition: str
    model_path: str
    # 代理索引与各客户端训练索引是否集合互斥（见 dataset.prepare_data）
    proxy_disjoint_from_clients: bool = False
    # 对比 FedProx 时客户端近端项系数（主训练 fl_system 未使用，仅实验脚本读取）
    fedprox_mu: float = 0.01

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> "ExperimentConfig":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        exp = raw.get("experiment") or {}
        srv = raw.get("server") or {}
        cli = raw.get("client") or {}
        out = raw.get("output") or {}

        proxy_ratio = float(cli.get("proxy_ratio", 0.04))
        if not 0.0 < proxy_ratio < 1.0:
            raise ValueError("client.proxy_ratio 必须在 (0, 1) 内，例如 0.01～0.05")

        shrink_sum = float(srv.get("shrink_sum", 1.0))
        if not 0.0 < shrink_sum <= 1.0:
            raise ValueError("server.shrink_sum 必须在 (0, 1]，例如 1.0 或 0.9（任务书「客户端系数和<1」）")

        batch_size = int(cli.get("batch_size", 128))
        if batch_size < 1:
            raise ValueError("client.batch_size 必须 >= 1")

        proxy_batch_size = int(srv.get("proxy_batch_size", 64))
        if proxy_batch_size < 1:
            raise ValueError("server.proxy_batch_size 必须 >= 1")

        eval_batch_size = int(exp.get("eval_batch_size", batch_size))
        if eval_batch_size < 1:
            raise ValueError("experiment.eval_batch_size 必须 >= 1")

        return cls(
            seed=int(exp.get("seed", 42)),
            data_root=str(exp.get("data_root", "./data")),
            dataset=str(exp.get("dataset", "cifar10")).lower(),
            eval_batch_size=eval_batch_size,
            device=str(srv.get("device", "auto")).lower(),
            global_rounds=int(srv.get("global_rounds", 30)),
            lam=float(srv.get("lam", 0.1)),
            aggregation_weight_lr=float(srv.get("aggregation_weight_lr", 0.01)),
            shrink_sum=shrink_sum,
            proxy_batch_size=proxy_batch_size,
            num_clients=int(cli.get("num_clients", 10)),
            batch_size=batch_size,
            local_epochs=int(cli.get("local_epochs", 2)),
            learning_rate=float(cli.get("learning_rate", 0.01)),
            momentum=float(cli.get("momentum", 0.9)),
            dirichlet_beta=float(cli.get("dirichlet_beta", 0.5)),
            proxy_ratio=proxy_ratio,
            partition=str(cli.get("partition", "dirichlet")).lower(),
            model_path=str(out.get("model_path", "global_model.pth")),
            proxy_disjoint_from_clients=bool(cli.get("proxy_disjoint_from_clients", False)),
            fedprox_mu=float(cli.get("fedprox_mu", 0.01)),
        )

    def resolve_device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            return torch.device("cuda")
        return torch.device("cpu")
