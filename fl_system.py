"""
联邦学习总控：读取 YAML 配置、设随机种子、准备数据并执行主训练循环。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import ExperimentConfig
from dataset import prepare_data
from models import create_model
from sever.adaptive_weighting_advanced import ServerLearnableAggregator, adaptive_aggregate
from src.AdaptiveWeighted.client_aw import ClientAdaptiveWeighted
from utils.seeding import set_seed


class FederatedLearningSystem:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self) -> None:
        cfg = self.config
        set_seed(cfg.seed)
        device = cfg.resolve_device()
        print(f"Status: device={device}, seed={cfg.seed}")
        print(
            f"Status: dataset={cfg.dataset}, partition={cfg.partition}, "
            f"clients={cfg.num_clients}, rounds={cfg.global_rounds}, "
            f"proxy_ratio={cfg.proxy_ratio:.4f}, dirichlet_beta={cfg.dirichlet_beta}, "
            f"shrink_sum={cfg.shrink_sum}, proxy_batch_size={cfg.proxy_batch_size}, "
            f"eval_batch_size={cfg.eval_batch_size}, "
            f"proxy_disjoint={cfg.proxy_disjoint_from_clients}"
        )

        print("Status: Preparing data and partitioning...")
        train_dataset, test_dataset, proxy_data, client_id_map = prepare_data(
            n_clients=cfg.num_clients,
            beta=cfg.dirichlet_beta,
            proxy_ratio=cfg.proxy_ratio,
            data_root=cfg.data_root,
            seed=cfg.seed,
            dataset=cfg.dataset,
            partition=cfg.partition,
            proxy_disjoint_from_clients=cfg.proxy_disjoint_from_clients,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=cfg.eval_batch_size, shuffle=False
        )
        global_model = create_model(cfg.dataset).to(device)

        learnable_server = ServerLearnableAggregator(
            num_clients=cfg.num_clients,
            device=device,
            aggregation_weight_lr=cfg.aggregation_weight_lr,
            lam=cfg.lam,
            momentum=cfg.momentum,
            proxy_batch_size=cfg.proxy_batch_size,
            shrink_sum=cfg.shrink_sum,
        )

        acc_round_history: list[float] = []
        loss_round_history: list[float] = []
        proxy_loss_round_history: list[float] = []
        beta_round_history: list[np.ndarray] = []

        for r in range(cfg.global_rounds):
            print(f"\n--- Communication Round {r + 1} ---")
            client_models = []

            for i in range(cfg.num_clients):
                local_indices = client_id_map[i]
                local_loader = DataLoader(
                    Subset(train_dataset, local_indices),
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    drop_last=True,
                )
                client = ClientAdaptiveWeighted(
                    client_id=i,
                    device=device,
                    batch_size=cfg.batch_size,
                    local_epochs=cfg.local_epochs,
                    learning_rate=cfg.learning_rate,
                    momentum=cfg.momentum,
                )
                local_model = client.run_local_training(global_model, local_loader)
                client_models.append(local_model)
                print(f"Client {i:02d}: Training Complete", end=" | ")
                if (i + 1) % 5 == 0:
                    print("")

            print("\nServer: Performing Adaptive Weighting Aggregation...")
            global_model, _current_weights, proxy_meta_loss = adaptive_aggregate(
                global_model,
                client_models,
                proxy_data,
                device,
                lam=cfg.lam,
                learnable_server=learnable_server,
            )
            beta_round_history.append(_current_weights.detach().cpu().numpy())
            proxy_loss_round_history.append(float(proxy_meta_loss))

            global_model.eval()
            ce_fn = nn.CrossEntropyLoss(reduction="sum")
            loss_total = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = global_model(images)
                    loss_total += ce_fn(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            mean_test_ce = loss_total / max(total, 1)
            acc_round_history.append(accuracy)
            loss_round_history.append(mean_test_ce)
            print(
                f"Round {r + 1} Result -> Global Test Accuracy: {accuracy:.2f}% | "
                f"Test loss (mean CE): {mean_test_ce:.4f} | "
                f"Proxy meta-loss: {proxy_meta_loss:.4f}"
            )

        print("\nSuccess: Federated training completed.")

        out_path = Path(cfg.model_path)
        if not out_path.is_absolute():
            out_path = _PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(global_model.state_dict(), str(out_path))
        print(f"\nSuccess: 模型权重已保存至 {out_path}")

        log_npz = out_path.with_name(out_path.stem + "_run_log.npz")
        np.savez_compressed(
            str(log_npz),
            accuracy=np.asarray(acc_round_history, dtype=np.float32),
            test_loss=np.asarray(loss_round_history, dtype=np.float32),
            proxy_loss=np.asarray(proxy_loss_round_history, dtype=np.float32),
            client_betas=np.stack(beta_round_history, axis=0).astype(np.float32),
        )
        print(
            f"Success: 每轮测试精度、测试集平均交叉熵、代理集 meta-loss 与客户端聚合系数已保存至 {log_npz}"
        )
