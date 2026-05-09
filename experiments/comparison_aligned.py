"""
与主程序 ``fl_system.py`` 对齐的对比实验：同一 ``prepare_data``、同一随机种子、
同一初始化权重下依次运行 FedAvg、FedProx（客户端近端项 + 服务端平均聚合）、
本文方法（``ServerLearnableAggregator`` + ``adaptive_aggregate``）。

用法（在项目根目录）::
    python experiments/comparison_aligned.py configs/default.yaml
    python experiments/comparison_aligned.py configs/default.yaml --methods fedavg adaptive
    python experiments/comparison_aligned.py configs/default.yaml --betas 0.1 0.5 1.0

第 4 章数据产物（每个 β）::
    - ``experiments/outputs/comparison_acc_beta{β}_....png``  三曲线图
    - ``experiments/outputs/comparison_acc_beta{β}_....npz`` 各方法逐轮 Acc
    - ``experiments/outputs/ch4_comparison_summary.csv``     全 β 汇总：末轮 Acc、达阈值轮次
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import ExperimentConfig
from dataset import prepare_data
from models import create_model
from sever.adaptive_weighting_advanced import ServerLearnableAggregator, adaptive_aggregate
from sever.fed_avg import fed_avg_aggregate
from src.AdaptiveWeighted.client_aw import ClientAdaptiveWeighted
from utils.seeding import set_seed


MethodName = Literal["fedavg", "fedprox", "adaptive"]


def _first_round_reaching(acc_hist: List[float], threshold_pct: float) -> Optional[int]:
    """返回首次 test acc >= threshold 的通信轮次（1-based）；达不到则 None。"""
    for i, a in enumerate(acc_hist):
        if a >= threshold_pct:
            return i + 1
    return None


def _evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, pred = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100.0 * correct / total


def _local_train_fedavg_like(
    global_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    local_epochs: int,
    lr: float,
    momentum: float,
) -> nn.Module:
    client = ClientAdaptiveWeighted(
        client_id=0,
        device=device,
        batch_size=train_loader.batch_size,
        local_epochs=local_epochs,
        learning_rate=lr,
        momentum=momentum,
    )
    return client.run_local_training(global_model, train_loader)


def _local_train_fedprox(
    global_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    local_epochs: int,
    lr: float,
    momentum: float,
    mu: float,
) -> nn.Module:
    local_model = copy.deepcopy(global_model)
    local_model.train()
    optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()
    global_params = {n: p.detach() for n, p in global_model.named_parameters()}

    for _ in range(local_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = local_model(images)
            loss = loss_fn(logits, labels)
            if mu > 0.0:
                prox = torch.zeros((), device=device)
                for name, lp in local_model.named_parameters():
                    gp = global_params[name]
                    prox = prox + (lp - gp).pow(2).sum()
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()

    return local_model


def run_method(
    cfg: ExperimentConfig,
    method: MethodName,
    train_ds,
    test_loader: DataLoader,
    proxy_ds,
    client_id_map: List[List[int]],
    device: torch.device,
    init_state: Dict[str, torch.Tensor],
) -> List[float]:
    set_seed(cfg.seed)
    global_model = create_model(cfg.dataset).to(device)
    global_model.load_state_dict(copy.deepcopy(init_state))

    learnable: ServerLearnableAggregator | None = None
    if method == "adaptive":
        learnable = ServerLearnableAggregator(
            num_clients=cfg.num_clients,
            device=device,
            aggregation_weight_lr=cfg.aggregation_weight_lr,
            lam=cfg.lam,
            momentum=cfg.momentum,
            proxy_batch_size=cfg.proxy_batch_size,
            shrink_sum=cfg.shrink_sum,
        )

    acc_hist: List[float] = []
    for r in range(cfg.global_rounds):
        client_models: List[nn.Module] = []
        for i in range(cfg.num_clients):
            local_loader = DataLoader(
                Subset(train_ds, client_id_map[i]),
                batch_size=cfg.batch_size,
                shuffle=True,
            )
            if method == "fedprox":
                local_m = _local_train_fedprox(
                    global_model,
                    local_loader,
                    device,
                    cfg.local_epochs,
                    cfg.learning_rate,
                    cfg.momentum,
                    cfg.fedprox_mu,
                )
            else:
                local_m = _local_train_fedavg_like(
                    global_model,
                    local_loader,
                    device,
                    cfg.local_epochs,
                    cfg.learning_rate,
                    cfg.momentum,
                )
            client_models.append(local_m)

        if method == "adaptive":
            global_model, _ = adaptive_aggregate(
                global_model,
                client_models,
                proxy_ds,
                device,
                lam=cfg.lam,
                learnable_server=learnable,
            )
        else:
            global_model = fed_avg_aggregate(global_model, client_models)

        acc_hist.append(_evaluate(global_model, test_loader, device))
        print(f"  [{method}] round {r + 1}/{cfg.global_rounds}  test_acc={acc_hist[-1]:.2f}%")

    return acc_hist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_PROJECT_ROOT / "configs" / "default.yaml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fedavg", "fedprox", "adaptive"],
        choices=["fedavg", "fedprox", "adaptive"],
        help="要运行的方法子集",
    )
    parser.add_argument(
        "--betas",
        nargs="*",
        type=float,
        default=None,
        help="若指定，则对每个 dirichlet_beta 各跑一轮完整对比并各存一图",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="50,55,60",
        help="Top-1 准确率阈值（%%），逗号分隔，用于「达到阈值的首个轮次」列",
    )
    parser.add_argument(
        "--rounds-override",
        type=int,
        default=None,
        help="覆盖 YAML 中的 global_rounds（快速试跑）",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _PROJECT_ROOT / cfg_path

    betas = args.betas if args.betas else [None]
    thr_list = [
        float(x.strip())
        for x in args.thresholds.split(",")
        if x.strip()
    ]

    summary_rows: List[Dict[str, object]] = []

    for beta in betas:
        cfg = ExperimentConfig.load(cfg_path)
        if args.rounds_override is not None:
            cfg = replace(cfg, global_rounds=int(args.rounds_override))
        if beta is not None:
            cfg = replace(cfg, dirichlet_beta=float(beta))

        set_seed(cfg.seed)
        device = cfg.resolve_device()
        train_ds, test_ds, proxy_ds, client_id_map = prepare_data(
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
            test_ds, batch_size=cfg.eval_batch_size, shuffle=False
        )

        ref = create_model(cfg.dataset).to(device)
        init_state = copy.deepcopy(ref.state_dict())
        del ref

        series: Dict[str, List[float]] = {}
        for m in args.methods:
            print(f"\n=== Running {m} (beta={cfg.dirichlet_beta}) ===")
            series[m] = run_method(
                cfg,
                m,
                train_ds,
                test_loader,
                proxy_ds,
                client_id_map,
                device,
                init_state,
            )

        out_dir = _PROJECT_ROOT / "experiments" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        beta_tag = str(cfg.dirichlet_beta).replace(".", "p")
        fig_path = out_dir / f"comparison_acc_beta{beta_tag}_{cfg.dataset}_{cfg.partition}.png"

        rounds = range(1, len(next(iter(series.values()))) + 1)
        plt.figure(figsize=(9, 5))
        styles = {
            "fedavg": ("g--", "FedAvg"),
            "fedprox": ("r-.", "FedProx"),
            "adaptive": ("b-", "Adaptive (learnable)"),
        }
        for m, acc in series.items():
            sty, lbl = styles[m]
            plt.plot(list(rounds), acc, sty, label=lbl, linewidth=2 if m == "adaptive" else 1.5)
        plt.xlabel("Communication round")
        plt.ylabel("Test accuracy (%)")
        plt.title(
            f"Aligned comparison | {cfg.dataset} | {cfg.partition} | beta={cfg.dirichlet_beta}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"\nSaved figure: {fig_path}")

        npz_path = fig_path.with_suffix(".npz")
        save_dict: Dict[str, object] = {
            "dirichlet_beta": np.float32(cfg.dirichlet_beta),
            "rounds": np.arange(1, len(next(iter(series.values()))) + 1, dtype=np.int32),
        }
        for m_name, acc in series.items():
            save_dict[f"acc_{m_name}"] = np.asarray(acc, dtype=np.float32)
        np.savez_compressed(str(npz_path), **save_dict)
        print(f"Saved curves: {npz_path}")

        for m_name, acc in series.items():
            row: Dict[str, object] = {
                "dataset": cfg.dataset,
                "dirichlet_beta": cfg.dirichlet_beta,
                "partition": cfg.partition,
                "method": m_name,
                "final_test_acc_pct": round(acc[-1], 4),
                "global_rounds": cfg.global_rounds,
            }
            for thr in thr_list:
                col = f"first_round_acc_ge_{thr:g}"
                rch = _first_round_reaching(acc, thr)
                row[col] = rch if rch is not None else ""
            summary_rows.append(row)

    out_dir = _PROJECT_ROOT / "experiments" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "ch4_comparison_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSaved summary table: {summary_csv}")


if __name__ == "__main__":
    main()
