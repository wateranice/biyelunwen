"""
从 ``fl_system`` 保存的 ``*_run_log.npz`` 绘制：

- 测试精度、测试 Loss、代理 meta-loss（若字段存在）
- 各客户端聚合系数 **热力图**
- 各客户端系数随通信轮次的 **折线轨迹图**（演化轨迹）

用法::
    python utils/plot_run_log.py weights/ch4/e4_weight_vis_cifar10_dirichlet_run_log.npz --out experiments/outputs/e4_weight_vis

另会输出各轮 ``sum_k beta_k`` 折线图（``*_beta_sum_per_round.png``），便于与 ``shrink_sum`` 对照。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def plot_npz(npz_path: Path, out_dir: Path | None = None) -> None:
    data = np.load(npz_path)
    acc = data["accuracy"]
    betas = data["client_betas"]
    rounds = np.arange(1, len(acc) + 1)
    out_dir = out_dir or npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem.replace("_run_log", "")

    plt.figure(figsize=(8, 4))
    plt.plot(rounds, acc, "b-", linewidth=2)
    plt.xlabel("Communication round")
    plt.ylabel("Test accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.title(f"Test accuracy — {stem}")
    plt.tight_layout()
    p1 = out_dir / f"{stem}_acc_from_log.png"
    plt.savefig(p1, dpi=200)
    plt.close()

    if "test_loss" in data.files:
        tl = data["test_loss"]
        plt.figure(figsize=(8, 4))
        plt.plot(rounds, tl, "r-", linewidth=2)
        plt.xlabel("Communication round")
        plt.ylabel("Test loss (mean cross-entropy, nats)")
        plt.grid(True, alpha=0.3)
        plt.title(f"Test loss vs round — {stem}")
        plt.tight_layout()
        p1b = out_dir / f"{stem}_test_loss_from_log.png"
        plt.savefig(p1b, dpi=200)
        plt.close()

    if "proxy_loss" in data.files:
        pl = data["proxy_loss"]
        plt.figure(figsize=(8, 4))
        plt.plot(rounds, pl, "m-", linewidth=2)
        plt.xlabel("Communication round")
        plt.ylabel("Proxy meta-loss (nats)")
        plt.grid(True, alpha=0.3)
        plt.title(f"Proxy set meta-loss vs round — {stem}")
        plt.tight_layout()
        p1c = out_dir / f"{stem}_proxy_loss_from_log.png"
        plt.savefig(p1c, dpi=200)
        plt.close()

    # 各轮客户端系数之和（等于 shrink_sum 时常为水平线，便于论文说明收缩设定）
    if betas.ndim == 2 and betas.shape[0] == len(acc):
        beta_sum = np.asarray(betas.sum(axis=1), dtype=np.float64)
        plt.figure(figsize=(8, 3))
        plt.plot(rounds, beta_sum, "k-", linewidth=1.8)
        plt.xlabel("Communication round")
        plt.ylabel(r"$\sum_k \beta_k$ (sum of client coeffs.)")
        plt.grid(True, alpha=0.3)
        plt.title(f"Sum of aggregation client coefficients — {stem}")
        plt.tight_layout()
        p_sum = out_dir / f"{stem}_beta_sum_per_round.png"
        plt.savefig(p_sum, dpi=200)
        plt.close()

    plt.figure(figsize=(10, 4))
    plt.imshow(betas.T, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(label=r"$\beta_k$ (client coeff.)")
    plt.xlabel("Round")
    plt.ylabel("Client index")
    plt.title(f"Aggregation client coefficients — {stem}")
    plt.tight_layout()
    p2 = out_dir / f"{stem}_betas_heatmap.png"
    plt.savefig(p2, dpi=200)
    plt.close()

    # 聚合权重演化轨迹：每客户端一条曲线（betas: shape [n_rounds, n_clients]）
    n_r, n_c = betas.shape[0], betas.shape[1]
    plt.figure(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for k in range(n_c):
        plt.plot(
            rounds[:n_r],
            betas[:, k],
            color=cmap(k % 10),
            linewidth=1.5,
            alpha=0.85,
            label=f"Client {k}",
        )
    plt.xlabel("Communication round")
    plt.ylabel(r"Client coefficient $\beta_k$ (after softmax $\times$ shrink)")
    plt.grid(True, alpha=0.3)
    plt.title(f"Aggregation weight trajectory — {stem}")
    ncol = 5 if n_c <= 10 else 6
    plt.legend(ncol=ncol, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    p3 = out_dir / f"{stem}_betas_trajectory.png"
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()

    saved: list[str] = [str(p1)]
    if "test_loss" in data.files:
        saved.append(str(out_dir / f"{stem}_test_loss_from_log.png"))
    if "proxy_loss" in data.files:
        saved.append(str(out_dir / f"{stem}_proxy_loss_from_log.png"))
    saved.extend([str(p2), str(p3)])
    if betas.ndim == 2 and betas.shape[0] == len(acc):
        saved.append(str(out_dir / f"{stem}_beta_sum_per_round.png"))
    print("Saved:\n  " + "\n  ".join(saved))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    plot_npz(args.npz.resolve(), args.out)


if __name__ == "__main__":
    main()
