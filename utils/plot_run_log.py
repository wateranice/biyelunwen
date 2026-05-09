"""
从 ``fl_system`` 保存的 ``*_run_log.npz`` 绘制测试精度曲线与各客户端聚合系数热力图。

用法::
    python utils/plot_run_log.py weights/adaptive/cifar10_dirichlet_run_log.npz
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
    print(f"Saved: {p1}\nSaved: {p2}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    plot_npz(args.npz.resolve(), args.out)


if __name__ == "__main__":
    main()
