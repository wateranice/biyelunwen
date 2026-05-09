"""
绘制联邦数据划分可视化（与论文章节「不同 Non-IID 划分」插图一致）：

1) **狄利克雷（Dirichlet）**：多组 ``dirichlet_beta``，堆叠条形图展示各客户端各类样本数。
2) **病态（Pathological）**：类互斥的「分片」划分；堆叠条形图。

- **狄利克雷** 与 ``dataset/data_utils._split_dirichlet`` 一致。
- **病态**：``n_clients <= n_classes`` 时与 ``_split_pathological`` 一致；``n_clients > n_classes`` 且整除时
  使用按类分片扩展（仅用于本图）。

用法::

    python experiments/plot_partition_distribution.py --dataset fmnist
    python experiments/plot_partition_distribution.py --dataset cifar10 --data-root ./data

产物：``experiments/outputs/partition_dirichlet_{dataset}.png``、
``partition_pathological_{dataset}.png``。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10, FashionMNIST

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.data_utils import _split_dirichlet, _split_pathological

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

FMNIST_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

COLORS = [
    "#4c4c4c",
    "#7b68ee",
    "#2e8b57",
    "#ff8c00",
    "#4682b4",
    "#8b4513",
    "#6a5acd",
    "#20b2aa",
    "#dc143c",
    "#5f9ea0",
]


def _load_train_targets(data_root: str, dataset: str) -> tuple[np.ndarray, int, str]:
    name = (dataset or "cifar10").lower()
    if name == "cifar10":
        ds = CIFAR10(root=data_root, train=True, download=True)
        tag = "cifar10"
    elif name in ("fmnist", "fashion_mnist", "fashion-mnist"):
        ds = FashionMNIST(root=data_root, train=True, download=True)
        tag = "fmnist"
    else:
        raise ValueError(f"未知 dataset={dataset!r}，支持: cifar10, fmnist")
    y = np.asarray(list(ds.targets), dtype=np.int64)
    return y, 10, tag


def _class_names_for_dataset(tag: str) -> tuple[str, ...]:
    if tag == "cifar10":
        return CIFAR10_CLASSES
    return FMNIST_CLASSES


def _dataset_label(tag: str) -> str:
    return "CIFAR-10" if tag == "cifar10" else "Fashion-MNIST"


def _label_indices_from_y(y_train: np.ndarray, n_classes: int) -> list[np.ndarray]:
    return [np.where(y_train == c)[0] for c in range(n_classes)]


def _client_class_counts(y_train: np.ndarray, client_id_map: list[list[int]], n_classes: int) -> np.ndarray:
    k = len(client_id_map)
    mat = np.zeros((k, n_classes), dtype=np.int64)
    for i, idxs in enumerate(client_id_map):
        if not idxs:
            continue
        labs = y_train[np.asarray(idxs, dtype=np.int64)]
        mat[i] = np.bincount(labs, minlength=n_classes)[:n_classes]
    return mat


def _split_pathological_sharded_for_plot(
    label_indices: list[np.ndarray],
    n_clients: int,
    n_classes: int,
    seed: int | None,
) -> list[list[int]]:
    if n_clients % n_classes != 0:
        raise ValueError(
            f"pathological 多客户端绘图需要 n_clients 能被 n_classes 整除，"
            f"当前 n_clients={n_clients}, n_classes={n_classes}"
        )
    rpc = n_clients // n_classes
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    cmap: list[list[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx = label_indices[c].copy()
        rng.shuffle(idx)
        parts = np.array_split(idx, rpc)
        for j, part in enumerate(parts):
            cid = c * rpc + j
            cmap[cid].extend(np.asarray(part).ravel().tolist())
    return cmap


def _sort_clients_for_display(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dominant = np.argmax(mat, axis=1)
    max_per_row = mat[np.arange(mat.shape[0]), dominant]
    order = np.lexsort((-max_per_row, dominant))
    return mat[order], order


def _plot_stacked_panel(
    ax,
    mat: np.ndarray,
    class_names: tuple[str, ...],
    title: str,
    xlabel: str = "Number of samples",
) -> None:
    k, n_c = mat.shape
    y = np.arange(k)
    left = np.zeros(k, dtype=np.float64)
    for c in range(n_c):
        w = mat[:, c].astype(np.float64)
        ax.barh(y, w, left=left, height=0.9, label=class_names[c], color=COLORS[c % len(COLORS)])
        left += w
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Client (sorted)")
    ax.invert_yaxis()


def plot_dirichlet_figure(
    y_train: np.ndarray,
    n_classes: int,
    n_clients: int,
    betas: list[float],
    seed: int,
    class_names: tuple[str, ...],
    ds_label: str,
    out_path: Path,
) -> None:
    n_panels = len(betas)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 8.0), squeeze=False)
    axes_flat = axes.ravel()

    for ax, beta in zip(axes_flat, betas):
        np.random.seed(int(seed) + int(1e6 * float(beta)) % 10_000_000)
        label_indices = _label_indices_from_y(y_train, n_classes)
        cmap = _split_dirichlet(label_indices, n_clients, float(beta), n_classes)
        mat = _client_class_counts(y_train, cmap, n_classes)
        mat_s, _ = _sort_clients_for_display(mat)
        _plot_stacked_panel(
            ax,
            mat_s,
            class_names,
            title=rf"$\beta = {beta:g}$",
        )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(10, n_classes),
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8,
    )
    fig.suptitle(f"{ds_label} — Dirichlet partition", y=1.08, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pathological_figure(
    y_train: np.ndarray,
    n_classes: int,
    n_clients: int,
    seed: int,
    class_names: tuple[str, ...],
    ds_label: str,
    out_path: Path,
) -> None:
    label_indices = _label_indices_from_y(y_train, n_classes)
    if n_clients > n_classes:
        if n_clients % n_classes != 0:
            raise ValueError(
                f"病态划分 n_clients={n_clients} 需能被 n_classes={n_classes} 整除。"
            )
        cmap = _split_pathological_sharded_for_plot(label_indices, n_clients, n_classes, seed)
        sub = rf"class-sharded | {n_clients} clients ({n_clients // n_classes} per class)"
    else:
        cmap = _split_pathological(label_indices, n_clients, n_classes, seed)
        sub = "array_split over permuted classes"

    mat = _client_class_counts(y_train, cmap, n_classes)
    mat_s, _ = _sort_clients_for_display(mat)

    h = max(8.0, min(22.0, 0.11 * n_clients + 2.0))
    fig, ax = plt.subplots(figsize=(7.0, h))
    _plot_stacked_panel(
        ax,
        mat_s,
        class_names,
        title=rf"Pathological — seed = {seed}, clients = {n_clients} | {sub}",
    )
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.suptitle(f"{ds_label} — Pathological (class-disjoint shards)", y=1.01, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Dirichlet / pathological partition stacks.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        type=str,
        default="fmnist",
        choices=["cifar10", "fmnist", "fashion_mnist", "fashion-mnist"],
        help="训练集划分可视化所用数据集（默认 Fashion-MNIST）",
    )
    parser.add_argument("--n-clients-dirichlet", type=int, default=100)
    parser.add_argument(
        "--n-clients-pathological",
        type=int,
        default=6,
        help="病态图客户端数；K≤C 时用 _split_pathological；K>C 且整除时用按类分片",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.05, 0.5, 5.0],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJECT_ROOT / "experiments" / "outputs",
    )
    args = parser.parse_args()

    y_train, n_classes, tag = _load_train_targets(args.data_root, args.dataset)
    class_names = _class_names_for_dataset(tag)
    ds_label = _dataset_label(tag)

    out_d = args.out_dir.resolve()
    plot_dirichlet_figure(
        y_train,
        n_classes,
        args.n_clients_dirichlet,
        list(args.betas),
        args.seed,
        class_names,
        ds_label,
        out_d / f"partition_dirichlet_{tag}.png",
    )
    plot_pathological_figure(
        y_train,
        n_classes,
        args.n_clients_pathological,
        args.seed,
        class_names,
        ds_label,
        out_d / f"partition_pathological_{tag}.png",
    )


if __name__ == "__main__":
    main()
