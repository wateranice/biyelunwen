"""按数据集名创建网络，供 YAML 切换数据时无需改训练脚本。"""
from __future__ import annotations

import torch.nn as nn

from models.models import SimpleCNN


def create_model(dataset: str, num_classes: int = 10) -> nn.Module:
    name = (dataset or "cifar10").lower()
    if name == "cifar10":
        return SimpleCNN(in_channels=3, num_classes=num_classes, input_size=32)
    if name == "mnist":
        return SimpleCNN(in_channels=1, num_classes=num_classes, input_size=28)
    if name in ("fmnist", "fashion_mnist", "fashion-mnist"):
        return SimpleCNN(in_channels=1, num_classes=num_classes, input_size=28)
    raise ValueError(
        f"未知 dataset={dataset!r}，请在 configs 中使用 cifar10 | mnist | fmnist"
    )
