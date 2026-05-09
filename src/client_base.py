"""
客户端基类（对齐 FedAPA-main/src/client_base.py 的职责划分）。
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    import torch


class ClientBaseFL:
    """持有本地训练超参；从当前全局模型复制后在本地 DataLoader 上训练。"""

    def __init__(
        self,
        client_id: int,
        device: "torch.device",
        batch_size: int,
        local_epochs: int,
        learning_rate: float,
        momentum: float,
    ):
        self.client_id = client_id
        self.device = device
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.loss_fn = nn.CrossEntropyLoss()

    def run_local_training(self, global_model: nn.Module, train_loader: DataLoader) -> nn.Module:
        local_model = copy.deepcopy(global_model)
        local_model.train()
        optimizer = optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
        )

        for _ in range(self.local_epochs):
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                loss = self.loss_fn(local_model(images), labels)
                loss.backward()
                optimizer.step()

        return local_model
