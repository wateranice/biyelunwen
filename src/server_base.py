"""
服务器基类（对齐 FedAPA-main/src/server_base.py 的最小公共字段）。
后续可在此扩展选客户端、评估等接口。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class ServerBaseFL:
    def __init__(self, num_clients: int, device: "torch.device"):
        self.num_clients = num_clients
        self.device = device
