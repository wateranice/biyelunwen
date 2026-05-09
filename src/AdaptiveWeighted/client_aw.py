"""
本课题客户端实现（对应 FedAPA-main/src/FedAPA/client_FedAPA.py 的位置）。
当前本地训练逻辑与基类一致，便于日后替换为 FedProx 等算法特化版本。
"""
from __future__ import annotations

from ..client_base import ClientBaseFL


class ClientAdaptiveWeighted(ClientBaseFL):
    """基于代理加权联邦设定下的客户端；本地训练沿用 SGD + CrossEntropy。"""

    pass
