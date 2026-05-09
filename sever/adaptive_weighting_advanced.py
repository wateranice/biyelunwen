"""
聚合入口与启发式回退；可学习服务器实现已迁至 ``src/AdaptiveWeighted/``（对齐 FedAPA 的 src 结构）。

- ``ServerLearnableAggregator``、``ClientAdaptiveWeighted``：见 ``src.AdaptiveWeighted``
- 本文件保留 ``adaptive_aggregate`` 与无服务器时的启发式路径，供 ``experiments/`` 等旧代码导入。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.AdaptiveWeighted.client_aw import ClientAdaptiveWeighted
from src.AdaptiveWeighted.server_aw import ServerLearnableAggregator

# 供外部 ``from sever.adaptive_weighting_advanced import ServerLearnableAggregator`` 等使用
__all__ = [
    "ClientAdaptiveWeighted",
    "ServerLearnableAggregator",
    "train_client_local",
    "adaptive_aggregate",
]


def train_client_local(
    global_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    momentum: float,
    local_epochs: int,
    client_id: int = 0,
) -> nn.Module:
    """便捷函数：内部构造 ``ClientAdaptiveWeighted`` 并执行一轮本地训练。"""
    client = ClientAdaptiveWeighted(
        client_id=client_id,
        device=device,
        batch_size=train_loader.batch_size,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
    )
    return client.run_local_training(global_model, train_loader)


def _heuristic_aggregate(
    global_model: nn.Module,
    client_models: List[nn.Module],
    proxy_data,
    device: torch.device,
    lam: float = 0.1,
) -> Tuple[nn.Module, torch.Tensor]:
    """原「loss → softmax + 控制变量」启发式（learnable_server 未传入时使用）。"""
    global_model.eval()
    K = len(client_models)

    w_t = {name: param.clone().detach() for name, param in global_model.named_parameters()}
    delta_W = []
    for client_model in client_models:
        dw = {}
        for name, param in client_model.named_parameters():
            dw[name] = w_t[name] - param.detach()
        delta_W.append(dw)

    proxy_loader = DataLoader(proxy_data, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    losses = []

    with torch.no_grad():
        for client_model in client_models:
            client_model.eval()
            total_loss = 0.0
            for images, labels in proxy_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = client_model(images)
                total_loss += criterion(outputs, labels).item()
            losses.append(total_loss / len(proxy_loader))

    loss_tensor = torch.tensor(losses, device=device)
    alpha_k = F.softmax(-loss_tensor, dim=0)
    avg_w = torch.ones_like(alpha_k) / K
    final_alpha = (1 - lam) * alpha_k + lam * avg_w
    control_multiplier = K * final_alpha

    global_dict = global_model.state_dict()
    for name in global_dict.keys():
        aggregated_param = torch.zeros_like(global_dict[name]).float()
        for k in range(K):
            term = w_t[name] - delta_W[k][name] * control_multiplier[k]
            aggregated_param += term / K
        global_dict[name] = aggregated_param

    global_model.load_state_dict(global_dict)
    return global_model, final_alpha


def adaptive_aggregate(
    global_model: nn.Module,
    client_models: List[nn.Module],
    proxy_data,
    device: torch.device,
    lam: float = 0.1,
    learnable_server: Optional[ServerLearnableAggregator] = None,
) -> Tuple[nn.Module, torch.Tensor]:
    """
    聚合入口。

    - 若传入 ``learnable_server``：在代理集上对服务器 ``weight_logits`` 做梯度下降并凸组合写回全局模型。
    - 若为 ``None``：使用原启发式聚合（兼容 experiments 等旧调用）。
    """
    if learnable_server is not None:
        return learnable_server.aggregate_round(global_model, client_models, proxy_data)
    return _heuristic_aggregate(global_model, client_models, proxy_data, device, lam=lam)
