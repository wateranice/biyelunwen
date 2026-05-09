"""
本课题服务器端：可学习聚合权重 + 代理集梯度（对应 FedAPA-main/src/FedAPA/server_FedAPA.py 的角色）。
"""
from __future__ import annotations

from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call  # type: ignore

from ..server_base import ServerBaseFL


class ServerLearnableAggregator(ServerBaseFL):
    """
    维护聚合权重 logits，在代理集上对分类损失关于 logits 反传并 SGD 一步，
    再按 lam 做向均匀分布的收缩后写回全局模型。

    **权重收缩（任务书「和 < 1」）**：``shrink_sum ∈ (0,1]`` 时，各客户端系数之和为 ``shrink_sum``，
    剩余 ``1 - shrink_sum`` 混合**本轮聚合前的全局参数**，使整体更新带「向全局锚点收缩」的正则意味。
    ``shrink_sum=1`` 时退化为仅客户端凸组合（与原先行为一致）。
    """

    def __init__(
        self,
        num_clients: int,
        device: torch.device,
        aggregation_weight_lr: float,
        lam: float = 0.1,
        momentum: float = 0.9,
        proxy_batch_size: int = 64,
        shrink_sum: float = 1.0,
    ):
        super().__init__(num_clients=num_clients, device=device)
        self.lam = lam
        self.proxy_batch_size = proxy_batch_size
        if not 0.0 < shrink_sum <= 1.0:
            raise ValueError("shrink_sum 必须在 (0, 1] 内，例如 1.0（不收缩）或 0.85（和<1）")
        self.shrink_sum = float(shrink_sum)

        self.weight_logits = nn.Parameter(torch.zeros(num_clients, device=device))
        self.optimizer = optim.SGD(
            [self.weight_logits],
            lr=aggregation_weight_lr,
            momentum=momentum,
        )

    def _softmax_weights(self) -> torch.Tensor:
        return F.softmax(self.weight_logits, dim=0)

    def _build_mixed_param_dict(
        self,
        global_model: nn.Module,
        client_models: List[nn.Module],
        w_unit: torch.Tensor,
    ) -> dict:
        """
        w_unit: softmax(logits)，和为 1；客户端上实际系数 beta = shrink_sum * w_unit（和为 shrink_sum），
        其余 (1-shrink_sum) 保留给当前全局参数。
        """
        beta = self.shrink_sum * w_unit
        residual = 1.0 - self.shrink_sum
        global_sd = global_model.state_dict()
        param_dict = {}
        for name, _ in global_model.named_parameters():
            stacked = torch.stack(
                [m.state_dict()[name].detach().float() for m in client_models],
                dim=0,
            )
            view = (-1,) + (1,) * (stacked.dim() - 1)
            client_part = (beta.view(view) * stacked).sum(0)
            g = global_sd[name].detach().float()
            param_dict[name] = client_part + residual * g
        return param_dict

    def aggregate_round(
        self,
        global_model: nn.Module,
        client_models: List[nn.Module],
        proxy_data: Any,
    ) -> Tuple[nn.Module, torch.Tensor]:
        K = len(client_models)
        if K != self.num_clients:
            raise ValueError(f"客户端数 {K} 与服务器初始化 {self.num_clients} 不一致")

        global_model.eval()
        for p in global_model.parameters():
            p.requires_grad_(False)

        proxy_loader = DataLoader(
            proxy_data,
            batch_size=self.proxy_batch_size,
            shuffle=False,
        )
        criterion = nn.CrossEntropyLoss()

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=self.device)
        n_batches = 0

        for images, labels in proxy_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            w_unit = self._softmax_weights()
            param_dict = self._build_mixed_param_dict(global_model, client_models, w_unit)
            logits = functional_call(
                global_model,
                param_dict,
                (images,),
                strict=False,
            )
            total_loss = total_loss + criterion(logits, labels)
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("代理数据集为空，无法更新聚合权重")

        loss = total_loss / n_batches
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            w = self._softmax_weights()
            w_final = (1.0 - self.lam) * w + self.lam / K
            w_final = w_final / w_final.sum()
            beta = self.shrink_sum * w_final
            residual = 1.0 - self.shrink_sum
            global_sd = global_model.state_dict()

            new_sd = dict(global_sd)
            for name, _ in global_model.named_parameters():
                stacked = torch.stack(
                    [m.state_dict()[name].float() for m in client_models],
                    dim=0,
                )
                view = (-1,) + (1,) * (stacked.dim() - 1)
                client_part = (beta.view(view) * stacked).sum(0)
                new_sd[name] = client_part + residual * global_sd[name].float()

            global_model.load_state_dict(new_sd, strict=True)

        # 下一轮客户端会 deepcopy 全局模型；必须恢复 requires_grad，否则本地 loss.backward 报错
        for p in global_model.parameters():
            p.requires_grad_(True)

        # 返回各客户端实际分到的质量 beta（和为 shrink_sum），便于日志与论文作图
        return global_model, beta.detach()
