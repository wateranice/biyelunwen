import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定 random / numpy / torch（含 CUDA）随机种子，便于实验复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
