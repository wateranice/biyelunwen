"""
消融与敏感性：在统一入口 ``FederatedLearningSystem`` 下串行多组超参，
每组写入独立 ``model_path``，便于与 ``*_run_log.npz`` 对照。

用法（项目根目录）::
    python experiments/ablation_thesis.py configs/default.yaml

可在下方 ``PRESETS`` 中增删字典项；键须为 ``ExperimentConfig`` 已有字段。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import ExperimentConfig
from fl_system import FederatedLearningSystem

# (输出文件名片段, {ExperimentConfig 字段覆盖})
PRESETS: list[tuple[str, dict]] = [
    ("baseline", {}),
    ("lam0", {"lam": 0.0}),
    ("shrink_sum_1", {"shrink_sum": 1.0}),
    ("proxy_001", {"proxy_ratio": 0.01}),
    ("proxy_008", {"proxy_ratio": 0.08}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_PROJECT_ROOT / "configs" / "default.yaml"),
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _PROJECT_ROOT / cfg_path

    base = ExperimentConfig.load(cfg_path)
    base_mp = Path(base.model_path)
    if not base_mp.is_absolute():
        base_mp = _PROJECT_ROOT / base_mp

    for tag, overrides in PRESETS:
        new_path = base_mp.with_name(f"abl_{tag}{base_mp.suffix}")
        cfg = replace(base, **overrides, model_path=str(new_path))
        print(f"\n========== Ablation: {tag} overrides={overrides} ==========")
        FederatedLearningSystem(cfg).run()


if __name__ == "__main__":
    main()
