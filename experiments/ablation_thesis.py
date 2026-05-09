"""
消融与敏感性（第 4 章 E4-7 / E4-8 / E4-9）：在 ``FederatedLearningSystem`` 下串行多组超参，
每组独立 ``model_path`` 与 ``*_run_log.npz``，并汇总到 ``experiments/outputs/`` 下 CSV。

用法（项目根目录）::

    # 默认使用 configs/ch4_compare.yaml，跑完全部预设（耗时长）
    python experiments/ablation_thesis.py

    # 指定配置（CIFAR-10）
    python experiments/ablation_thesis.py configs/ch4_compare.yaml

    # 更快试跑 / 预实验：改用 Fashion-MNIST（数据与模型更小，单轮更快）
    python experiments/ablation_thesis.py configs/ch4_fmnist.yaml

    # 只跑其中几组（逗号分隔标签）
    python experiments/ablation_thesis.py configs/ch4_compare.yaml --only baseline,proxy_001,shrink_090

    # 快速试跑 2 轮
    python experiments/ablation_thesis.py configs/ch4_compare.yaml --only baseline --rounds-override 2

    # 列出可用预设标签
    python experiments/ablation_thesis.py --list

预设键须为 ``ExperimentConfig`` 已有字段；可在 ``PRESETS`` 中增删 ``(tag, overrides)``。
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import ExperimentConfig
from fl_system import FederatedLearningSystem

# (输出文件名片段, {ExperimentConfig 字段覆盖})
# baseline：与 yaml 一致（默认 proxy_ratio=0.04, shrink_sum=1.0, lam=0.1）
PRESETS: List[Tuple[str, Dict[str, object]]] = [
    ("baseline", {}),
    # E4-7：代理数据规模（约 1%～5% 及扩展；数值为占训练集比例）
    ("proxy_001", {"proxy_ratio": 0.01}),
    ("proxy_002", {"proxy_ratio": 0.02}),
    ("proxy_003", {"proxy_ratio": 0.03}),
    ("proxy_005", {"proxy_ratio": 0.05}),
    # E4-8 / E4-9：收缩系数 shrink_sum（客户端系数和 < 1 向全局锚点混合）
    ("shrink_085", {"shrink_sum": 0.85}),
    ("shrink_090", {"shrink_sum": 0.9}),
    # 均匀混合强度 lam（0 表示关闭与均匀分布的 KL 相关项，依实现语义）
    ("lam_0", {"lam": 0.0}),
]


def _preset_tags() -> List[str]:
    return [t for t, _ in PRESETS]


def _select_presets(only: Sequence[str] | None) -> List[Tuple[str, Dict[str, object]]]:
    if not only:
        return list(PRESETS)
    want = {x.strip() for x in only if x.strip()}
    unknown = want - set(_preset_tags())
    if unknown:
        raise SystemExit(f"未知 --only 标签: {sorted(unknown)}。用 --list 查看可用标签。")
    return [(t, o) for t, o in PRESETS if t in want]


def _read_final_metrics_from_npz(npz_path: Path) -> Tuple[float, float, float]:
    z = np.load(str(npz_path))
    acc = z["accuracy"]
    tl = z["test_loss"]
    pl = z["proxy_loss"]
    return float(acc[-1]), float(tl[-1]), float(pl[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="第 4 章消融：代理比例 / 收缩系数 / lam 等")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_PROJECT_ROOT / "configs" / "ch4_compare.yaml"),
        help="基准 YAML（默认 ch4_compare.yaml）",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="只运行指定标签，逗号分隔，例如 baseline,proxy_001,shrink_090",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="打印全部预设标签与覆盖字段后退出",
    )
    parser.add_argument(
        "--rounds-override",
        type=int,
        default=None,
        help="覆盖 global_rounds（快速试跑）",
    )
    args = parser.parse_args()

    if args.list:
        print("可用预设 (--only 使用下列标签):\n")
        for tag, ov in PRESETS:
            print(f"  {tag:16s}  {ov if ov else '(yaml 默认)'}")
        return

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _PROJECT_ROOT / cfg_path

    only_list = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else []
    selected = _select_presets(only_list)

    base = ExperimentConfig.load(cfg_path)
    base_mp = Path(base.model_path)
    if not base_mp.is_absolute():
        base_mp = _PROJECT_ROOT / base_mp

    out_dir = _PROJECT_ROOT / "experiments" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / (
        f"ablation_summary_{base.dataset}_{base.partition}.csv"
    )

    summary_rows: List[Dict[str, object]] = []

    for tag, overrides in selected:
        new_path = base_mp.with_name(f"abl_{tag}{base_mp.suffix}")
        cfg = replace(base, **overrides, model_path=str(new_path))
        if args.rounds_override is not None:
            cfg = replace(cfg, global_rounds=int(args.rounds_override))

        print(f"\n========== Ablation: {tag} overrides={overrides} ==========")
        FederatedLearningSystem(cfg).run()

        log_npz = new_path.with_name(new_path.stem + "_run_log.npz")
        if not log_npz.is_file():
            print(f"Warning: 未找到日志 {log_npz}，跳过汇总该行。")
            continue
        fa, fl, fp = _read_final_metrics_from_npz(log_npz)
        summary_rows.append(
            {
                "tag": tag,
                "dataset": cfg.dataset,
                "partition": cfg.partition,
                "dirichlet_beta": cfg.dirichlet_beta,
                "global_rounds": cfg.global_rounds,
                "proxy_ratio": cfg.proxy_ratio,
                "shrink_sum": cfg.shrink_sum,
                "lam": cfg.lam,
                "aggregation_weight_lr": cfg.aggregation_weight_lr,
                "final_test_acc_pct": round(fa, 4),
                "final_test_loss_mean_ce": round(fl, 6),
                "final_proxy_meta_loss": round(fp, 6),
                "model_path": str(new_path),
                "run_log_npz": str(log_npz),
            }
        )

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSaved ablation summary: {summary_csv}")


if __name__ == "__main__":
    main()
