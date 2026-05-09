import argparse
import sys
from dataclasses import replace
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import ExperimentConfig
from fl_system import FederatedLearningSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="联邦学习主入口（自适应聚合）")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_PROJECT_ROOT / "configs" / "default.yaml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--rounds-override",
        type=int,
        default=None,
        help="覆盖 YAML 中的 global_rounds（例如 10 轮快速出 run_log 图）",
    )
    parser.add_argument(
        "--plot-after",
        action="store_true",
        help="训练结束后自动调用 utils/plot_run_log 生成 Acc/Loss/代理 loss 与 β 热力图、轨迹图（E4-6）",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="与 --plot-after 联用：图保存目录（默认 experiments/outputs/e4_weight_vis）",
    )
    args = parser.parse_args()

    rel = Path(args.config)
    cfg_path = rel if rel.is_absolute() else _PROJECT_ROOT / rel
    if not cfg_path.is_file():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")
    config = ExperimentConfig.load(cfg_path)
    if args.rounds_override is not None:
        config = replace(config, global_rounds=int(args.rounds_override))
        print(f"Status: global_rounds overridden to {config.global_rounds}")
    FederatedLearningSystem(config).run()

    if args.plot_after:
        from utils.plot_run_log import plot_npz

        out_path = Path(config.model_path)
        if not out_path.is_absolute():
            out_path = _PROJECT_ROOT / out_path
        log_npz = out_path.with_name(out_path.stem + "_run_log.npz")
        if not log_npz.is_file():
            raise FileNotFoundError(f"未找到日志文件，无法绘图: {log_npz}")
        plot_out = args.plot_out
        if plot_out is None:
            plot_out = _PROJECT_ROOT / "experiments" / "outputs" / "e4_weight_vis"
        else:
            plot_out = plot_out if plot_out.is_absolute() else _PROJECT_ROOT / plot_out
        print(f"\nStatus: --plot-after 将图保存到 {plot_out}")
        plot_npz(log_npz.resolve(), plot_out)


if __name__ == "__main__":
    main()
