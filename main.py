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


if __name__ == "__main__":
    main()
