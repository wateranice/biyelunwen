import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import ExperimentConfig
from fl_system import FederatedLearningSystem


def main() -> None:
    rel = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs") / "default.yaml"
    cfg_path = rel if rel.is_absolute() else _PROJECT_ROOT / rel
    if not cfg_path.is_file():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")
    config = ExperimentConfig.load(cfg_path)
    FederatedLearningSystem(config).run()


if __name__ == "__main__":
    main()
