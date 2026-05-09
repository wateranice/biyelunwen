#!/usr/bin/env bash
# 论文相关实验一键顺序执行（建议在服务器数据盘 + screen/tmux 中运行）
#
#   cd /root/autodl-tmp/biyelunwen
#   source .venv/bin/activate
#   screen -S thesis
#   bash scripts/run_thesis_experiments.sh
#   # Ctrl+A D 挂起 screen
#
# 可通过环境变量跳过某几步，例如：SKIP_MAIN=1 bash scripts/run_thesis_experiments.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

echo "========== 1/3 第4章：FedAvg / FedProx / Adaptive，多 β + 阈值汇总 =========="
python experiments/comparison_aligned.py configs/ch4_compare.yaml \
  --betas 0.05 0.1 0.3 0.5 1.0 --thresholds 50,55,60

if [[ "${SKIP_MAIN:-0}" != "1" ]]; then
  echo "========== 2/3 本文方法单管线训练：保存权重 + *_run_log.npz（可选） =========="
  python main.py configs/ch4_compare.yaml
else
  echo "========== 2/3 已跳过（SKIP_MAIN=1） =========="
fi

if [[ "${SKIP_ABLATION:-0}" != "1" ]]; then
  echo "========== 3/3 消融：lam / shrink_sum / proxy_ratio 等（每组满轮，较久） =========="
  python experiments/ablation_thesis.py configs/ch4_compare.yaml
else
  echo "========== 3/3 已跳过（SKIP_ABLATION=1） =========="
fi

echo "Done. 查看 experiments/outputs/ch4_comparison_summary_*_*.csv 与 weights/ 下产物。"
