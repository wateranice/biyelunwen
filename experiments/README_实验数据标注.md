# `experiments/` 下实验数据标注（对照《第4章_实验编号与任务书对照表》）

> 说明：下列仅根据 **本仓库当前磁盘上已有文件** 归纳；**服务器上已跑完但未下载** 的内容不在此列。Python 脚本（`.py`）为代码，**不**计入「实验数据」。

---

## 1. 各文件夹里「已经有什么」

| 本地路径 | 已有数据（摘要） | 对应编号（若可判定） |
|----------|------------------|----------------------|
| **`outputs/`** | `ch4_comparison_summary.csv`：无 `dataset` 列，`global_rounds=6`，仅 \(\beta\in\{0.1,0.5,1.0\}\) 三档 | **试跑/旧版**，**不宜**作为论文正式 E4-1 |
| **`outputs_server/`** | `ch4_comparison_summary.csv`：`dataset=cifar10`，`partition=dirichlet`，**30 轮**，\(\beta\in\{0.05,0.1,0.3,0.5,1.0\}\) 全组 × 三方法 | **E4-1**（CIFAR Dirichlet）、**E3-2**、**E4-4** 汇总；E4-5 曲线需同批 **`comparison_*_cifar10_dirichlet.*`**（本机若未一并保存请从服务器补） |
| **`outputs_server_25477/outputs/`** | `ch4_comparison_summary_fmnist_dirichlet.csv`：**FMNIST**、Dirichlet、30 轮、四组 \(\beta\) × 三方法；另有部分 `*_cifar10_dirichlet.*`、`*_from_log*` 等混放 | **E4-3** 主表齐全；CIFAR 图为**另次实验**残留，勿与 FMNIST 表混为一谈 |
| **`outputs_quanzhong/`** | `ch4_comparison_summary_fmnist_dirichlet.csv` 等（与 25477 同源备份时可核对） | 多为 **E4-3** 备份；另含旧版无 `dataset` 的 `ch4_comparison_summary.csv` 时注意甄别 |
| **`outputs_48425/outputs/outputs/`** | `ch4_comparison_summary_mnist_dirichlet.csv`：**MNIST**、Dirichlet、30 轮、四组 \(\beta\) × 三方法 | **E4-2** 汇总齐全（路径多一层 `outputs` 为下载时目录习惯，不影响数据） |
| **`outputs_server_autodl/outputs/`** | `ch4_comparison_summary.csv`：仅 **pathological**、CIFAR-10、两档 \(\beta\) × 三方法 | **E3-3** 病态汇总（与论文 4.5 一致）；若你曾在服务器还有 **Dirichlet CIFAR** 大表，可能未拷进本路径，需自行核对 |

---

## 2. 对照任务书：哪些「在本机 experiments 下」算已做完

| 编号 | 结论（基于上表） |
|------|------------------|
| **E3-2**（Dirichlet 设定） | 已随 **E4-1** 体现在 `outputs_server` 的 CIFAR Dirichlet 实验中 |
| **E4-1** | **有**：`outputs_server/ch4_comparison_summary.csv`（CIFAR、30 轮、多 \(\beta\)） |
| **E4-2**（MNIST） | **有**：`outputs_48425/.../ch4_comparison_summary_mnist_dirichlet.csv` |
| **E4-3**（FMNIST） | **有**：`outputs_server_25477/...` 与 `outputs_quanzhong/` 下 FMNIST 汇总 |
| **E3-3**（病态） | **有**：`outputs_server_autodl/...` 中 pathological 汇总（论文 4.5 已写） |
| **E4-4 / E4-5** | **部分**：汇总在 `outputs_server` 等；**Loss 曲线**依赖各目录是否含 `comparison_loss_*.png` / `*.npz`，请按文件名自查；本机 `experiments/` 下 **未** 扫到完整一套 `e4_weight_vis` 或全量 `comparison_*` 时，视为 **图文件可能仍在服务器** |

---

## 3. 本机 `experiments/` 下**未看到**或**需你自行确认**的项

| 编号 | 说明 |
|------|------|
| **E3-1** | 代理比例 1%～5% 为 **配置项**，随主实验生效；**独立消融表**见 **E4-7**（`ablation_summary_*.csv` 应在 `experiments/outputs/` 或服务器，**本目录树当前未出现**） |
| **E4-6** | 权重可视化产物默认在 **`experiments/outputs/e4_weight_vis/`** 及 **`weights/ch4/e4_weight_vis_*_run_log.npz`**；**不在**当前 `experiments/` 子目录列表中 → **视为未下载到本仓库或未跑完** |
| **E4-7 / E4-8 / E4-9** | `ablation_thesis.py` 产出 **`ablation_summary_{dataset}_{partition}.csv`** 与 **`weights/ch4/abl_*.pth`**；**本机 experiments 下未列出** → **未完成或未拉回**（若在服务器仅 `scp` 即可） |

---

## 4. 建议下一步（写论文前）

1. **从服务器补拉**：`comparison_acc/loss/proxy_loss_*` 的 **`.png` / `.npz`**（与已有 CSV 同批），满足 **E4-5** 插图。  
2. **E4-6**：在服务器跑 `main.py configs/ch4_weight_vis.yaml --plot-after`（或 FMNIST 版），将 **`e4_weight_vis/`** 与 **`*_run_log.npz`** 拷回本机。  
3. **E4-7～E4-9**：跑完 `ablation_thesis.py` 后下载 **`ablation_summary_*.csv`** 与 **`abl_*_run_log.npz`**。  
4. **清理/归档**：`outputs/` 中 **6 轮** 旧表建议标注「非正式」或删除，避免与 **E4-1** 正式结果混淆。

---

*本文档随本机下载内容变化；更新日期以 git 为准。对照表路径：`第4章_实验编号与任务书对照表.md`。*
