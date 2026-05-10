# Fashion-MNIST 三方法对比实验（第 4 章 / 任务书「三数据集」之一）

本节依据本机目录 **`experiments/outputs_server_25477/outputs/`** 中的汇总表与曲线文件，整理 **Fashion-MNIST** 在 **Dirichlet Non-IID** 下 **FedAvg、FedProx、本文自适应聚合（Adaptive）** 的对比结果，对应实验编号 **E4-3**，并与 CIFAR-10 主实验（第 4.4 节）写法并列，便于体现任务书「多数据集」要求。

---

## 实验设定（与 `configs/ch4_fmnist.yaml` 一致）

- **数据集：** Fashion-MNIST（10 类，\(28\times28\) 灰度）。  
- **模型：** 与仓库实现一致的轻量 CNN（`create_model("fmnist")`），全局与各客户端结构相同。  
- **Non-IID：** 训练集按 **Dirichlet** 数量划分分配至 \(K=10\) 个客户端；配置字段 **`dirichlet_beta`** 与论文中异构强度参数 **\(\alpha\)** 对应，**\(\alpha\) 越小** 客户端标签分布越不均衡。本节报告 **\(\alpha\in\{0.05,0.1,0.5,1.0\}\)** 四组取值。  
- **通信与本地训练：** 全局轮数 **30**；每客户端每轮 **2** 个 local epoch，批大小 **128**，SGD **学习率 0.01**、**动量 0.9**；FedProx 近端系数 **\(\mu=0.01\)**（与配置 `fedprox_mu` 一致）。  
- **代理数据：** 占训练集比例 **4%**，且与客户端训练索引 **集合互斥**（`proxy_disjoint_from_clients: true`）。  
- **运行命令（可复核）：**  
  `python experiments/comparison_aligned.py configs/ch4_fmnist.yaml --betas 0.05 0.1 0.5 1.0 --thresholds 50,55,60`  

评价指标与第 4.2 节一致：**末轮全局模型在测试集上的 Top-1 准确率（%）**；以及相对阈值 \(T\in\{50\%,55\%,60\%\}\) 的 **首次达到该精度的通信轮次** \(R(T)\)（见汇总表列）。

---

## 末轮测试精度（%）

**表 4-X　Fashion-MNIST、Dirichlet、30 轮末轮 Top-1 精度**

| \(\alpha\) | FedAvg | FedProx | Adaptive |
|------------|--------|---------|----------|
| 0.05 | 79.40 | 80.60 | **81.77** |
| 0.10 | 83.17 | 82.71 | **83.04** |
| 0.50 | 89.37 | 89.20 | 89.04 |
| 1.00 | 89.29 | 89.07 | **89.48** |

（数据摘自 **`ch4_comparison_summary_fmnist_dirichlet.csv`**。）

---

## 达阈值通信轮次 \(R(T)\)

**表 4-X（续）　首次达到测试精度 \(\geq T\) 的轮次（空表示 30 轮内未达到）**

| \(\alpha\) | 方法 | \(R(50\%)\) | \(R(55\%)\) | \(R(60\%)\) |
|------------|------|-------------|-------------|-------------|
| 0.05 | FedAvg | 3 | 8 | 10 |
| 0.05 | FedProx | 3 | 8 | 10 |
| 0.05 | Adaptive | 3 | 10 | 10 |
| 0.10 | FedAvg | 3 | 4 | 6 |
| 0.10 | FedProx | 3 | 4 | 6 |
| 0.10 | Adaptive | 3 | 4 | 6 |
| 0.50 | 三方法 | 2 | 2 | 2 |
| 1.00 | FedAvg | 1 | 1 | 2 |
| 1.00 | FedProx | 1 | 1 | 2 |
| 1.00 | Adaptive | 1 | 1 | 1 |

在 **\(\alpha=0.05\)**（强异构）下，**Adaptive** 末轮精度 **高于** FedAvg / FedProx，且 **\(R(55\%)\)** 为 **10**（相对 FedAvg/FedProx 的 **8**）——是否强调「更快达 55%」需结合 **`comparison_acc_beta0p05_fmnist_dirichlet.png`** 全曲线判断（若 Adaptive 中后期反超，则末轮 \(R(55\%)\) 列可能晚于另两种方法的首达轮次定义方式，定稿时请与图一致表述）。

---

## 曲线与可复核文件（`outputs_server_25477/outputs/`）

同目录下保存有各 \(\alpha\) 的 **准确率 / 测试集平均交叉熵 / 代理 meta-loss** 对比图及数值：

- **准确率：** `comparison_acc_beta0p05_fmnist_dirichlet.png`、`comparison_acc_beta0p1_fmnist_dirichlet.png`、`comparison_acc_beta0p5_fmnist_dirichlet.png`、`comparison_acc_beta1p0_fmnist_dirichlet.png`  
- **测试 Loss：** `comparison_loss_beta0p05_fmnist_dirichlet.png` 等（同 \(\alpha\) 命名规则）  
- **代理 Loss（Adaptive 有效）：** `comparison_proxy_loss_beta*_fmnist_dirichlet.png`  
- **数值：** `comparison_acc_beta*_fmnist_dirichlet.npz`（内含 `acc_fedavg` / `acc_fedprox` / `acc_adaptive` 及 `test_loss_*`、`proxy_loss_*` 等字段）

汇总表路径：**`ch4_comparison_summary_fmnist_dirichlet.csv`**。

---

## 简要讨论（写作提示）

1. **强异构 \(\alpha=0.05\)：** 本文方法末轮精度 **81.77%**，相对 FedAvg **+2.37** 个百分点、相对 FedProx **+1.17** 个百分点，与「可学习聚合缓解客户端漂移」的叙述方向一致。  
2. **\(\alpha=0.1\)：** 三方法末轮精度接近，Adaptive **83.04%** 略高于 FedAvg。  
3. **\(\alpha\in\{0.5,1.0\}\)**（异构减弱）：三方法末轮精度 **约 89%** 量级，差异缩小；此时 **通信与局部训练配置** 对瓶颈的影响更大，可在局限中说明「温和 Non-IID 下增益空间有限」。  
4. **与 CIFAR-10 对照：** FMNIST 上末轮绝对精度整体高于 CIFAR-10 同设定下的典型水平，**不宜跨数据集直接比较百分数**；应强调 **相对增益** 与 **曲线形态**（见各 `comparison_acc_*_fmnist_dirichlet.png`）。

---

*表号「4-X」请在全论文章节中统一编号。数据根目录：`experiments/outputs_server_25477/outputs/`。*
