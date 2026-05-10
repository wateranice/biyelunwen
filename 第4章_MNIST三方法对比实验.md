# MNIST 三方法对比实验（第 4 章 / 任务书「三数据集」— E4-2）

本节依据 **`experiments/outputs_48425/outputs/outputs/ch4_comparison_summary_mnist_dirichlet.csv`**（SSH 端口 **48425** 实例下载结果）整理 **MNIST** 在 **Dirichlet Non-IID** 下 **FedAvg、FedProx、本文自适应聚合（Adaptive）** 的对比，与 CIFAR-10、Fashion-MNIST 并列，满足任务书「多数据集」要求。

---

## 实验设定（与 `configs/ch4_mnist.yaml` 一致）

- **数据集：** MNIST（10 类，\(28\times28\) 灰度）。  
- **模型：** 仓库中 `create_model("mnist")` 对应的轻量 CNN，全局与各客户端结构相同。  
- **Non-IID：** 训练集按 **Dirichlet** 数量划分分配至 \(K=10\) 个客户端；配置字段 **`dirichlet_beta`** 与论文异构强度 **\(\alpha\)** 对应，**\(\alpha\) 越小** 客户端类别分布越不均衡。本节报告 **\(\alpha\in\{0.05,0.1,0.5,1.0\}\)**。  
- **通信与本地训练：** 全局轮数 **30**；每客户端每轮 **2** 个 local epoch，批大小 **128**，SGD **学习率 0.01**、**动量 0.9**；FedProx **\(\mu=0.01\)**。  
- **代理数据：** 训练集比例 **4%**，与客户端索引 **互斥**（`proxy_disjoint_from_clients: true`）。  
- **可复核命令：**  
  `python experiments/comparison_aligned.py configs/ch4_mnist.yaml --betas 0.05 0.1 0.5 1.0 --thresholds 50,55,60`  

评价指标：末轮 **测试集 Top-1 准确率（%）**；相对阈值 \(T\in\{50\%,55\%,60\%\}\) 的 **首次达到该精度的通信轮次** \(R(T)\)（见 CSV 列）。

---

## 末轮测试精度（%）

**表 4-X　MNIST、Dirichlet、30 轮末轮 Top-1 精度**

| \(\alpha\) | FedAvg | FedProx | Adaptive |
|:------------:|:------:|:-------:|:--------:|
| 0.05 | 97.37 | 97.08 | **97.65** |
| 0.10 | 98.34 | 98.24 | **98.41** |
| 0.50 | 99.02 | 98.99 | 98.99 |
| 1.00 | 99.05 | 99.03 | **99.12** |

（数据摘自 `ch4_comparison_summary_mnist_dirichlet.csv`。）

---

## 达阈值通信轮次 \(R(T)\)

**表 4-X（续）　首次达到测试精度 \(\geq T\) 的轮次**

| \(\alpha\) | 方法 | \(R(50\%)\) | \(R(55\%)\) | \(R(60\%)\) |
|:------------:|:------:|:-----------:|:-----------:|:-----------:|
| 0.05 | FedAvg | 4 | 6 | 7 |
| 0.05 | FedProx | 6 | 7 | 7 |
| 0.05 | Adaptive | 6 | 7 | 7 |
| 0.10 | 三方法 | 2 | 2 | 2 |
| 0.50 | 三方法 | 1 | 1 | 2 |
| 1.00 | 三方法 | 1 | 1 | 1 |

在 **\(\alpha=0.05\)**（较强异构）下，**FedAvg** 的 \(R(50\%)=4\) 早于 FedProx / Adaptive 的 **6**，但 **末轮精度** Adaptive **最高**（97.65%），写作时宜结合 **`comparison_acc_beta0p05_mnist_dirichlet.png`** 看全程曲线再表述「首达阈值」与「末轮精度」的关系。

---

## 数据与曲线文件位置

汇总表路径（本机）：

`experiments/outputs_48425/outputs/outputs/ch4_comparison_summary_mnist_dirichlet.csv`

同批实验的 **Acc / Loss / Proxy loss** 曲线与 **`.npz`** 一般位于同一 **`outputs`** 目录下，文件名形如：

`comparison_{acc,loss,proxy_loss}_beta{α}_mnist_dirichlet.{png,npz}`

（若下载时未包含全部图，可从原服务器 **`experiments/outputs/`** 补拷。）

---

## 简要讨论（写作提示）

1. **MNIST 上末轮精度整体很高**（约 **97%～99%**），三方法差距 **小于 1 个百分点** 的情况较多，可强调：在**相对简单**的数据集上，异构强度 \(\alpha\) 从 **0.5** 增至 **1.0** 时三者趋于饱和，**增益空间有限**。  
2. **\(\alpha=0.05\)** 时 Adaptive **末轮略优**，可作为「强异构下可学习聚合仍有边际收益」的佐证。  
3. **勿与 CIFAR-10 / FMNIST 直接比绝对百分数**；应分数据集报告，并统一 **\(\alpha\)、\(K\)、轮数** 等符号。

---

*表号「4-X」请按全论文统一编号。若将 `outputs_48425` 整理为单层目录，请相应更新文中路径。*
