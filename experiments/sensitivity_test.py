import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import matplotlib.pyplot as plt
from models import SimpleCNN
from sever import fed_avg_aggregate, adaptive_aggregate
from dataset import prepare_data
import copy
from torch.utils.data import DataLoader, Subset

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 实验参数设置
betas = [0.1, 0.5, 5.0]  # 测试三种不同的异构程度
rounds = 5  # 临时跑5轮，之后改跑30-50轮
mu = 0.01  # FedProx 的近端项系数


def run_specific_experiment(beta, algo='adaptive'):
    """
    运行特定算法在特定异构程度下的实验
    algo 选项: 'fedavg', 'fedprox', 'adaptive'
    """
    # 1. 准备数据划分
    train_ds, test_ds, proxy, client_map = prepare_data(n_clients=10, beta=beta)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 2. 初始化全局模型
    global_model = SimpleCNN().to(device)

    for r in range(rounds):
        client_models = []
        # 模拟 10 个客户端的本地训练
        for i in range(10):
            local_model = copy.deepcopy(global_model)
            local_model.train()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
            loader = DataLoader(Subset(train_ds, client_map[i]), batch_size=64, shuffle=True)

            for img, lbl in loader:
                img, lbl = img.to(device), lbl.to(device)
                optimizer.zero_grad()

                # 计算预测输出
                outputs = local_model(img)
                loss = torch.nn.CrossEntropyLoss()(outputs, lbl)

                # 如果是 FedProx，增加近端项 (Proximal Term)
                if algo == 'fedprox':
                    proximal_term = 0.0
                    for local_p, global_p in zip(local_model.parameters(), global_model.parameters()):
                        proximal_term += (local_p - global_p).norm(2) ** 2
                    loss += (mu / 2) * proximal_term

                loss.backward()
                optimizer.step()

            client_models.append(local_model)

        # 3. 服务端聚合策略
        if algo == 'fedavg' or algo == 'fedprox':
            # FedProx 和 FedAvg 在聚合阶段逻辑一致，区别在本地训练
            global_model = fed_avg_aggregate(global_model, client_models)
        elif algo == 'adaptive':
            # 自适应加权聚合
            global_model, _, _ = adaptive_aggregate(global_model, client_models, proxy, device)

    # 4. 评估最终模型在测试集上的准确率
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, lbl in test_loader:
            img, lbl = img.to(device), lbl.to(device)
            out = global_model(img)
            _, pred = torch.max(out.data, 1)
            total += lbl.size(0)
            correct += (pred == lbl).sum().item()

    final_acc = 100 * correct / total
    return final_acc


if __name__ == "__main__":
    # 主程序：开始执行多组对比实验
    results_fedavg = []
    results_fedprox = []
    results_adaptive = []

    for b in betas:
        print(f"\n 正在测试异构度 Beta = {b} ...")

        print("跑 FedAvg...")
        results_fedavg.append(run_specific_experiment(b, 'fedavg'))

        print("跑 FedProx...")
        results_fedprox.append(run_specific_experiment(b, 'fedprox'))

        print("跑 mine (Adaptive)...")
        results_adaptive.append(run_specific_experiment(b, 'adaptive'))

    # 绘制三算法鲁棒性对比柱状图
    plt.figure(figsize=(12, 7))
    x = range(len(betas))
    width = 0.25  # 柱子宽度

    # 绘制三组对比柱
    plt.bar([i - width for i in x], results_fedavg, width=width, label='FedAvg', color='#95a5a6')  # 灰色
    plt.bar([i for i in x], results_fedprox, width=width, label='FedProx', color='#e74c3c')  # 红色
    plt.bar([i + width for i in x], results_adaptive, width=width, label='mine (Adaptive)', color='#3498db')  # 蓝色

    # 图表装饰
    plt.xticks(x, [f'Beta={b} (Non-IID)' for b in betas])
    plt.ylabel('Final Test Accuracy (%)', fontsize=12)
    plt.title('Algorithm Robustness under different Non-IID Levels (CIFAR-10)', fontsize=14)
    plt.legend(loc='lower right')
    plt.ylim(min(results_fedavg + results_fedprox + results_adaptive) - 10, 40)  # Y轴
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 保存并展示
    plt.tight_layout()
    plt.savefig('robustness_test.png', dpi=300)
    print("\n 更新后的鲁棒性测试图已保存为 'robustness_test.png'")
    plt.show()
