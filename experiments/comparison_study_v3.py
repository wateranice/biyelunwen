import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import copy
import matplotlib.pyplot as plt

from models import SimpleCNN
from sever import fed_avg_aggregate, adaptive_aggregate
from dataset import prepare_data

# 1. 实验全局配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_clients = 10
rounds = 30  # 之后轮次
beta = 0.5  # Non-IID 程度
batch_size = 128
mu = 0.01  # FedProx 的近端项系数

print(f"--- 实验开始: CIFAR-10 | 客户端: {n_clients} | Beta: {beta} ---")

# 2. 准备数据
train_ds, test_ds, proxy_ds, client_map = prepare_data(n_clients=n_clients, beta=beta)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def run_experiment(method='fedavg'):
    print(f"\n正在运行算法: {method.upper()}...")
    global_model = SimpleCNN().to(device)

    acc_history = []
    loss_history = []

    for r in range(rounds):
        client_models = []
        total_round_loss = 0
        num_batches = 0

        for i in range(n_clients):
            local_model = copy.deepcopy(global_model)
            local_model.train()
            optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)

            loader = DataLoader(Subset(train_ds, client_map[i]), batch_size=batch_size, shuffle=True)

            # 本地训练 1 轮
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                output = local_model(images)
                base_loss = nn.CrossEntropyLoss()(output, labels)

                # FedProx 核心逻辑：添加近端项惩罚
                if method == 'fedprox':
                    proximal_term = 0.0
                    for l_p, g_p in zip(local_model.parameters(), global_model.parameters()):
                        proximal_term += (l_p - g_p).pow(2).sum()
                    loss = base_loss + (mu / 2) * proximal_term
                else:
                    loss = base_loss

                loss.backward()
                optimizer.step()

                total_round_loss += loss.item()
                num_batches += 1

            client_models.append(local_model)

        # 服务端聚合
        if method == 'adaptive':
            # 你的创新算法：基于代理数据的自适应加权
            global_model, _, _ = adaptive_aggregate(global_model, client_models, proxy_ds, device)
        else:
            # FedAvg 和 FedProx 默认使用简单权重平均
            global_model = fed_avg_aggregate(global_model, client_models)

        # 评估本轮准确率
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_round_loss / num_batches

        acc_history.append(accuracy)
        loss_history.append(avg_loss)

        if (r + 1) % 5 == 0 or r == 0:
            print(f"Round {r + 1}/{rounds} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

    return acc_history, loss_history


if __name__ == "__main__":
    # 3. 顺序执行三个实验
    fedavg_acc, fedavg_loss = run_experiment(method='fedavg')
    fedprox_acc, fedprox_loss = run_experiment(method='fedprox')
    adaptive_acc, adaptive_loss = run_experiment(method='adaptive')

    # 4. 绘图与保存 (满足需求 4.1)
    epochs = range(1, rounds + 1)

    # 图 1：准确率对比
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, fedavg_acc, 'g--', label='FedAvg (Baseline)')
    plt.plot(epochs, fedprox_acc, 'r-.', label='FedProx (Baseline)')
    plt.plot(epochs, adaptive_acc, 'b-', linewidth=2, label='mine (Adaptive Weighting)')
    plt.title(f'Test Accuracy Comparison (CIFAR-10, Beta={beta})')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('result_accuracy_comparison.png', dpi=300)

    # 图 2：收敛损失对比
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, fedavg_loss, 'g--', label='FedAvg')
    plt.plot(epochs, fedprox_loss, 'r-.', label='FedProx')
    plt.plot(epochs, adaptive_loss, 'b-', linewidth=2, label='mine (Adaptive)')
    plt.title('Convergence Curve (Loss vs. Rounds)')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('result_loss_comparison.png', dpi=300)

    plt.show()
    print("\nSuccess: 所有对比实验完成，结果图片已保存！")
