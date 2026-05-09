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

# 1. 实验配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_clients = 10
rounds = 30  # 轮次
beta = 0.5  # 设置中等程度的 Non-IID
batch_size = 128  # 适当增大以加速训练

print(f"Starting CIFAR-10 Comparison: {n_clients} Clients, Beta={beta}")

# 2. 准备数据
train_ds, test_ds, proxy_ds, client_map = prepare_data(n_clients=n_clients, beta=beta)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def run_experiment(method='adaptive'):
    # 初始化模型
    global_model = SimpleCNN().to(device)
    acc_history = []

    for r in range(rounds):
        client_models = []
        # 模拟客户端本地训练
        for i in range(n_clients):
            local_model = copy.deepcopy(global_model)
            local_model.train()
            optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)

            indices = client_map[i]
            loader = DataLoader(Subset(train_ds, indices), batch_size=batch_size, shuffle=True)

            # 每个客户端本地练 2 轮以加快收敛
            for epoch in range(2):
                for imgs, labels in loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(local_model(imgs), labels)
                    loss.backward()
                    optimizer.step()
            client_models.append(local_model)

        # 服务器聚合
        if method == 'adaptive':
            global_model, _, _ = adaptive_aggregate(global_model, client_models, proxy_ds, device)
        else:
            global_model = fed_avg_aggregate(global_model, client_models)

        # 测试全局准确率
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = global_model(imgs)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        acc = 100 * correct / total
        acc_history.append(acc)
        print(f"{method.upper()} Round {r + 1}: Accuracy = {acc:.2f}%")

    return acc_history


if __name__ == "__main__":
    # 3. 执行对比实验
    print("\n--- Training with FedAvg ---")
    fedavg_acc = run_experiment(method='fedavg')

    print("\n--- Training with Adaptive Weighting ---")
    adaptive_acc = run_experiment(method='adaptive')

    # 4. 绘制对比图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, rounds + 1), fedavg_acc, label='FedAvg', marker='o', color='gray')
    plt.plot(range(1, rounds + 1), adaptive_acc, label='Ours (Adaptive)', marker='s', color='blue')
    plt.title(f'Comparison on CIFAR-10 (Non-IID Beta={beta})')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('cifar10_comparison.png')
    plt.show()
