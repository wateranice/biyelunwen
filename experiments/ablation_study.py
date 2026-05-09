import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import matplotlib.pyplot as plt
from models import SimpleCNN
from sever import adaptive_aggregate
from dataset import prepare_data
from torch.utils.data import DataLoader, Subset
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 测试不同比例的代理数据（600, 1200, 3000条）
proxy_sizes = [600, 1200, 3000]
beta = 0.1  # 专门在最难的情况下测试
rounds = 10


def run_ablation(p_size):
    print(f"\n[Ablation] Testing Proxy Size: {p_size}")
    train_ds, test_ds, _, client_map = prepare_data(n_clients=10, beta=beta)
    # 动态切出不同大小的代理数据
    p_data, _ = torch.utils.data.random_split(train_ds, [p_size, len(train_ds) - p_size])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = SimpleCNN().to(device)
    acc_history = []

    for r in range(rounds):
        client_models = []
        for i in range(10):
            local_model = copy.deepcopy(model)
            local_model.train()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
            loader = DataLoader(Subset(train_ds, client_map[i]), batch_size=64, shuffle=True)
            for img, lbl in loader:
                img, lbl = img.to(device), lbl.to(device)
                optimizer.zero_grad()
                torch.nn.CrossEntropyLoss()(local_model(img), lbl).backward()
                optimizer.step()
            client_models.append(local_model)

        # 聚合
        model, _, _ = adaptive_aggregate(model, client_models, p_data, device)

        # 测试
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for img, lbl in test_loader:
                img, lbl = img.to(device), lbl.to(device)
                _, pred = torch.max(model(img).data, 1)
                total += lbl.size(0)
                correct += (pred == lbl).sum().item()
        acc_history.append(100 * correct / total)
    return acc_history


if __name__ == "__main__":
    # 运行并绘图
    results = {size: run_ablation(size) for size in proxy_sizes}
    plt.figure(figsize=(10, 6))
    for size, accs in results.items():
        plt.plot(range(1, rounds + 1), accs, marker='o', label=f'Proxy Size: {size}')

    plt.title('Ablation Study: Does more Proxy Data help at Beta=0.1?')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('ablation_study.png')
    plt.show()
