import torch
import torch.nn as nn
import torch.nn.functional as F


def adaptive_aggregate(global_model, client_models, proxy_data, device, lam=0.1):
    """
    带权重收缩的自适应聚合
    """
    global_model.eval()
    proxy_loader = torch.utils.data.DataLoader(proxy_data, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    losses = []
    # 获取每个客户端的loss
    with torch.no_grad():
        for client_model in client_models:
            client_model.eval()
            total_loss = 0
            for images, labels in proxy_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = client_model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            losses.append(total_loss / len(proxy_loader))

    # 基础自适应权重
    loss_tensor = torch.tensor(losses)
    adaptive_w = F.softmax(-loss_tensor, dim=0)

    # 权重收缩逻辑 (Weight Shrinkage)
    avg_w = torch.ones_like(adaptive_w) / len(client_models)
    final_weights = (1 - lam) * adaptive_w + lam * avg_w

    # 执行聚合
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.zeros_like(global_dict[k]).float()
        for i in range(len(client_models)):
            global_dict[k] += final_weights[i] * client_models[i].state_dict()[k].to(device)

    global_model.load_state_dict(global_dict)
    return global_model, final_weights
