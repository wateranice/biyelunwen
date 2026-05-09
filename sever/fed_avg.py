import torch


def fed_avg_aggregate(global_model, client_models):
    """
    FedAvg聚合：将所有客户端模型参数取平均值
    """
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        # 累加所有客户端在该层的权重
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))],
                                     0).mean(0)

    # 更新全局模型
    global_model.load_state_dict(global_dict)
    return global_model
