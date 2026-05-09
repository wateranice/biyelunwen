import torch
import copy


def fed_prox_aggregate(global_model, client_models):
    """
    FedProx 的服务端聚合逻辑
    注：在算法原理上，FedProx 在服务端采用等权重聚合，
    与 FedAvg 一致。其核心差异在于客户端本地训练时的 Proximal Term。
    """
    # 深度拷贝全局模型架构
    update_model = copy.deepcopy(global_model)
    global_dict = update_model.state_dict()

    # 初始化权重字典
    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key])

    # 数量
    n_clients = len(client_models)

    # 等权重累加 (1/N)
    for client_model in client_models:
        client_dict = client_model.state_dict()
        for key in global_dict:
            global_dict[key] += client_dict[key] / n_clients

    # 更新全局模型参数
    update_model.load_state_dict(global_dict)
    return update_model
