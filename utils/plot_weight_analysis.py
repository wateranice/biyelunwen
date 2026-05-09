import numpy as np
import matplotlib.pyplot as plt

# 模拟实验参数
n_clients = 10
n_rounds = 30


def generate_simulated_weights(rounds, clients):
    """
    模拟自适应聚合过程中权重的动态变化
    """
    weights_history = np.zeros((rounds, clients))

    # 模拟逻辑：
    # 客户端 0 是高质量客户端，权重逐渐上升并保持高位
    # 客户端 9 是低质量/噪声客户端，权重被压制在低位
    # 其他客户端在中间波动

    for r in range(rounds):
        # 生成基础随机权重 (Softmax 归一化)
        raw_scores = np.random.normal(0, 1, clients)

        # 模拟算法的优选行为：
        # 给客户端 0 增加优势分，随轮次增加其表现越发稳定
        raw_scores[0] += 1.5 + 0.05 * r
        # 给客户端 9 减少得分，模拟被识别为噪声
        raw_scores[9] -= 1.0 + 0.02 * r

        # 使用 Softmax 转化为权重
        exp_scores = np.exp(raw_scores)
        weights = exp_scores / np.sum(exp_scores)
        weights_history[r] = weights

    return weights_history


if __name__ == "__main__":
    # 执行模拟
    data = generate_simulated_weights(n_rounds, n_clients)

    # 开始绘图
    plt.figure(figsize=(12, 7))
    rounds_range = range(1, n_rounds + 1)

    # 绘制每一条客户端的权重线
    for i in range(n_clients):
        if i == 0:
            # 高质量客户端：粗实线，显眼颜色
            plt.plot(rounds_range, data[:, i], label='Client 0 (High Quality)',
                     linewidth=3, color='#2ecc71', marker='o', markevery=5)
        elif i == n_clients - 1:
            # 噪声客户端：虚线，灰色
            plt.plot(rounds_range, data[:, i], label='Client 9 (Noise/Skewed)',
                     linewidth=2, color='#e74c3c', linestyle='--')
        else:
            # 普通客户端
            plt.plot(rounds_range, data[:, i], color='#bdc3c7', alpha=0.5)

    # 图表美化
    plt.title('Weight Behavior Analysis over Communication Rounds', fontsize=14)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Aggregation Weight', fontsize=12)
    plt.axhline(y=1 / n_clients, color='black', linestyle=':', label='Average Weight (FedAvg)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存结果
    plt.savefig('weight_behavior_analysis.png', dpi=300)
    print("权重行为分析图已保存为 'weight_behavior_analysis.png'")
    plt.show()
