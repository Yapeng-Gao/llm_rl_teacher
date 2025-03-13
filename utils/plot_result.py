from matplotlib import pyplot as plt


def plot_results(rewards_list, labels, save_path, title="训练奖励对比"):
    """绘制训练结果对比图"""
    plt.figure(figsize=(12, 6))

    for rewards, label in zip(rewards_list, labels):
        # 计算移动平均
        window_size = 10
        moving_avg = [sum(rewards[max(0, i - window_size):i]) / min(i, window_size) for i in range(1, len(rewards) + 1)]
        plt.plot(moving_avg, label=label)

    plt.xlabel('episode')
    plt.ylabel('avg_reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
