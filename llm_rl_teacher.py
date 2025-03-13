# llm_rl_teacher.py
"""
LLMs as Strategic Policy Teachers for RL Agents
基于CartPole-v1环境的实现
"""
import gymnasium as gym
import numpy as np
import torch
import random
import argparse
from tqdm import tqdm
from core.agent import DQNAgent
from utils.plot_result import plot_results


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(42)


def train(agent, env, episodes=200, max_steps=500, render=False):
    """训练智能体"""
    rewards = []
    episode_lengths = []

    for episode in tqdm(range(episodes), desc="训练进度"):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 学习
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        episode_lengths.append(step + 1)

        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = sum(rewards[-10:]) / 10
            print(f"Episode {episode + 1}/{episodes}, 平均奖励: {avg_reward:.2f}, 探索率: {agent.epsilon:.2f}")

    return rewards, episode_lengths


def evaluate(agent, env, episodes=10, max_steps=500):
    """评估智能体"""
    rewards = []
    episode_lengths = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # 在评估时使用贪婪策略
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = agent.policy_net(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        episode_lengths.append(step + 1)

    avg_reward = sum(rewards) / len(rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)
    print(f"评估结果: 平均奖励 = {avg_reward:.2f}, 平均步数 = {avg_length:.2f}")
    return avg_reward, avg_length


def llm_rl_teacher():
    parser = argparse.ArgumentParser(description='LLM作为RL智能体的策略教师')
    parser.add_argument('--episodes', type=int, default=200, help='训练轮次')
    parser.add_argument('--render', action='store_true', help='渲染环境')
    parser.add_argument('--no-llm', action='store_true', help='不使用LLM')
    parser.add_argument('--llm-weight', type=float, default=0.3, help='LLM建议权重')
    parser.add_argument('--save', action='store_true', help='保存模型')
    parser.add_argument('--load', action='store_true', help='加载模型')
    parser.add_argument('--model-path', type=str, default="models/dqn_model.pth", help='模型路径')
    args = parser.parse_args()

    # 创建环境
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 创建智能体
    agent_with_llm = DQNAgent(state_size, action_size, use_llm=True, llm_weight=args.llm_weight)
    agent_without_llm = DQNAgent(state_size, action_size, use_llm=False)

    if args.load:
        agent_with_llm.load(args.model_path)
        agent_without_llm.load(args.model_path.replace('.pth', '_no_llm.pth'))

    # 训练
    if not args.no_llm:
        print("正在训练使用LLM的智能体...")
        rewards_with_llm, lengths_with_llm = train(agent_with_llm, env, episodes=args.episodes)

        print("正在训练不使用LLM的智能体...")
        rewards_without_llm, lengths_without_llm = train(agent_without_llm, env, episodes=args.episodes)

        # 绘制结果
        plot_results(
            [rewards_with_llm, rewards_without_llm],
            ["with_LLM", "without_LLM"],
            "media/training_result.png",
            "Comparison of the impact of LLM strategies on teachers"
        )

        # 评估
        print("\n使用LLM的智能体评估:")
        evaluate(agent_with_llm, env)

        print("\n不使用LLM的智能体评估:")
        evaluate(agent_without_llm, env)

        if agent_with_llm.use_llm:
            print(f"\nLLM总查询次数: {agent_with_llm.llm_teacher.query_count}")
    else:
        # 只训练不使用LLM的智能体
        print("正在训练基础智能体...")
        rewards_without_llm, lengths_without_llm = train(agent_without_llm, env, episodes=args.episodes)

        # 绘制结果
        plot_results(
            [rewards_without_llm],
            ["base_DQN"],
            "DQN_result"
        )

        # 评估
        print("\n基础智能体评估:")
        evaluate(agent_without_llm, env)

    # 保存模型
    if args.save:
        if not args.no_llm:
            agent_with_llm.save(args.model_path)
        agent_without_llm.save(args.model_path.replace('.pth', '_no_llm.pth'))


if __name__ == "__main__":
    llm_rl_teacher()
