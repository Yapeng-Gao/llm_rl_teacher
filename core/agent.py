# DQN智能体
import os
import random
import torch
from torch import nn, optim
from core.model import DQN
from core.policy import LLMPolicyTeacher
from utils.buffer import ReplayBuffer
from utils.transition import Transition


class DQNAgent:
    def __init__(self, state_size, action_size, use_llm=False, llm_weight=0.3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.use_llm = use_llm
        self.llm_weight = llm_weight
        self.query_frequency = 20  # 每20步查询一次LLM
        self.step_count = 0

        # 神经网络
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # LLM教师
        if use_llm:
            self.llm_teacher = LLMPolicyTeacher()
            self.action_history = []
            self.reward_history = []

    def select_action(self, state):
        rl_action = self._select_rl_action(state)

        if not self.use_llm:
            return rl_action

        # 定期咨询LLM
        if self.step_count % self.query_frequency == 0:
            llm_advice = self.llm_teacher.get_policy_advice(
                state, self.action_history, self.reward_history
            )
            llm_action = llm_advice["action"]
            confidence = llm_advice["confidence"] / 10.0  # 转换为0-1范围

            # 结合RL和LLM的建议
            if random.random() < self.llm_weight * confidence:
                return llm_action

        return rl_action

    def _select_rl_action(self, state):
        # ε-greedy策略
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def learn(self, state, action, reward, next_state, done):
        # 存储经验
        self.memory.push(state, action, next_state, reward, done)

        # 记录动作和奖励，用于LLM咨询
        if self.use_llm:
            self.action_history.append(action)
            self.reward_history.append(reward)
            # 只保留最近的50个动作和奖励
            if len(self.action_history) > 50:
                self.action_history.pop(0)
                self.reward_history.pop(0)

        # 更新步数计数
        self.step_count += 1

        # 当记忆库足够大时，进行批量学习
        if len(self.memory) < self.batch_size:
            return

        # 从记忆库中随机采样
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 准备批量数据
        non_final_mask = torch.tensor([not done for done in batch.done], dtype=torch.bool)
        non_final_next_states = torch.FloatTensor([s for s, d in zip(batch.next_state, batch.done) if not d])

        state_batch = torch.FloatTensor(batch.state)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)

        # 计算当前Q值
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 计算期望的Q值
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # 计算损失
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # 梯度裁剪
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 定期更新目标网络
        if self.step_count % 200 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path="dqn_model.pth"):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"模型已保存到 {path}")

    def load(self, path="dqn_model.pth"):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            print(f"模型已从 {path} 加载")
        else:
            print(f"找不到模型文件 {path}")