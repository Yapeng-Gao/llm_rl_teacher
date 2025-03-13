# LLM策略教师
import json
import random
import numpy as np
from openai import OpenAI


class LLMPolicyTeacher:
    def __init__(self, api_key, model="gpt-3.5-turbo", max_tokens=300, temperature=0.7):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.query_count = 0
        self.client = OpenAI(api_key=api_key)
        self.cache = {}  # 简单的缓存机制

    def format_cartpole_state(self, state):
        """将CartPole状态转化为文本描述"""
        cart_pos, cart_vel, pole_angle, pole_vel = state

        description = f"""
        环境: CartPole
        当前状态:
        - 小车位置: {cart_pos:.4f} (-2.4至+2.4，0为中心)
        - 小车速度: {cart_vel:.4f}
        - 杆子角度: {pole_angle:.4f} 弧度 (0为垂直向上)
        - 杆子角速度: {pole_vel:.4f}

        目标: 通过左右移动小车，保持杆子直立，防止倒下。
        动作选择: 0 (向左推) 或 1 (向右推)
        """
        return description

    def create_prompt(self, state, action_history=None, reward_history=None):
        """创建提示"""
        state_description = self.format_cartpole_state(state)

        history_text = ""
        if action_history and reward_history:
            history_text = "最近的动作和奖励:\n"
            for i in range(min(5, len(action_history))):
                history_text += f"- 动作: {'左' if action_history[i] == 0 else '右'}, 奖励: {reward_history[i]}\n"

        prompt = f"""你是一个强化学习智能体的策略顾问。以下是当前状态的描述:
        {state_description}

        {history_text}

        请分析当前状态并提供以下内容:
        1. 对当前状态的简短分析
        2. 建议的下一步动作: 0 (左) 或 1 (右)，并说明理由
        3. 置信度 (1-10)

        请以JSON格式返回，包含以下字段:
        {{
            "analysis": "你的状态分析",
            "action": 0或1,
            "reason": "建议此动作的理由",
            "confidence": 介于1-10之间的数字
        }}

        只返回这个JSON，不要包含其他文本。
        """
        return prompt

    def get_policy_advice(self, state, action_history=None, reward_history=None):
        """获取LLM对当前状态的策略建议"""
        # 将状态转换为可哈希的形式用于缓存
        state_key = tuple(np.round(state, 2))

        # 检查缓存
        if state_key in self.cache:
            return self.cache[state_key]

        # 创建提示
        prompt = self.create_prompt(state, action_history, reward_history)
        try:
            # 查询LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个强化学习专家，提供明确的策略建议。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            self.query_count += 1
            response_text = response.choices[0].message.content

            # 尝试解析JSON响应
            try:
                # 找到JSON部分
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > 0:
                    json_str = response_text[json_start:json_end]
                    advice = json.loads(json_str)
                else:
                    # 如果没有找到JSON格式，尝试从文本中提取建议
                    if "action: 0" in response_text.lower() or "action:0" in response_text.lower():
                        action = 0
                    elif "action: 1" in response_text.lower() or "action:1" in response_text.lower():
                        action = 1
                    else:
                        action = random.randint(0, 1)

                    advice = {
                        "analysis": "无法解析完整JSON",
                        "action": action,
                        "reason": "从文本中提取",
                        "confidence": 5
                    }
            except json.JSONDecodeError:
                # 如果JSON解析失败，提供默认值
                advice = {
                    "analysis": "JSON解析失败",
                    "action": random.randint(0, 1),
                    "reason": "解析错误",
                    "confidence": 5
                }

            # 确保所有必要的字段都存在
            required_fields = ["analysis", "action", "reason", "confidence"]
            for field in required_fields:
                if field not in advice:
                    if field == "action":
                        advice[field] = random.randint(0, 1)
                    elif field == "confidence":
                        advice[field] = 5
                    else:
                        advice[field] = "未提供"

            # 缓存结果
            self.cache[state_key] = advice
            return advice

        except Exception as e:
            print(f"LLM查询错误: {e}")
            # 如果出错，返回随机建议
            return {
                "analysis": f"LLM查询出错: {str(e)}",
                "action": random.randint(0, 1),
                "reason": "错误恢复",
                "confidence": 5
            }
