def act(self, state):
    # RL策略建议
    rl_action, rl_probs = self.rl_agent.predict(state, deterministic=False)

    # LLM建议 (定期查询以节省API调用)
    if self.should_query_llm():
        llm_advice = self.llm_teacher.get_policy_advice(state, self.action_history, self.reward_history)
        llm_action = self.env.action_space.parse(llm_advice['recommended_action'])

        # 结合两种建议
        final_action = self._combine_actions(rl_action, llm_action, llm_advice['confidence'])
        return final_action
    else:
        return rl_action