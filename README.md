## 使用指南

这个代码框架实现了一个完整的"LLM作为强化学习策略教师"系统，并在CartPole-v1环境上测试。以下是使用步骤：

### 1. 安装依赖

```bash
pip install gymnasium torch numpy matplotlib tqdm openai
```

### 2.配置模型

conf/config.yaml
```bash
llm:
  models:
    - model_name: "qwen2.5-72b-instruct"
      description: "Primary model for general tasks"
      api_key: ""
      base_url: "https://qwen.aliyuncs.com/v1"
      max_tokens: 2048
      temperature: 0.7
      top_p: 0.9
```

### 3. 运行代码

基本运行（同时训练有LLM和无LLM的智能体进行对比）：

```bash
python llm_rl_teacher.py
```

只训练基础智能体（不使用LLM）：

```bash
python llm_rl_teacher.py --no-llm
```

其他可选参数：

- `--episodes 300`：设置训练轮次为300
- `--llm-weight 0.5`：增加LLM建议的权重
- `--save`：保存训练好的模型
- `--load`：从之前保存的模型继续训练

### 4. 代码结构说明

1. **LLMPolicyTeacher**：负责与LLM交互，提供策略建议
   - 支持状态描述格式化
   - 包含简单的响应缓存机制
   - 具有错误处理和备选方案
2. **DQNAgent**：基础强化学习智能体
   - 支持有/无LLM辅助两种模式
   - 提供可调节的LLM影响权重参数
   - 包含完整的DQN实现（经验回放、目标网络等）
3. **主程序**：包含训练、评估和可视化功能
   - 对比实验：同时训练两个智能体，比较LLM对学习效果的影响
   - 结果可视化：自动生成对比图
   - 支持模型保存和加载

### 5. 设计特性

- **适应性LLM查询**：每隔固定步数查询一次LLM，减少API调用次数
- **建议合并机制**：根据LLM给出的置信度动态调整其影响权重
- **稳健性**：具备错误处理机制，即使LLM查询失败也能继续训练
- **参数可配置**：通过命令行参数可灵活调整关键参数

这个框架可以作为研究"LLM作为策略教师"的起点，您可以根据需要扩展到其他环境或调整LLM集成方式。