"""
LangChain Agent 示例
本文件演示了如何创建和使用LangChain Agent
Agent与Chain的主要区别：
- Chain: 是一系列预定义步骤的顺序执行，按照固定流程处理输入并产生输出
- Agent: 能够根据当前情况智能地选择要执行的步骤，具有决策能力，可以根据输入动态选择工具和行动
"""

from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import InMemorySaver

# 1. 初始化模型实例
# 使用通义千问聊天模型作为Agent的基础模型
model = ChatTongyi()

# 2. 定义系统提示
# 系统提示用于告诉Agent它的角色和任务
system_prompt = "你是人工智能助手。需要帮助用户解决各种问题。"

# 4. 创建Agent
# create_agent函数创建一个具有记忆功能的智能Agent
# 参数说明：
# - model: 用于处理输入和生成响应的基础语言模型
# - system_prompt: 定义Agent角色和行为的系统提示
# - checkpointer: 记忆组件，用于保存和恢复对话状态
agent = create_agent(
    model=model,  # 聊天模型
    system_prompt=system_prompt,
)

# 调用Agent处理用户请求
# invoke方法执行Agent，传入消息和配置
# 消息格式为：{"role": "角色", "content": "消息内容"}
# config中的thread_id用于区分不同用户的会话，确保对话状态隔离
response = agent.invoke(
    {"messages": [{"role": "user", "content": "你是谁"}]}
)
# 参数传递说明：
# 1. 第一个参数 {"messages": [{"role": "user", "content": "你是谁"}]} 作为input参数传递给invoke方法
#    - 这个参数包含了用户的消息，格式遵循OpenAI的消息格式
#    - "messages"键包含一个消息数组，每条消息都有角色(role)和内容(content)
# 2. config参数作为RunnableConfig类型传递给invoke方法的config参数
#    - config参数用于传递配置信息，如会话ID、回调函数、标签等
#    - 这里的thread_id用于标识不同用户的会话，确保不同用户的对话状态相互隔离
# 3. invoke方法在langgraph.pregel.main.Pregel类中定义，接收这些参数并处理：
#    def invoke(self, input: InputT | Command | None, config: RunnableConfig | None = None, ...)
#    - input参数接收消息数据
#    - config参数接收配置信息
#    - Pregel是基于论文《Pregel: A System for Large-Scale Graph Processing》的图计算模型，在langgraph中用于管理复杂的计算流程

# Agent的工作原理详解：
# 1. 决策机制：Agent内部使用LLM作为推理引擎，根据用户输入决定采取什么行动
# 2. 工具使用：Agent可以访问预先定义的工具(tool)，当需要获取额外信息或执行特定任务时调用这些工具
# 3. 思考过程：Agent通常会执行"思考-行动-观察"循环(thought-action-observation loop)
#    - 思考(thought)：分析用户请求，决定下一步行动
#    - 行动(action)：执行工具调用或直接生成响应
#    - 观察(observation)：观察行动结果，决定是否需要进一步行动或结束
# 4. 记忆管理：通过checkpointer(记忆组件)保存和恢复对话状态，使得Agent能够在多轮对话中保持上下文
# 5. 执行流程：
#    - 接收用户输入
#    - LLM分析输入并决定是否需要使用工具
#    - 如果需要使用工具，则调用相应工具并获取结果
#    - 根据工具结果和原始输入生成最终响应
#    - 将交互历史保存到记忆组件中

# OpenAI消息格式详解：
# OpenAI的消息格式是聊天模型API的标准输入格式，包含以下几种角色：
# - "system": 系统消息，用于设置助手的行为、个性或上下文
#   示例: {"role": "system", "content": "你是一个有用的人工智能助手"}
# - "user": 用户消息，表示用户的输入或问题
#   示例: {"role": "user", "content": "今天天气怎么样？"}
# - "assistant": 助手消息，表示AI的回复
#   示例: {"role": "assistant", "content": "抱歉，我无法获取实时天气信息"}
# - "function": 函数消息，表示工具调用的结果（在某些API版本中使用）
#   示例: {"role": "function", "name": "get_weather", "content": "{\"temp\": 25, \"condition\": \"sunny\"}"}
# 
# 消息数组的典型结构：
# [
#   {"role": "system", "content": "你是一个数学老师"},
#   {"role": "user", "content": "什么是勾股定理？"},
#   {"role": "assistant", "content": "勾股定理是指在直角三角形中..."},
#   {"role": "user", "content": "能给我举个例子吗？"},
#   {"role": "assistant", "content": "当然，比如一个三角形..."}
# ]
# 这种格式允许模型理解对话历史和上下文，从而生成更相关的回复。

print(f"LLM输出：{response}")
