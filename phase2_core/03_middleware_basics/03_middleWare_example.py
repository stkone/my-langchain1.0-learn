"""
LangGraph 中间件使用示例
此文件演示了如何在LangGraph中使用中间件，特别是SummarizationMiddleware（摘要中间件）
中间件是在AI模型调用前后或包装模型调用执行的函数，用于实现日志记录、监控、缓存等功能
"""

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# ========================
# LangChain 系统内置中间件详解
# ========================

"""
LangChain 提供了多种预构建的中间件来处理常见任务：

1. SummarizationMiddleware（摘要中间件）：
   - 功能：当对话历史接近模型的上下文窗口限制时，自动对历史消息进行摘要
   - 适用场景：长时间运行的对话、多轮对话、需要压缩历史记录以节省token
   - 参数说明：
     * model: 用于生成摘要的模型
     * max_tokens_before_summary: 触发摘要前的最大token数
     * messages_to_keep: 摘要后保留的消息数量
     * summary_prompt: 用于摘要的提示词模板

2. LoggingMiddleware（日志中间件）：
   - 功能：记录所有模型调用和响应，便于调试和分析
   - 适用场景：开发调试、性能分析、问题排查

3. CacheMiddleware（缓存中间件）：
   - 功能：缓存模型响应，避免重复请求相同的输入
   - 适用场景：提高响应速度、减少API调用成本

4. RateLimitingMiddleware（限流中间件）：
   - 功能：控制API调用频率，防止超出服务提供商的速率限制
   - 适用场景：遵守API配额限制、保护后端服务

5. RetryMiddleware（重试中间件）：
   - 功能：在网络错误或其他临时故障时自动重试请求
   - 适用场景：提高系统可靠性、处理网络波动

6. MetricsMiddleware（指标中间件）：
   - 功能：收集各种性能指标如响应时间、成功率等
   - 适用场景：性能监控、系统优化
"""

# 创建一个聊天模型实例（这里使用通义千问模型）
model = ChatTongyi()

# 创建内存存储器，用于保存对话状态
memory = InMemorySaver()

# 创建带有自定义中间件的代理
agent = create_agent(
    model=model,
    tools=[],  # 工具列表，此处为空表示不使用任何工具
    system_prompt="你是一个 helpful 的助手。",  # 系统提示词，定义助手的行为
    checkpointer=InMemorySaver(),  # 状态检查点，用于恢复对话状态
    # 中间件列表，可以配置多个中间件，它们会按顺序执行
    middleware=[
        # SummarizationMiddleware 摘要中间件配置
        SummarizationMiddleware(
            model=model,  # 用于生成摘要的模型
            max_tokens_before_summary=80,  # 当历史消息达到80个token时触发摘要
            messages_to_keep=1,  # 摘要后保留最后1条消息，清理旧的历史记录
            summary_prompt="请将以下对话历史进行简洁的摘要，保留关键信息: {messages}"  # 用于摘要的提示词
        ),
    ],
    debug=True  # 启用调试模式，输出更多调试信息
)

# 模拟长对话场景，演示摘要中间件的工作原理
print("\n模拟长对话场景...")
print("说明：当对话历史达到80个token时，SummarizationMiddleware会自动触发摘要功能")
print("摘要会保留关键信息并清除较早的消息，从而节省token并保持上下文相关性")

# 定义一系列模拟对话消息
demo_messages = [
    "用户询问你是谁",
    "用户计算商品价格：数量10，单价25.5",
    "用户再次询问你能做什么？",
    "用户想要生成一个介绍湖南的文案，要求100字左右，包含三湘四水，人文历史",
    "用户继续询问更多GPU产品信息",
    "用户要求计算2*20"
]

# 循环处理每条消息，模拟多轮对话过程
for i, message in enumerate(demo_messages, 1):
    print(f"\n💬 第{i}轮对话: {message}")
    # 调用代理处理当前消息
    result = agent.invoke({
        "messages": [HumanMessage(content=message)]  # 将用户消息包装成HumanMessage对象
    },
        # 配置参数，其中thread_id用于区分不同用户的对话线程
        config={"configurable": {"thread_id": "testsummarizationMiddleware"}}
    )
    
    # 打印AI的回复
    print(f"🤖 AI回复: {result['messages'][-1].content}")

print("\n📝 总结：")
print("1. SummarizationMiddleware 在对话历史过长时自动进行摘要，有效管理上下文窗口")
print("2. 这对于长时间运行的对话系统特别有用，可防止超出模型的上下文长度限制")
print("3. 通过合理设置max_tokens_before_summary和messages_to_keep参数，可以平衡信息保留和token使用效率")
