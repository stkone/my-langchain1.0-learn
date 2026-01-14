"""
LangChain Agent 中间件装饰器示例

本文件演示了如何使用 LangChain 提供的中间件装饰器来拦截和修改代理执行过程。
与继承 AgentMiddleware 类的方式不同，这里使用函数装饰器来实现相同的功能。

重点介绍 wrap_model_call 包装器的工作原理。
"""

from typing import Any, Callable

# 导入必要的模块
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model, wrap_model_call, ModelRequest, \
    ModelResponse
from langchain_community.chat_models import ChatTongyi
from langgraph.runtime import Runtime

# 创建一个聊天模型实例（这里使用通义千问模型）
model = ChatTongyi()


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    使用 @before_model 装饰器的函数
    在模型调用之前自动执行
    
    参数:
        state: 当前代理状态，包含对话历史等信息
        runtime: 运行时上下文
    
    返回:
        可选的字典，会合并到状态中；返回 None 表示不修改状态
    """
    print(f"即将调用模型： {len(state['messages'])} 个消息")
    return None

@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    使用 @after_model 装饰器的函数
    在模型调用之后自动执行
    
    参数:
        state: 模型调用后的代理状态
        runtime: 运行时上下文
    """
    print(f"已经调用完模型： {len(state['messages'])} 个消息")
    return None

@wrap_model_call
def round_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    使用 @wrap_model_call 装饰器的函数 - 这是最强大的中间件类型
    
    wrap_model_call 与其他中间件装饰器的关键区别：
    1. 它包装了整个模型调用过程，可以完全控制调用流程
    2. 可以在调用前、调用中和调用后都进行干预
    3. 可以修改请求、响应，甚至完全替换模型调用逻辑
    4. 可以捕获异常、实现重试机制、添加缓存等
    
    参数说明：
        request (ModelRequest): 模型请求对象，包含要发送给模型的数据
        handler (Callable): 实际的模型调用处理器，必须调用它才能执行模型
    
    返回：
        ModelResponse: 模型响应对象
    
    与 before_model/after_model 的区别：
    - before_model/after_model: 分别在模型调用前后执行，但不能修改调用过程本身
    - wrap_model_call: 完全包装模型调用，可以在调用前后执行任意逻辑，并可修改请求/响应
    """
    print(f"模型调用前置处理 request： request={request}")
    print(f"模型调用前置处理 handler： handler={handler}")
    
    # 调用原始模型处理逻辑 - 这是核心步骤！
    # handler 函数实际上会执行模型调用并返回结果
    result = handler(request)
    
    print(f"模型调用后，模型返回结果： {result}")
    return result


# 创建带有自定义中间件的代理
agent = create_agent(
    model=model,  # 使用的模型
    tools=[],  # 工具列表（此示例中为空）
    system_prompt="你是一个 helpful 的助手。",  # 系统提示词
    # 按照列表顺序依次执行中间件
    middleware=[log_before_model, round_model, log_after_model]
)

# 调用代理并传递初始消息
result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
print(result)

"""
关于 wrap_model_call 的深入理解：

1. 执行顺序：
   - log_before_model (before_model 装饰器函数)
   - round_model 的前置部分 (wrap_model_call 装饰器函数中的第一部分)
   - handler(request) - 实际模型调用
   - round_model 的后置部分 (wrap_model_call 装饰器函数中的第二部分)
   - log_after_model (after_model 装饰器函数)

2. wrap_model_call 的强大之处：
   - 可以修改传入模型的请求 (request)
   - 可以修改模型返回的响应 (result)
   - 可以实现重试逻辑 (如果 handler 抛出异常)
   - 可以添加缓存机制 (跳过实际模型调用)
   - 可以记录详细的性能指标

3. 实际应用场景：
   - 缓存：如果请求已知，直接返回缓存结果而不调用模型
   - 限流：控制模型调用频率
   - 重试：在模型调用失败时自动重试
   - 转换：在请求发送前或响应返回后转换数据格式
   - 监控：详细记录每个模型调用的性能数据
"""


