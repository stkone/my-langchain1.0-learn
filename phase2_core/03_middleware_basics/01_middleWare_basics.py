"""
LangChain Agent 中间件示例

本文件演示了如何创建和使用 LangChain Agent 中间件。
中间件允许我们在代理执行过程中插入自定义逻辑，
如在模型调用前后执行特定操作。
"""

from typing import Any

# 导入必要的模块
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_community.chat_models import ChatTongyi
from langgraph.runtime import Runtime

# 创建一个聊天模型实例（这里使用通义千问模型）
model = ChatTongyi()

class MyMiddleware(AgentMiddleware):
    """
    自定义中间件类
    
    AgentMiddleware 是 LangChain 提供的一个基类，允许开发者在代理执行过程中
    注入自定义行为。主要通过重写 before_model 和 after_model 方法实现。
    
    使用场景：
    - 日志记录
    - 性能监控
    - 输入/输出验证
    - 安全检查
    - 数据预处理/后处理
    """
    
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        在模型被调用之前执行的钩子函数
        
        参数:
            state (AgentState): 当前代理的状态，包含消息历史等信息
            runtime (Runtime): 运行时上下文对象
            
        返回:
            dict[str, Any] | None: 可选地返回要合并到状态中的额外数据，或返回 None
        """
        # 打印 AgentState 信息
        print("=== AgentState 信息 ===")
        print(f"消息数量: {len(state['messages'])}")
        print(f"所有消息: {state['messages']}")
        print(f"状态键: {list(state.keys())}")
        
        # 打印 Runtime 信息
        print("\n=== Runtime 信息 ===")
        print(f"Runtime 类型: {type(runtime)}")
        print(f"Runtime 内容: {runtime}")
        
        print(f"\n即将调用模型，当前有 {len(state['messages'])} 个消息")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        在模型调用之后执行的钩子函数
        
        参数:
            state (AgentState): 模型调用后的代理状态
            runtime (Runtime): 运行时上下文对象
            
        返回:
            dict[str, Any] | None: 可选地返回要合并到状态中的额外数据，或返回 None
        """
        print(f"模型返回消息: {state['messages'][-1].content}")
        return None

# 创建带有自定义中间件的代理
agent = create_agent(
    model=model,  # 使用的模型
    tools=[],     # 工具列表（此示例中为空）
    system_prompt="你是一个 helpful 的助手。",  # 系统提示词
    middleware=[MyMiddleware()]  # 中间件列表，给出类的对象
)

# 调用代理并传递初始消息
result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
print(result)

"""
生产环境使用建议：

1. 错误处理：
   - 在中间件中添加适当的异常处理逻辑
   - 避免中间件中的错误影响整个代理执行流程

2. 性能考虑：
   - 中间件代码应该高效，避免长时间运行的操作
   - 对于耗时操作，考虑异步处理

3. 日志记录：
   - 使用专业的日志库（如 logging）而不是 print 语句
   - 添加适当的日志级别（DEBUG, INFO, WARNING, ERROR）

4. 配置管理：
   - 将中间件的行为配置化，便于在不同环境中调整
   - 使用环境变量控制中间件功能的开关

5. 监控和追踪：
   - 集成应用性能监控（APM）工具
   - 记录关键指标如响应时间、请求量等

6. 安全性：
   - 在 before_model 中验证输入数据的安全性
   - 在 after_model 中过滤敏感信息

7. 测试：
   - 为中间件编写单元测试
   - 确保中间件不会破坏代理的核心功能
"""

