# 导入必要的库
from langchain.agents import create_agent  # 创建AI智能体的主要函数
from langchain_community.chat_models import ChatTongyi  # 使用通义千问作为聊天模型
from langgraph.checkpoint.memory import InMemorySaver  # 内存存储器，用于保存会话状态

# 创建语言模型实例
model = ChatTongyi()


def example_1_no_memory():
    """
    示例 1：没有内存的 Agent
    在这种情况下，每次调用 agent.invoke() 都是完全独立的，
    Agent 不会记住之前的对话内容。
    """
    print("\n" + "=" * 70)
    print("示例 1：没有内存的 Agent")
    print("=" * 70)

    # 创建没有 checkpointer 的 Agent
    # 注意：没有内存的 Agent 每次调用都是独立的，无法记住之前的对话
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。"
    )

    print("\n第一轮对话：")
    response1 = agent.invoke({
        "messages": [{"role": "user", "content": "我叫张三"}]
    })
    print(f"Agent: {response1['messages'][-1].content}")

    print("\n第二轮对话：")
    # 在没有内存的情况下，Agent 不知道之前说过什么
    response2 = agent.invoke({
        "messages": [{"role": "user", "content": "我叫什么？"}]
    })
    print(f"Agent: {response2['messages'][-1].content}")


# ============================================================================
# 示例 2：使用 InMemorySaver 添加内存
# ============================================================================

def example_2_with_memory():
    """
    示例2：使用 InMemorySaver 添加短期内存
    
    关键概念：
    1. checkpointer=InMemorySaver() - 为 Agent 添加内存功能
    2. config={"configurable": {"thread_id": "xxx"}} - 为每个会话分配唯一ID
    
    关于 checkpointer 参数：
    - checkpointer 是 LangGraph 中用于持久化和恢复状态的组件
    - 它允许 Agent 记住之前的对话历史和中间步骤
    - 没有 checkpointer，每次调用都是无状态的
    
    关于 InMemorySaver：
    - InMemorySaver 是 LangGraph 提供的一个简单的内存检查点保存器
    - 它将对话历史存储在内存中，使用线程ID(thread_id)来区分不同的会话
    - 适用于开发和测试环境，但在生产环境中不推荐使用 一般开发环境使用SQLite 作为轻量级嵌入式数据库，非常适合 LangGraph 的状态持久化需求，尤其适合不需要高并发或大型数据存储的场景。SqliteSaver 的包装器，
    -生产使用在生产环境中，对于需要更高性能和可扩展性的场景，
    可以使用 PostgresSaver 或 RedisSaver 替代 SqliteSaver。
    """
    print("\n" + "=" * 70)
    print("示例 2：使用 InMemorySaver 添加内存")
    print("=" * 70)

    # 创建带内存的 Agent
    # checkpointer 参数告诉 Agent 如何保存和恢复对话历史
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        checkpointer=InMemorySaver()  # 添加内存管理，让 Agent 能够记住对话历史
    )

    # config 中指定 thread_id
    # thread_id 用于区分不同的用户会话，确保每个用户的对话历史相互隔离
    config = {"configurable": {"thread_id": "conversation_1"}}

    print("\n第一轮对话：")
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫张三"}]},
        config=config  # 传入 config，包含 thread_id 信息
    )
    print(f"Agent: {response1['messages'][-1].content}")

    print("\n第二轮对话（同一个 thread_id）：")
    # 由于使用了相同的 thread_id，Agent 可以访问之前的对话历史
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么？"}]},
        config=config  # 使用相同的 thread_id，因此 Agent 可以记住之前的信息
    )
    print(f"Agent: {response2['messages'][-1].content}")


# ============================================================================
# 示例 3：多个会话（不同 thread_id）
# ============================================================================

def example_3_multiple_threads():
    """
    示例3：管理多个独立的会话
    
    关键概念：不同的 thread_id = 不同的对话历史
    每个 thread_id 对应一个独立的对话历史，互不影响
    
    InMemorySaver 实现原理：
    - InMemorySaver 内部使用一个字典结构来存储会话数据
    - 键(key)通常是 (thread_id, checkpoint_id) 的组合
    - 值(value)包含对话历史、中间步骤等状态信息
    - 当调用 agent.invoke 时，会根据 thread_id 查找对应的对话历史
    - 执行完成后，更新对应 thread_id 的状态信息
    
    生产环境建议：
    1. InMemorySaver 仅适用于开发和测试环境
    2. 生产环境中应使用数据库支持的检查点保存器，如：
       - PostgresSaver: 基于 PostgreSQL 的持久化存储
       - RedisSaver: 基于 Redis 的高性能存储
       - DynamoDBSaver: 基于 AWS DynamoDB 的存储
    3. 生产环境还需要考虑：
       - 数据持久性和备份
       - 并发访问控制
       - 性能优化
       - 安全性（敏感对话数据保护）
       - 清理过期会话数据的机制
    """
    print("\n" + "=" * 70)
    print("示例 3：多个独立会话")
    print("=" * 70)

    # 创建带内存的 Agent
    # 同一个 Agent 实例可以管理多个独立的会话
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        checkpointer=InMemorySaver()  # 使用 InMemorySaver 管理多个会话
    )

    # 会话 1 - Alice 的对话
    config1 = {"configurable": {"thread_id": "user_alice"}}
    print("\n[会话 1 - Alice]")
    agent.invoke(
        {"messages": [{"role": "user", "content": "我叫 Alice"}]},
        config=config1  # 为 Alice 分配唯一的 thread_id
    )
    print("Alice: 我叫 Alice")

    # 会话 2 - Bob 的对话
    config2 = {"configurable": {"thread_id": "user_bob"}}
    print("\n[会话 2 - Bob]")
    agent.invoke(
        {"messages": [{"role": "user", "content": "我叫 Bob"}]},
        config=config2  # 为 Bob 分配另一个唯一的 thread_id
    )
    print("Bob: 我叫 Bob")

    # 回到会话 1 - 测试 Alice 的记忆
    print("\n[回到会话 1 - Alice]")
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么？"}]},
        config=config1  # 使用 Alice 的 thread_id，Agent 应该记得她的名字
    )
    print(f"Agent: {response1['messages'][-1].content}")

    # 回到会话 2 - 测试 Bob 的记忆
    print("\n[回到会话 2 - Bob]")
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么？"}]},
        config=config2  # 使用 Bob 的 thread_id，Agent 应该记得他的名字
    )
    print(f"Agent: {response2['messages'][-1].content}")


def example_5_inspect_memory():
    print("\n" + "="*70)
    print("示例 5：查看内存状态")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        checkpointer=InMemorySaver()
    )

    config = {"configurable": {"thread_id": "inspect_thread"}}

    # 进行几轮对话
    print("\n进行对话...")
    agent.invoke(
        {"messages": [{"role": "user", "content": "你好"}]},
        config=config
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢编程"}]},
        config=config
    )

    # 再次调用，查看返回的完整状态
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢什么？"}]},
        config=config
    )
    print("\n目前所有的消息：",response["messages"])
    print("\n对话历史中的消息数量:", len(response['messages']))

    # for msg in response['messages'][-3:]:
    #     msg_type = msg.__class__.__name__
    #     content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
    #     print(f"  {msg_type}: {content}")

if __name__ == "__main__":
    example_5_inspect_memory()

