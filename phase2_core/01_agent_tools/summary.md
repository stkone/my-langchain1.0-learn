# LangChain Agent 入门

## 一、基础概念

### 1. Agent 是什么？

Agent（智能代理）是LangChain框架中的一种高级组件，它能够：

- **动态决策**：根据用户输入智能选择执行路径
- **工具调用**：能够访问并使用各种工具扩展能力
- **自主思考**：执行"思考-行动-观察"的循环过程
- **保持上下文**：通过记忆组件维持对话状态

### 2. Agent 与 Chain 的区别


| 特性     | Chain              | Agent                        |
| -------- | ------------------ | ---------------------------- |
| 执行方式 | 预定义的固定流程   | 动态决策，灵活选择执行路径   |
| 工具使用 | 通常不涉及外部工具 | 可调用多种工具完成复杂任务   |
| 决策能力 | 无自主决策能力     | 根据上下文自主决定下一步行动 |
| 适用场景 | 简单、确定性任务   | 复杂、需要灵活应对的场景     |

## 二、代码结构分析

### 1. 基础Agent (01_agent.py)

```python
# 创建基础Agent
agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    checkpointer=memory
)
```

- **功能**：创建一个没有外部工具的基础Agent
- **用途**：适合简单对话场景，仅依赖LLM自身能力

### 2. 带工具的Agent (02_agent_with_tools.py)

```python
# 通过Tool类注册工具
tools = [
    Tool(name="get_current_time", func=get_current_time, description="获取当前时间"),
    Tool(name="recom_drink", func=recom_drink, description="推荐附近饮品店"),
    Tool(name="get_weather", func=get_weather, description="获取天气信息")
]

agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=tools,
    checkpointer=memory
)
```

- **功能**：创建一个能调用三种自定义工具的Agent
- **特点**：显式创建Tool对象，详细定义工具描述

### 3. 简洁工具注册 (03_agent_with_tools.py)

```python
# 通过@tool装饰器注册工具
@tool
def get_current_time(input: str = "") -> str:
    """返回当前时间"""
    ...

agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=[recom_drink, get_weather, get_current_time],
    checkpointer=memory
)
```

- **功能**：实现与02文件相同的功能
- **特点**：使用装饰器语法，代码更简洁

## 三、核心组件详解

### 1. 模型配置 (Model)

```python
model = ChatTongyi(
    temperature=0,  # 降低随机性，使输出更确定
    extra_body={"enable_search": False}  # 禁用内置搜索，避免与自定义工具冲突
)
```

- **temperature**: 控制输出随机性，0表示最确定
- **extra_body**: 传递特定于模型的参数

### 2. 系统提示 (System Prompt)

```python
system_prompt = "你是人工智能助手。需要帮助用户解决各种问题。"
```

- **作用**：定义Agent角色和行为准则
- **最佳实践**：清晰、具体，包含任务边界

### 3. 记忆组件 (Memory)

```python
memory = InMemorySaver()
```

- **功能**：保存对话历史，维护上下文
- **实现**：`InMemorySaver`将状态保存在内存中
- **会话隔离**：通过`thread_id`参数区分不同用户的对话

### 4. 工具定义 (Tools)

- **工具本质**：普通Python函数，被特殊包装后供Agent调用
- **输入/输出**：通常接受字符串输入，返回字符串结果
- **描述重要性**：工具描述决定Agent何时调用该工具

## 四、工具注册方式对比

### 1. Tool类方式（02文件）

```python
tools = [
    Tool(
        name="get_current_time",
        func=get_current_time,
        description="当你想知道现在的时间的时候，非常有用。"
    )
]
```

**优点**：

- 明确定义工具名称、函数和描述
- 描述可详细定制，提高Agent调用准确性
- 适合复杂场景，提供更多配置选项

### 2. @tool装饰器方式（03文件）

```python
@tool
def get_current_time(input: str = "") -> str:
    """返回当前时间"""
    ...
```

**优点**：

- 语法简洁，代码量少
- 函数定义和工具注册一体化
- 适合简单工具快速实现

**选择建议**：

- 简单场景或原型开发：使用@tool装饰器
- 复杂应用或需要精细控制：使用Tool类

## 五、Agent执行流程

Agent的工作流程遵循"思考-行动-观察"循环：

1. **接收输入**：用户请求被传递给Agent

   ```python
   response = agent.invoke(
       {"messages": [{"role": "user", "content": "推荐附近的饮品店"}]},
       config={"configurable": {"thread_id": "user_1"}}
   )
   ```
2. **分析决策**：LLM决定是否需要调用工具

   - 根据用户请求和工具描述判断
   - 例如："推荐附近的饮品店"会触发recom_drink工具
3. **工具调用**（如需要）：

   - 执行对应的Python函数
   - 获取工具返回的结果
4. **生成响应**：

   - 结合工具结果和原始请求生成最终回复
   - 将对话历史保存到记忆组件
5. **返回结果**：

   ```python
   if isinstance(response, dict) and 'messages' in response:
       response_content = response["messages"][-1].content
   ```

## 六、深入理解：Agent架构

### 1. 基于Pregel的执行模型

```python
# Agent本质是一个Pregel图计算模型
def invoke(self, input: InputT | Command | None, config: RunnableConfig | None = None, ...)
```

- **Pregel**：源自Google的大规模图处理系统
- **在LangGraph中**：将Agent执行过程建模为状态转换图
- **优势**：支持复杂的状态管理和条件分支

### 2. OpenAI消息格式

Agent使用标准化的消息格式：

```python
{
    "messages": [
        {"role": "user", "content": "你是谁"},
        {"role": "assistant", "content": "我是一个AI助手"}
    ]
}
```

- **角色类型**：system, user, assistant, function(工具调用)
- **标准化优势**：兼容各种LLM API，易于扩展

### 3. 内存检查点 (Checkpointer)

```python
checkpointer=memory
```

- **功能**：保存和恢复执行状态
- **实现**：`InMemorySaver`是最简单的实现
- **扩展**：可替换为数据库存储，支持持久化和大规模应用

## 七、学习路径建议

### 1. 入门阶段

- 理解Agent与Chain的基本区别
- 掌握基础Agent创建和调用
- 学习简单的工具注册方式（@tool装饰器）

### 2. 进阶阶段

- 深入理解Tool类的高级配置
- 学习自定义记忆组件
- 掌握多工具协同工作模式

### 3. 专家阶段

- 研究自定义Agent执行逻辑
- 实现自定义状态管理
- 优化工具调用策略和性能

### 4. 实践建议

- 从小型工具集开始，逐步扩展
- 详细记录工具调用日志，便于调试
- 为工具描述提供清晰、具体的指导
- 结合业务场景设计专用工具

通过这三个示例文件，你可以看到LangChain Agent从基础到实用的演进过程。掌握这些概念和实践，将为你构建强大的AI应用奠定坚实基础。

## 八、Function Call、Tools与MCP的技术演进与差异分析

### 1. Function Call（函数调用）

#### 1.1 定义与原理

Function Call是早期大语言模型与外部系统交互的机制，允许LLM直接调用预定义的函数。

- **工作原理**：LLM根据用户输入和上下文，预测应该调用哪个函数及相应的参数，并以结构化JSON格式返回函数名和参数
- **执行流程**：LLM → JSON输出 → 解析器解析 → 执行对应函数 → 结果返回LLM → 生成最终响应

#### 1.2 解决的核心问题

- **能力扩展**：突破LLM固有的知识限制，接入实时数据、计算能力等
- **精确控制**：确保LLM按照预设逻辑执行操作，而非自由生成
- **结构化交互**：提供一种标准化的方式让LLM表达意图

#### 1.3 技术局限

- **集成紧密**：每个函数都需要预先在系统中注册
- **缺乏灵活性**：难以动态调整可用函数列表
- **错误处理复杂**：函数调用失败时的处理机制有限

### 2. Tools（工具）

#### 2.1 定义与原理

Tools是Function Call的进一步抽象和标准化，提供了更丰富的元数据和更灵活的执行机制。

- **工作原理**：在Function Call基础上增加了工具描述、输入验证、异步执行等功能
- **元数据丰富**：不仅包含函数名和参数，还包括工具描述、输入Schema等

#### 2.2 解决的核心问题

- **语义理解增强**：通过详细的工具描述帮助LLM更好理解何时使用何种工具
- **类型安全**：通过Schema定义确保输入参数的正确性
- **可扩展性**：支持动态添加和移除工具
- **组合能力**：支持多个工具的串联和并行执行

#### 2.3 相较于Function Call的优势

- **更好的决策**：LLM可以根据工具描述做出更准确的调用决策
- **错误容错**：更完善的参数验证和错误处理机制
- **生态统一**：提供了统一的接口抽象，便于不同工具的集成

### 3. MCP（Model Context Protocol）

#### 3.1 定义与原理

MCP是最新提出的协议标准，旨在标准化模型与外部上下文源之间的通信，超越了传统的工具调用概念。

- **工作原理**：定义了一套标准化的API协议，允许模型通过统一接口访问各种外部服务和数据源
- **协议特性**：基于HTTP/HTTPS，采用JSON-RPC格式，支持双向通信

#### 3.2 解决的核心问题

- **标准化互操作**：消除了不同供应商之间的集成壁垒
- **上下文丰富化**：不仅限于工具调用，还包括数据检索、状态同步等多种交互模式
- **生态系统建设**：为第三方开发者提供了标准化的扩展接口
- **安全性增强**：内置认证、授权和审计机制

#### 3.3 相较于前两者的根本性改进

- **协议层面统一**：不再是特定框架的实现，而是跨平台的标准
- **双向通信**：支持外部系统主动向模型推送信息
- **服务发现**：自动发现和注册可用的服务端点
- **版本管理**：标准化的版本控制和兼容性管理

### 4. 技术演进路径与应用场景


| 层次 | 技术方案      | 主要解决的问题     | 适用场景       | 扩展性 |
| ---- | ------------- | ------------------ | -------------- | ------ |
| 1    | Function Call | 基础能力扩展       | 简单工具集成   | 低     |
| 2    | Tools         | 标准化和可维护性   | 复杂应用开发   | 中     |
| 3    | MCP           | 生态标准化和互操作 | 大规模生态系统 | 高     |

#### 4.1 选择原则

- **原型验证**：Function Call足以满足基本需求
- **生产应用**：Tools提供更好的工程实践
- **生态建设**：MCP是长期发展方向

#### 4.2 技术融合趋势

现代AI应用架构趋向于多层次融合：

- 底层仍基于Function Call的执行机制
- 中层通过Tools提供标准化抽象
- 上层利用MCP实现跨系统互操作

这种演进反映了AI系统从简单扩展到复杂生态的发展历程，每一层都在前一层的基础上解决了特
