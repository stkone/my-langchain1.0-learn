



# LangChain 1.0 中间件系统指南

> 中间件是LangChain架构中的强大组件，它让你能够在AI模型调用前后注入自定义逻辑，实现数据预处理、后处理、监控、安全等关键功能。本指南基于四个示例文件，为你系统化讲解中间件从基础到进阶的完整知识体系。

## 一、中间件基础概念

### 1.1 什么是中间件？
中间件是在AI模型调用前后执行的代码块，用于：
- 预处理输入数据（如脱敏、格式化）
- 后处理输出结果（如过滤、转换）
- 监控与记录（如日志、性能指标）
- 流量控制（如限流、缓存）
- 错误处理与重试机制

### 1.2 LangChain中间件的两种实现方式

LangChain 1.0提供两种中间件实现范式：
- **类继承方式**：继承`AgentMiddleware`基类
- **装饰器方式**：使用`@before_model`、`@after_model`、`@wrap_model_call`装饰器

## 二、类继承方式实现中间件

### 2.1 基础结构
```python
from langchain.agents.middleware import AgentMiddleware

class MyMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """在模型调用前执行"""
        # 处理逻辑
        return None  # 或返回需要合并到state的字典
    
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """在模型调用后执行"""
        # 处理逻辑
        return None
```

### 2.2 关键参数详解
- **AgentState**：包含当前对话状态
  - `state['messages']`：消息历史
  - 可访问所有对话上下文
- **Runtime**：运行时上下文
  - 包含线程ID、配置信息
  - 用于跨请求状态管理

### 2.3 应用示例
```python
agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个 helpful 的助手。",
    middleware=[MyMiddleware()]  # 注册中间件
)
```

## 三、装饰器方式实现中间件

### 3.1 三种装饰器类型

```python
from langchain.agents.middleware import before_model, after_model, wrap_model_call

@before_model
def log_before(state, runtime):
    print("模型调用前执行")
    return None

@after_model
def log_after(state, runtime):
    print("模型调用后执行")
    return None

@wrap_model_call
def custom_wrapper(request, handler):
    print("包装整个模型调用过程")
    result = handler(request)  # 关键：调用原始处理函数
    print("模型已返回结果")
    return result
```

### 3.2 `wrap_model_call` 高级用法
这是最强大的中间件类型，可以：
- 完全控制模型调用流程
- 修改请求和响应内容
- 实现缓存机制（跳过实际调用）
- 添加重试逻辑
- 记录精确性能指标

### 3.3 执行顺序
当多个中间件组合使用时，执行顺序为：
1. 所有`before_model`装饰器函数
2. `wrap_model_call`装饰器的前置部分
3. 实际模型调用
4. `wrap_model_call`装饰器的后置部分
5. 所有`after_model`装饰器函数

## 四、系统内置中间件详解

### 4.1 SummarizationMiddleware（摘要中间件）
**功能**：当对话历史接近模型上下文限制时，自动进行摘要压缩
```python
SummarizationMiddleware(
    model=model,
    max_tokens_before_summary=80,  # 触发摘要的token阈值
    messages_to_keep=1,            # 摘要后保留的消息数量
    summary_prompt="请简洁摘要以下对话: {messages}"
)
```

### 4.2 其他内置中间件

| 中间件类型 | 功能 | 适用场景 |
|------------|------|----------|
| **LoggingMiddleware** | 记录所有模型调用和响应 | 调试、分析、审计 |
| **CacheMiddleware** | 缓存模型响应，避免重复调用 | 提高性能、降低API成本 |
| **RateLimitingMiddleware** | 控制API调用频率 | 防止超出服务配额 |
| **RetryMiddleware** | 网络错误时自动重试 | 提高系统可靠性 |
| **MetricsMiddleware** | 收集性能指标 | 监控、优化 |

## 五、实战案例：数据脱敏中间件

### 5.1 需求分析
- 自动识别并脱敏敏感信息（邮箱、手机号等）
- 避免重复处理
- 保留业务语义
- 高性能处理

### 5.2 核心实现
```python
class DesedMiddleware(AgentMiddleware):
    def __init__(self, patterns=None):
        super().__init__()
        # 定义脱敏规则：(正则表达式, 替换文本)
        self.patterns = patterns or [
            (r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[EMAIL]'),
            (r'(\+86)?1[3-9]\d{9}', '[PHONE]')
        ]
    
    def _desensitize_text(self, text: str) -> str:
        """核心脱敏逻辑"""
        # 预检查：只有可能包含敏感信息时才处理
        if '@' not in text and not re.search(r'1[3-9]\d{9}', text):
            return text
        
        # 应用所有脱敏规则
        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text)
        return text
    
    def before_model(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """在模型调用前对所有消息进行脱敏处理"""
        if 'messages' in state:
            # 遍历并处理每条消息
            for message in state['messages']:
                if hasattr(message, 'content') and isinstance(message.content, str):
                    message.content = self._desensitize_text(message.content)
        return state
```

### 5.3 性能优化技巧
1. **预检查**：先快速判断是否包含敏感字符，避免不必要的正则匹配
2. **避免重复处理**：检查内容是否已包含脱敏标记
3. **选择性处理**：只处理非空内容
4. **日志控制**：只有内容实际变化时才输出详细日志

## 六、生产环境最佳实践

### 6.1 错误处理
```python
def before_model(self, state, runtime):
    try:
        # 中间件逻辑
    except Exception as e:
        # 记录错误但不中断流程
        logger.error(f"中间件执行失败: {str(e)}")
        # 可选：返回原始状态或部分处理的状态
        return state
```

### 6.2 性能考虑
- 避免在中间件中执行长时间运行的操作
- 复杂处理考虑异步执行
- 设置超时机制
- 使用批量处理减少开销

### 6.3 安全实践
- 在`before_model`中验证输入数据
- 在`after_model`中过滤敏感信息
- 定期审核中间件代码
- 实现访问控制和权限检查

### 6.4 配置管理
```python
class ConfigurableMiddleware(AgentMiddleware):
    def __init__(self, config=None):
        self.config = config or {
            "enabled": True,
            "log_level": "INFO",
            "patterns": [...] 
        }
    
    def before_model(self, state, runtime):
        if not self.config.get("enabled", True):
            return state  # 中间件被禁用
        # 其余逻辑
```

## 七、常见问题与解决方案

### 7.1 状态管理问题
**问题**：中间件修改了状态但未返回，导致修改丢失
**解决方案**：
```python
def before_model(self, state, runtime):
    # 修改state
    state["modified"] = True
    return state  # 重要：必须返回修改后的状态
```

### 7.2 性能瓶颈
**问题**：复杂的中间件处理导致响应延迟
**解决方案**：
- 添加预检查条件
- 使用异步处理
- 实现缓存机制
- 限制处理范围

### 7.3 多中间件协作
**问题**：多个中间件的执行顺序和依赖关系
**解决方案**：
- 明确中间件执行顺序
- 使用中间件间通信机制
- 设计松耦合的中间件架构

## 八、总结

LangChain 1.0的中间件系统为开发者提供了强大而灵活的扩展机制，通过：
- **类继承**和**装饰器**两种实现方式
- **内置中间件**提供常用功能
- **自定义中间件**满足特定业务需求
- **生产级实践**确保系统稳定性和安全性

掌握中间件技术，你将能够构建更加健壮、安全、高效的AI应用。从基础的记录日志到复杂的业务逻辑处理，中间件都是连接AI模型与实际业务需求的桥梁。

> **下一步建议**：尝试修改示例代码，添加自己的中间件功能，逐步构建适合你业务场景的AI系统。记住：好的中间件设计应该像水电一样，无感知但不可或缺。