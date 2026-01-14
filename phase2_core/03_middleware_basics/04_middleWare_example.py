"""
LangChain Agent 中间件示例 - 数据脱敏功能实现

本文件演示了如何创建自定义中间件来处理敏感数据脱敏。
中间件是在AI模型调用前后执行的代码块，可用于预处理输入或后处理输出。
"""

import re  # 正则表达式库，用于匹配邮箱、手机号等敏感信息
from typing import Dict, Any  # 类型提示，帮助IDE提供更好的代码补全和错误检查

from langchain.agents import create_agent  # 创建智能体的主要函数
from langchain.agents.middleware import AgentMiddleware  # 基础中间件类
from langchain_community.agent_toolkits.load_tools import load_tools  # 加载预构建工具
from langchain_community.chat_models import ChatTongyi  # 通义千问聊天模型
from langgraph.checkpoint.memory import InMemorySaver  # 内存检查点存储器，用于保持对话状态


class DesedMiddleware(AgentMiddleware):
    """
    自定义数据脱敏中间件

    中间件的作用：
    - 在AI模型调用前对输入数据进行处理（如脱敏）
    - 在AI模型调用后对输出数据进行处理
    - 提供一种统一的方式来处理常见任务，而无需修改核心逻辑
    """

    def __init__(self, patterns: list = None):
        """
        初始化脱敏中间件

        Args:
            patterns: 自定义的脱敏模式列表，每个元素是一个元组(正则表达式, 替换文本)
        """
        super().__init__()  # 调用父类构造函数
        # 定义默认的脱敏规则：邮箱和手机号
        self.patterns = patterns or [
            # 邮箱正则：匹配类似 user@domain.com 的格式
            (r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[EMAIL]'),
            # 手机号正则：匹配11位手机号（可选+86前缀）
            (r'(\+86)?1[3-9]\d{9}', '[PHONE]')
        ]

    def _desensitize_text(self, text: str) -> str:
        """
        对文本进行脱敏处理的核心方法

        Args:
            text: 待处理的文本

        Returns:
            处理后的文本（敏感信息已被替换）
        """
        # 如果内容为空或已经包含脱敏标记，则跳过处理（避免重复处理）
        if not text or '[EMAIL]' in text or '[PHONE]' in text:
            return text

        # 快速预检查：只有当可能包含敏感信息时才继续处理（性能优化）
        if '@' not in text and not re.search(r'1[3-9]\d{9}', text):
            return text

        print(f"脱敏前: {text}")
        original_text = text  # 保存原始文本用于比较

        # 遍历所有脱敏规则并应用
        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text)  # 使用正则表达式替换敏感信息

        # 只有当内容发生变化时才打印日志（避免无意义的日志）
        if original_text != text:
            print(f"脱敏后: {text}")

        return text

    def before_model(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        模型调用前的处理方法

        这是中间件的关键部分，在AI模型收到请求之前执行。
        可以修改传给模型的数据，比如脱敏、格式化等。

        Args:
            state: 当前的状态字典，包含了会话的所有信息

        Returns:
            修改后的状态字典 中间件修改了 一定要返回
        """
        print("中间件DesensitizeDataMiddleware - before_model 被调用")

        # 检查状态中是否包含消息列表
        if 'messages' in state:
            messages = state['messages']  # 获取消息列表
            processed_any = False  # 标记是否进行了任何处理（用于控制日志输出）

            # 遍历所有消息
            for message in messages:
                # 检查消息是否有内容属性且内容是字符串类型
                if hasattr(message, 'content') and isinstance(message.content, str):
                    # 只处理非空内容且未被脱敏的内容（避免重复处理）
                    if (message.content and
                            '[EMAIL]' not in message.content and
                            '[PHONE]' not in message.content):

                        # 快速预检查：只有当可能包含敏感信息时才继续处理（性能优化）
                        if ('@' in message.content or
                                re.search(r'1[3-9]\d{9}', message.content)):

                            # 只有在真正需要处理时才打印日志
                            if not processed_any:
                                print("进行脱敏处理.....")

                            original_content = message.content  # 保存原始内容用于比较
                            message.content = self._desensitize_text(message.content)  # 执行脱敏

                            # 只有当内容发生变化时才打印详细日志
                            if original_content != message.content:
                                print(f"消息内容已从 '{original_content}' 修改为 '{message.content}'")
                                processed_any = True  # 标记已处理

            if processed_any:
                print("脱敏处理完成！")

        return state

    def after_model(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        模型调用后的处理方法

        这个方法在AI模型返回响应之后执行。
        可以用来处理模型的输出，比如解密、格式化等。

        Args:
            state: 模型处理后的状态字典

        Returns:
            修改后的状态字典
        """
        # 当前示例中不需要对输出进行特殊处理，直接返回原状态
        return state


# 创建一个聊天模型实例（这里使用通义千问模型作为示例）
model = ChatTongyi()

# 创建内存存储器，用于保存对话状态
# 这使得AI能够记住之前的对话历史，提供连续的对话体验
memory = InMemorySaver()

# 导入工具，load_tools支持的工具可以在load_tools.py中查看
# arxiv工具可以查询学术论文信息
tools = load_tools(["arxiv"])

# 系统提示词设计 - 定义AI助手的角色和行为
system_prompt = "你是一个专业的论文查询助手，使用arxiv工具为用户查询论文信息，回答需简洁准确，包含论文标题、作者、发表时间和核心摘要。"

# 创建带有中间件的智能体
# 这里将自定义的DesedMiddleware添加到中间件列表中
agent_with_middleware = create_agent(
    model=model,  # 使用的AI模型
    tools=tools,  # 可用的工具列表
    system_prompt=system_prompt,  # 系统角色提示
    checkpointer=memory,  # 启用状态保存
    middleware=[DesedMiddleware()]  # 添加自定义中间件
)

# 测试数据脱敏功能
print("测试: 电子邮件脱敏")
email_input = "我的邮箱是test.user@example.com，请帮我查询论文1605.08386"
print(f"输入内容: {email_input}")

# 调用智能体，传入用户消息
result1 = agent_with_middleware.invoke(
    {"messages": [{"role": "user", "content": email_input}]},  # 用户消息
    config={"configurable": {"thread_id": "middleware_test_1"}}  # 配置参数，thread_id用于区分不同对话线程
)

# 输出最终结果
print("结果:", result1["messages"][-1].content)