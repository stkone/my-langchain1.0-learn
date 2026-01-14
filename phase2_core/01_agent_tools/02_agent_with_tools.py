"""
AI智能体工具示例程序
本程序演示了如何创建一个带有自定义工具的AI智能体，包括时间查询、天气查询和饮品店推荐功能。
"""

# 导入必要的库
import datetime  # 用于获取当前时间
import json      # 用于JSON数据处理（虽然本例中未直接使用，但通常在API交互中很有用）

# 从langchain库导入相关组件
from langchain.agents import create_agent  # 创建AI智能体的主要函数
from langchain_community.chat_models import ChatTongyi  # 使用通义千问作为聊天模型
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.tools import Tool  # 用于定义自定义工具
from langgraph.checkpoint.memory import InMemorySaver  # 内存存储器，用于保存会话状态

# 配置AI模型参数
model = ChatTongyi(
    extra_body={"enable_search": False}  # 禁用内置搜索功能，避免与我们自定义的工具冲突
)

# 系统提示词 - 定义AI助手的角色和行为
system_prompt = "你是人工智能助手。需要帮助用户解决各种问题。"

# 创建内存保存器 - 用于维护对话历史和上下文
memory = InMemorySaver()

# 定义工具函数 - 获取当前时间
def get_current_time(input: str = "") -> str:
    """
    获取当前系统时间的工具函数
    参数: input - 输入字符串（本函数忽略此参数）
    返回: 包含当前时间的字符串
    """
    print("get_current_time 函数被调用")  # 调试输出，显示函数被调用
    current_datetime = datetime.now()  # 获取当前日期和时间
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')  # 格式化时间为可读字符串
    result = f"当前时间：{formatted_time}。"  # 构建结果字符串
    print(result)  # 输出结果到控制台
    return result  # 返回结果给AI智能体

# 定义工具函数 - 获取天气信息
def get_weather(input: str = "") -> str:
    """
    模拟获取天气信息的工具函数
    参数: input - 输入字符串（本函数忽略此参数）
    返回: 包含天气信息的字符串
    注意：这是模拟数据，在实际应用中这里会连接真实的天气API
    """
    print("get_weather 函数被调用")  # 调试输出，显示函数被调用
    result = "天气信息：晴转多云，温度23℃，风级3级。"  # 模拟的天气信息
    return result  # 返回结果给AI智能体

# 定义工具函数 - 推荐饮品店
def recom_drink(input: str = "") -> str:
    """
    推荐附近饮品店的工具函数
    参数: input - 输入字符串（本函数忽略此参数）
    返回: 包含附近饮品店信息的字符串
    注意：这是模拟数据，在实际应用中这里会根据用户位置查询真实商家信息
    """
    print("recom_drink 函数被调用")  # 调试输出，显示函数被调用
    result = "距离您500米内有如下饮料店：\n\n1、蜜雪冰城\n2、茶颜悦色\n\n另外距离您200米内有惠民便利店，里面应该有矿泉水或其他饮品"
    return result  # 返回结果给AI智能体

# 定义可用工具列表 - 这些是AI智能体可以使用的工具
tools = [
    # 第一个工具：获取当前时间
    Tool(
        name="get_current_time",  # 工具名称（必须唯一），建议与函数名保持一致
        func=get_current_time,    # 对应的实际函数
        # 工具的描述，AI智能体会根据这个描述决定何时使用该工具
        description="当你想知道现在的时间的时候，非常有用。"
    ),
    # 第二个工具：推荐附近的饮品店
    Tool(
        name="recom_drink",       # 工具名称
        func=recom_drink,         # 对应的实际函数
        # 工具的详细描述，说明其用途和适用场景
        description="当你需要为用户推荐附近的饮品店时非常有用。这个工具会返回附近具体的饮品店名称和位置信息。仅当用户明确要求推荐饮品店或饮料店时使用此工具。"
    ),
    # 第三个工具：获取天气信息
    Tool(
        name="get_weather",       # 工具名称
        func=get_weather,         # 对应的实际函数
        # 简洁的工具描述
        description="获取天气信息。"
    )
]

# 创建AI智能体实例
agent = create_agent(
    model=model,                # 使用配置好的AI模型
    system_prompt=system_prompt, # 使用定义的系统提示词
    tools=tools               # 提供可用的工具列表
)

# 执行智能体 - 向AI发送请求并获取响应
response = agent.invoke(
    # 消息内容 - 用户的请求
    {"messages": [{"role": "user", "content": "推荐附近的饮品店"}]},
)

# 打印完整响应对象（调试用）
print(f"回复: {response}")

# 解析并打印最终响应内容
if isinstance(response, dict) and 'messages' in response:
    # 如果响应是包含消息的字典，则提取最后一条消息的内容
    response_content = response["messages"][-1].content
else:
    # 如果响应不是预期的字典格式，转换为字符串并输出警告
    response_content = str(response)
    print(f"警告: 响应格式异常: {response_content}")

# 最终输出AI的回复内容
print(response_content)
