# AI助手工具实现
# 实现带自定义工具的AI助手
# 注：此文件展示使用@tool装饰器注册工具的方式，与02文件中的Tool类方式不同

# 导入库
import datetime

# 导入langchain组件
from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool

# 模型配置
model = ChatTongyi(
    extra_body={"enable_search": False}
)

# 系统提示
system_prompt = "你是人工智能助手。需要帮助用户解决各种问题。"

# 方式二：使用@tool装饰器定义工具（这种方式更简洁）
# 与02文件中使用Tool类的方式相比，@tool装饰器提供了更简洁的语法
@tool
def get_current_time(input: str = "") -> str:
    """
    返回当前时间
    """
    print("get_current_time 函数被调用")
    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    result = f"当前时间：{formatted_time}。"
    print(result)
    return result


# 方式二：使用@tool装饰器定义工具
@tool
def get_weather(input: str = "") -> str:
    """
    返回模拟天气信息
    """
    print("get_weather 函数被调用")
    result = "天气信息：晴转多云，温度23℃，风级3级。"
    return result


# 方式二：使用@tool装饰器定义工具
@tool
def recom_drink(input: str = "") -> str:
    """
    返回附近饮品店信息
    """
    print("recom_drink 函数被调用")
    result = "距离您500米内有如下饮料店：\n\n1、蜜雪冰城\n2、茶颜悦色\n\n另外距离您200米内有惠民便利店，里面应该有矿泉水或其他饮品"
    return result


# 创建智能体 - 注册工具的方式
# 方式二：直接将函数对象添加到工具列表中（使用@tool装饰器后，函数本身就是工具对象）
# 与02文件中的方式相比：不需要显式创建Tool对象，语法更简洁
agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=[recom_drink, get_weather,get_current_time],  # 直接使用函数对象列表
)
# agent.bind_tools([recom_drink, get_weather,get_current_time]) 或者使用bind_tools方法
# 执行请求
response = agent.invoke(
    {"messages": [{"role": "user", "content": "推荐附近的饮品店"}]},
    config={"configurable": {"thread_id": "user_1"}}
)

# 输出结果
print(f"回复: {response}")

# 解析响应
if isinstance(response, dict) and 'messages' in response:
    response_content = response["messages"][-1].content
else:
    response_content = str(response)
    print(f"警告: 响应格式异常: {response_content}")

# 显示AI回复
print(response_content)
