"""
信息补全型对话Agent实现

【核心原理】
该文件实现了一个基于模板的信息补全对话系统，核心机制：
1. 意图识别：根据用户输入匹配预定义业务模板
2. 信息完整性检查：判断用户输入是否包含模板所需全部字段
3. 多轮对话：通过历史记录维护对话上下文，引导用户补充缺失信息
4. 结构化输出：使用JSON格式统一处理状态和内容

【技术要点】
- RunnableWithMessageHistory：自动管理对话历史注入和更新
- ChatMessageHistory：内存中的消息存储
- JsonOutputParser：结构化输出解析
"""

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
import json

# ==================== 1. 初始化LLM模型 ====================
# 【原理】ChatTongyi实现LangChain的BaseChatModel接口，提供标准对话能力
model = ChatTongyi()
# 参数说明：
# - model: 默认"qwen-turbo"，轻量级对话模型即可满足意图识别需求
# - temperature: 默认0.7，意图识别建议降低至0.3-0.5提高确定性

# ==================== 2. 定义业务模板 ====================
# 【原理】预定义业务模板作为信息收集的schema，明确每个业务场景需要收集的字段
# 这是结构化对话的基础，Agent根据模板判断信息完整性

templates = {
    "订机票": ["起点", "终点", "时间", "座位等级", "座位偏好"],
    "订酒店": ["城市", "入住日期", "退房日期", "房型", "人数"],
}
# 模板设计原则：
# - 字段名清晰无歧义
# - 必填字段优先排列
# - 避免字段间依赖（如"退房日期"应在"入住日期"之后）

# 用户输入示例
user_input = "我想订一张长沙去北京的机票"

# ==================== 3. 意图识别 ====================
# 【原理】使用LLM进行零样本分类，将用户输入映射到预定义的业务模板
# 这是多轮对话的第一步，确定后续需要收集哪些信息字段

intent_prompt = PromptTemplate(
    input_variables=["user_input", "templates"],
    template="根据用户输入 '{user_input}'，选择最合适的业务模板。可用模板如下：{templates}。请返回模板名称。"
)
# PromptTemplate参数说明：
# - input_variables: 模板中需要填充的变量列表
# - template: 提示模板字符串，使用{变量名}占位

# 构建意图识别链：Prompt → LLM → 输出
intent_chain = intent_prompt | model
# 【原理】管道语法(|)将组件串联，输出作为下一组件输入

# 执行意图识别
intent = intent_chain.invoke({
    "user_input": user_input,
    "templates": str(list(templates.keys()))
}).content
# invoke()参数说明：
# - 字典形式提供input_variables对应的值
# - .content提取AIMessage中的文本内容

print("意图：", intent)

# 获取对应模板字段列表
selected_template = templates.get(intent)
print("模板：", selected_template)

# ==================== 4. 构建信息完整性检查Prompt ====================
# 【原理】将用户输入与模板字段对比，判断信息是否完整
# 使用JSON结构化输出，便于程序解析处理

info_prompt = f"""
    请根据用户原始问题和模板，判断原始问题是否完善。如果问题缺乏需要的信息，请生成一个友好的请求，
    明确指出需要补充的信息。若问题完善后，返回包含所有信息的完整问题。只回答跟模板信息匹配的答案。其他问题不回复。

    ### 原始问题    
    {user_input}

    ### 模板
    {",".join(selected_template)}                                   

    ### 输出示例
    {{
        "isComplete": true,
        "content": "`完整问题`"
    }}
    {{
        "isComplete": false,
        "content": "`友好的引导用户补充需要的信息`"
    }}                                       
"""
# 【输出格式设计】
# - isComplete: bool类型，标记信息是否完整
# - content: 完整问题或引导补充的提示语

# ==================== 5. 初始化对话历史管理 ====================

# 创建内存中的消息历史存储
chat_history = ChatMessageHistory()
# 【原理】ChatMessageHistory提供内存中的消息列表存储
# 支持add_user_message()、add_ai_message()等方法

# 构建带历史记录的对话模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个信息补充助手，任务是分析用户问题是否完整。"),
    ("placeholder", "{history}"),  # 历史记录占位符，运行时自动填充
    ("human", "{input}"),
])
# ChatPromptTemplate.from_messages参数说明：
# - 元组列表，每个元组为(角色, 内容)
# - 角色：system/human/ai/placeholder
# - "placeholder"用于运行时动态填充，如历史记录

# 构建基础链
info_chain = prompt | model

# ==================== 6. 包装为带历史记录的Runnable ====================
# 【核心原理】RunnableWithMessageHistory自动处理历史记录的注入和更新
with_message_history = RunnableWithMessageHistory(
    info_chain,                           # 基础Runnable
    lambda session_id: chat_history,      # 历史记录获取函数
    input_messages_key="input",           # 输入字典中用户消息的key
    history_messages_key="history"        # Prompt中历史占位符的key
)
# 参数详解：
# - runnable: 需要包装的基础链
# - get_session_history: 函数，接收session_id返回ChatMessageHistory
# - input_messages_key: 输入中用户消息的字段名
# - history_messages_key: Prompt模板中历史占位符的变量名
# 【工作机制】
# 1. 调用时从历史存储获取已有消息
# 2. 将历史消息填充到Prompt的{history}占位符
# 3. 执行链得到响应
# 4. 自动将用户输入和AI响应添加到历史存储

# ==================== 7. 首次调用：判断信息完整性 ====================
# 【原理】首次调用携带完整Prompt，让LLM判断用户初始输入是否满足模板要求

info_request = with_message_history.invoke(
    input={"input": info_prompt},
    config={"configurable": {"session_id": "unused"}}
).content
# invoke参数说明：
# - input: 输入字典，对应input_messages_key指定的字段
# - config: 配置字典，configurable.session_id用于区分不同对话会话

# 解析JSON响应
parser = JsonOutputParser()
json_data = parser.parse(info_request)
# JsonOutputParser原理：
# - 自动从LLM输出中提取JSON格式内容
# - 处理常见的JSON格式问题（如多余的markdown标记）
# - 返回Python字典对象

print("json_data：", json_data)

# ==================== 8. 多轮对话循环：引导用户补充信息 ====================
# 【核心原理】当isComplete为false时，循环引导用户补充缺失字段
# 每轮循环：显示引导语 → 用户输入 → AI判断 → 更新状态

while json_data.get('isComplete', False) is False:
    try:
        # 显示AI生成的引导信息（黄色加粗）
        # \033[1;33m：ANSI转义序列，设置字体为黄色加粗
        # \033[0m：重置字体样式
        user_answer = input(f"\033[1;33m{json_data['content']}\033[0m\n请补充：")

        # 提交用户补充信息给AI处理
        # 【关键】with_message_history自动将本轮对话加入历史
        info_request = with_message_history.invoke(
            input={"input": user_answer},
            config={"configurable": {"session_id": "unused"}}
        ).content

        # 解析AI响应，更新完整性状态
        json_data = parser.parse(info_request)

    except json.JSONDecodeError:
        # 红色加粗显示错误信息
        print("\033[1;31m[错误] AI返回了无效的JSON格式，请重试\033[0m")
        continue
    except KeyError:
        print("\033[1;31m[错误] 响应格式异常，正在终止流程\033[0m")
        break

# 【循环退出条件】
# - isComplete变为true：信息已完整
# - KeyError异常：响应格式异常，终止流程

# ==================== 9. 输出最终结果 ====================
# 绿色加粗显示最终完整查询
print(f"\033[1;32m[最终查询] {info_request}\033[0m")

# ==================== 10. 流程总结 ====================
'''
【完整执行流程】

步骤1: 用户输入 → 意图识别 → 匹配业务模板
步骤2: 构建信息完整性检查Prompt（含模板字段）
步骤3: 首次调用LLM判断信息完整性
步骤4: 如果isComplete=false，进入多轮对话循环
        循环内：
        - 显示引导语（content字段）
        - 获取用户补充输入
        - 调用LLM（自动携带历史上下文）
        - 解析响应，更新isComplete状态
步骤5: isComplete=true时退出循环，输出最终结果

【关键技术点】
1. RunnableWithMessageHistory：自动历史管理，无需手动维护上下文
2. JsonOutputParser：结构化输出，便于程序状态机控制
3. 模板驱动：业务模板决定需要收集的信息字段
4. 多轮对话：通过while循环实现交互式信息补全
'''