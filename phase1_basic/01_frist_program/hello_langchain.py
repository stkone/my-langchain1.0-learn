import os

from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from openai import OpenAI

from common_ai.ai_variable import *


# init_chat_model 不支持tongyi的模型 目前支持
# Supported model providers are: groq, azure_ai,
# huggingface, cohere, mistralai, azure_openai,
# together, xai, google_genai,
# bedrock_converse, anthropic, openai,
# deepseek, ollama, fireworks,
# google_anthropic_vertex,
# perplexity, ibm, bedrock,
# google_vertexai
def get_chain_example1():
    """
      使用 init_chat_model 方法创建语言模型实例并调用
      该方法通过指定模型名称、模型提供商、API密钥和基础URL来初始化模型
      然后向模型发送一个关于人工智能定义的请求并打印返回结果
      """
    model = init_chat_model(
        model=ALI_TONGYI_MAX_MODEL,
        model_provider=ALI_TONGYI,
        api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),  # API 密钥（可选，可从环境变量读取）
        base_url=ALI_TONGYI_URL,
    )
    response = model.invoke("你好！请用一句话介绍什么是人工智能。")
    print(f"\n返回对象类型: {type(response)}")
    print(f"返回对象: {response}")


def get_chain_example2():
    """
    使用 ChatOpenAI 方法创建语言模型实例并调用
    该方法通过指定模型名称、API密钥和基础URL来初始化模型
    然后向模型发送一个关于人工智能定义的请求并打印返回结果，包括内容详情
    """
    model = ChatOpenAI(
        model=ALI_TONGYI_MAX_MODEL,
        api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
        base_url=ALI_TONGYI_URL,
    )
    response = model.invoke("你好！请用一句话介绍什么是人工智能。")
    print(f"\n返回对象类型: {type(response)}")
    print(f"返回对象: {response}")
    print(f"返回对象的值: {response.content}")


def get_chain_example3():
    model = OpenAI(
                   api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
                   base_url=ALI_TONGYI_URL
                   )
    response = model.chat.completions.create(
        model=ALI_TONGYI_MAX_MODEL,
        messages=[{"role": "user", "content": "你好！请用一句话介绍什么是人工智能。"}]
    )
    print(f"\n返回对象类型: {type(response)}")
    print(f"返回对象: {response}")
    print(f"返回对象的值: {response.choices[0].message.content}")


def get_chain_example4():
    """
    使用 ChatTongyi 方法创建语言模型实例并调用
    该方法直接初始化通义千问模型，然后向模型发送一个关于人工智能定义的请求
    并打印返回结果，包括内容详情
    """
    model = ChatTongyi();
    response = model.invoke("你好！请用一句话介绍什么是人工智能。")
    print(f"\n返回对象类型: {type(response)}")
    print(f"返回对象: {response}")
    print(f"返回对象的值: {response.content}")


if __name__ == '__main__':
    # get_chain_example1()
    # get_chain_example2()
    get_chain_example3()
    # get_chain_example4()
