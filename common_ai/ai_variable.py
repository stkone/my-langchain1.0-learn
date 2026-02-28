#可用模型列表，以及获得访问模型的客户端
#实际使用时可以根据自己的实际情况调整
# 通义常用变量
import inspect
import os

from langchain_community.document_compressors import DashScopeRerank
from langchain_openai import OpenAI

ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen-max-latest"
ALI_TONGYI="tongyi"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qvq-max-latest"
ALI_TONGYI_EMBEDDING_MODEL = "text-embedding-v3"
ALI_TONGYI_RERANK_MODEL = "gte-rerank-v2"
ALI_TONGYI_EMBEDDING_V3 = "text-embedding-v3"
ALI_TONGYI_EMBEDDING_V4 = "text-embedding-v4"

ALI_TONGYI_RERANK_MODEL = "gte-rerank-v2"

ALI_TONGYI_API_KEY_SYSVAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen-max-latest"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qvq-max-latest"
ALI_TONGYI_EMBEDDING = "text-embedding-v3"
ALI_TONGYI_RERANK = "gte-rerank-v2"
DEEPSEEK_API_KEY_OS_VAR_NAME = "Deepseek_Key"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"


# 使用原生api获得指定平台的客户端 (默认是：阿里通义千问)
def get_normal_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
                      base_url=ALI_TONGYI_URL,
                      verbose=False, debug=False):
    """
    使用原生api获得指定平台的客户端，但未指定具体模型，缺省平台为阿里云百炼
    也可以通过传入api_key，base_url两个参数来覆盖默认值
    verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    """
    function_name = inspect.currentframe().f_code.co_name
    if (verbose):
        print(f"{function_name}-平台：{base_url}")
    if (debug):
        print(f"{function_name}-平台：{base_url},key：{api_key}")
    return OpenAI(api_key=api_key, base_url=base_url)

def get_ali_rerank(top_n=3):
    '''
    通过LangChain获得一个阿里重排序模型的实例
    :return: 阿里通义千问嵌入模型的实例
    '''
    return DashScopeRerank(
        model=ALI_TONGYI_RERANK_MODEL, dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
        top_n=top_n
)

