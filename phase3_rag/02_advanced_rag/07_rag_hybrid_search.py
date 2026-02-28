"""
向量和分词器的混合搜索 (Hybrid Search)

【核心原理】
混合搜索结合了向量检索(语义相似度)和关键词检索(BM25)的优势：
1. 向量检索：基于语义理解，能捕捉查询与文档的深层含义关联
2. 关键词检索：基于词频统计，对精确匹配和罕见词更敏感
3. 混合策略：通过EnsembleRetriever融合两种检索结果，提升召回率和准确性

【适用场景】
- 文档库同时包含语义丰富内容和精确术语
- 单一检索方式无法满足多样化查询需求
- 需要平衡语义理解和精确匹配的场景
"""
import os

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME

# ==================== 1. 初始化LLM模型 ====================
# 【原理】ChatTongyi是阿里云通义千问的对话模型封装
# 它实现了LangChain的BaseChatModel接口，支持标准的消息输入输出
# 在RAG流程中负责：接收检索结果+问题，生成最终回答
model = ChatTongyi(
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
    # 参数说明：
    # - dashscope_api_key: 阿里云DashScope平台的API密钥
    # - model: 默认使用"qwen-turbo"，可通过参数指定其他模型如"qwen-max"
    # - temperature: 控制生成随机性，默认0.7，越低越确定性
    # - max_tokens: 最大生成token数
)

# ==================== 2. 初始化嵌入模型 ====================
# 【原理】向量嵌入(Embedding)将离散文本映射到连续高维向量空间
# - 语义相似的文本在向量空间中距离更近（余弦相似度衡量）
# - 这是实现语义检索的基础，比传统关键词匹配更能理解上下文含义
# - text-embedding-v3是阿里云最新版嵌入模型，支持多语言、长文本
llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 使用最新的向量嵌入模型
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)  # 从环境变量获取API密钥
    # 参数说明：
    # - model: 嵌入模型名称，text-embedding-v3支持8192 tokens输入
    # - dashscope_api_key: API认证密钥
)

# ==================== 3. 辅助函数定义 ====================
def pretty_print_docs(docs):
    """
    格式化打印文档列表，增强可读性

    【参数说明】
    - docs: List[Document] - 文档对象列表，每个Document包含page_content和metadata

    【原理】
    通过分隔线和编号将多个文档内容清晰展示，便于对比不同检索方式的结果差异
    """
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# ==================== 4. 文档加载与处理 ====================
# 【原理】RAG流程的第一步：将原始文档加载并切分为适合检索的文本块
# - 文档切分是为了控制上下文长度，避免超出LLM的token限制
# - 重叠(ch_overlap)确保语义连贯性，防止关键信息被截断

# 加载文档
loader = TextLoader("../Data/deepseek百度百科.txt", encoding="utf-8")
# 参数说明：
# - file_path: 文档路径，支持相对路径
# - encoding: 文件编码，中文文档通常使用"utf-8"
docs = loader.load()
# 返回：List[Document]，每个Document包含page_content(文本内容)和metadata(元数据)

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,    # 每个文本块的最大字符数
    chunk_overlap=50,  # 相邻块之间的重叠字符数，保持上下文连贯
    # 其他重要参数：
    # - separators: 分割符列表，默认["\n\n", "\n", " ", ""]
    # - length_function: 计算长度的函数，默认len
)
split_docs = text_splitter.split_documents(docs)
# 【原理】RecursiveCharacterTextSplitter采用递归分割策略：
# 1. 先尝试用第一个分隔符(如\n\n段落)分割
# 2. 若块仍太大，用下一个分隔符(如\n换行)继续分割
# 3. 直到块大小符合要求，保持文本语义完整性

# ==================== 5. 构建向量数据库 ====================
# 【原理】Chroma是开源向量数据库，负责存储文档向量并提供相似度检索
# - from_documents() 自动完成：文本→嵌入向量→存储的全流程
# - 内部使用HNSW等近似最近邻算法，实现高效的相似度搜索
vectorstore = Chroma.from_documents(
    documents=split_docs,      # 要存储的文档列表
    embedding=llm_embeddings   # 嵌入模型，用于将文本转为向量
    # 其他常用参数：
    # - collection_name: 集合名称，用于区分不同数据集
    # - persist_directory: 持久化目录，默认内存存储
)

# 定义查询问题
question = "相关评价"

# ==================== 6. 单独检索方式对比 ====================

# 6.1 向量检索 (语义相似度)
# 【原理】基于向量空间的余弦相似度，找到与查询语义最接近的文档
# - 优势：理解同义词、上下文含义，如"电脑"和"计算机"视为相似
# - 局限：对精确术语、罕见词匹配效果可能不佳
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# 参数说明：
# - search_kwargs: 检索配置字典
#   - k: 返回最相似的k个文档
#   - score_threshold: 相似度阈值过滤（可选）
#   - filter: 元数据过滤条件（可选）

doc_vector_retriever = vector_retriever.invoke(question)
# invoke() 是LangChain 1.0的标准调用方式，统一了各类组件的接口
print("-------------------向量检索-------------------------")
pretty_print_docs(doc_vector_retriever)

# 6.2 关键词检索 (BM25算法)
# 【原理】BM25是经典的关键词检索算法，基于词频-逆文档频率(TF-IDF)改进
# - 计算查询词与文档的相关性得分
# - 优势：对精确匹配、罕见术语敏感，可解释性强
# - 局限：无法理解语义，同义词被视为不同词
BM25_retriever = BM25Retriever.from_documents(split_docs)
# from_documents() 参数说明：
# - documents: 文档列表，构建倒排索引
# - k: 返回文档数量，可通过类属性或实例属性设置

BM25Retriever.k = 3  # 设置返回文档数量
doc_BM25Retriever = BM25_retriever.invoke(question)
print("-------------------BM25检索-------------------------")
pretty_print_docs(doc_BM25Retriever)

# ==================== 7. 混合检索 (核心功能) ====================
# 【核心原理】EnsembleRetriever - 检索结果融合算法
#
# 1. 多路召回：同时调用多个检索器，获取各自的候选文档集合
# 2. 分数归一化：将不同检索器的分数映射到统一尺度（Reciprocal Rank Fusion）
#    - RRF公式：score = Σ(1 / (k + rank))，k通常取60
#    - 相比线性归一化(x-min)/(max-min)，RRF对排名更鲁棒
# 3. 加权融合：根据weights参数加权各检索器的贡献
# 4. 重排序：按融合后的综合分数排序，返回Top-K结果
#
# 【优势】结合向量检索的语义理解能力和BM25的精确匹配能力，
# 在复杂查询场景下显著提升召回率和准确性

ensembleRetriever = EnsembleRetriever(
    retrievers=[BM25_retriever, vector_retriever],  # 检索器列表，顺序不重要
    weights=[0.5, 0.5]                              # 各检索器的权重，和为1.0
    # 参数说明：
    # - retrievers: List[BaseRetriever] - 要融合的检索器列表
    # - weights: List[float] - 对应检索器的权重，控制各自贡献比例
    # - id_key: str - 文档唯一标识字段，默认"id"，用于去重
)

retriever_doc = ensembleRetriever.invoke(question)
print("-------------------混合检索-------------------------")
print(retriever_doc)

# ==================== 8. 构建RAG Chain ====================
# 【原理】LangChain 1.0 的管道式链式调用（Pipe-based Composition）
# 使用 | 运算符将多个组件串联，数据从左向右流动：
# 输入 → 数据准备 → Prompt构建 → LLM调用 → 输出解析

# 8.1 创建Prompt模板
# 【原理】ChatPromptTemplate将模板字符串转换为可复用的Prompt构建器
# 支持变量插值 {variable}，运行时动态填充
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
# 其他创建方式：
# - from_messages(): 从消息列表创建，适合多轮对话
# - 支持 SystemMessage, HumanMessage, AIMessage 等消息类型

# 8.2 构建Chain 1: 混合检索 + LLM
# 【原理】RunnableMap是LangChain 1.0的核心组件，用于并行/映射数据转换
# - 接收输入字典，对每个key应用对应的转换函数
# - 输出新的字典，供后续组件使用
chain1 = RunnableMap({
    "context": lambda x: ensembleRetriever.invoke(x["question"]),  # 检索相关文档
    "question": lambda x: x["question"]                            # 透传问题
    # 【原理】lambda函数在这里作为Runnable的简写形式
    # 实际等价于：RunnableLambda(func=lambda x: ...)
}) | prompt | model | StrOutputParser()
# 管道流程详解：
# 1. RunnableMap: 输入{"question": "..."} → 输出{"context": [...], "question": "..."}
# 2. prompt: 将字典填充到模板，生成ChatPromptValue对象
# 3. model: 调用LLM，接收PromptValue，返回AIMessage
# 4. StrOutputParser(): 从AIMessage中提取文本内容

# 8.3 构建Chain 2: 纯向量检索 + LLM (对比用)
chain2 = RunnableMap({
    "context": lambda x: vector_retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | StrOutputParser()

# ==================== 9. 执行并对比结果 ====================
print("------------模型回复------------------------")

print("------------向量检索+BM25[0.5, 0.5]------------------------")
# 【原理】chain.invoke() 是标准调用入口，内部执行：
# 1. 输入验证和转换
# 2. 按管道顺序调用各组件
# 3. 自动处理中间状态传递
# 4. 返回最终输出
print(chain1.invoke({"question": question}))

print("------------向量检索------------------------")
print(chain2.invoke({"question": question}))

# 【扩展知识】LangChain 1.0 还提供了其他调用方式：
# - chain.stream(): 流式输出，适合长文本生成
# - chain.batch(): 批量处理，提高吞吐量
# - chain.ainvoke(): 异步调用，非阻塞IO