"""
RAG (Retrieval-Augmented Generation) 是 LangChain 的核心应用场景之一
一般分成3个大部分
1. 文档处理阶段——索引前阶段
    1. 获取文档
    2. 分割文档
    3. 向量化文档
    4. 存入向量数据库中
2. 文档搜索阶段——索引阶段
3. 大模型生成阶段——生成阶段
"""
import os
# 安装 pip install langchain_chroma
# 加载word文档 安装 pip install docx2txt
# 加载json文档 安装 pip install jq
# 加载pdf文档  安装 pip install pymupdf
# 加载HTML文档 安装 pip install unstructured
# 加载MD文档   安装 pip install markdown +  pip install unstructured
import langchain
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, Docx2txtLoader
from langchain_community.embeddings import DashScopeEmbeddings

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME

#---------------------------------以下为 第一部分：文档处理阶段——索引前阶段----------------------------------------------------#

# 1. 指定要加载的Word文档路径
loader = Docx2txtLoader("../../../Data/人事管理流程.docx")

# 2. 加载文档、转换格式化成document
documents = loader.load()


"""
RecursiveCharacterTextSplitter 是关键函数
==========================================
参数详解：
----------
1. chunk_size (int): 
   - 每个文本块的最大字符数
   - 默认值通常为 1000 字符
   - 建议根据下游模型的上下文窗口调整
   - 中文文本建议：300-800字符
   - 英文技术文档建议：800-1500字符

2. chunk_overlap (int):
   - 相邻文本块之间的重叠字符数
   - 用于保持上下文连贯性，防止语义断裂
   - 建议设置为 chunk_size 的 10-20%
   - 过小可能导致上下文丢失，过大造成冗余存储

3. separators (List[str]):
   - 分隔符优先级列表，按顺序尝试分割
   - 默认值：["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
   - 策略：优先在段落边界分割，其次是句子边界，最后是单词边界
   - 可根据不同文档类型自定义

4. length_function (Callable):
   - 自定义长度计算函数
   - 默认使用 len() 计算字符数
   - 可改为基于 token 的计算：lambda text: len(encoding.encode(text))

5. is_separator_regex (bool):
   - 分隔符是否为正则表达式
   - 默认 False，分隔符被视为普通字符串

使用场景：
----------
✅ 适用场景：
- 通用文档处理（PDF、Word、TXT等）
- RAG系统中的知识库构建
- 长文本预处理
- 需要保持语义完整性的文本分割

❌ 不适用场景：
- 结构化数据（JSON、XML等）
- 代码文件（建议使用专门的代码分割器）
- 日志文件（可考虑固定长度分割）

最佳实践：
----------
1. 中文文档：chunk_size=500-800, overlap=50-100
2. 英文技术文档：chunk_size=800-1200, overlap=80-150
3. 代码文件：chunk_size=1000-1500, overlap=100-200
4. 法律文档：较小的chunk_size以保持条款完整性
"""

# 3. 构建文档切割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 切块大小：500字符
    chunk_overlap=50,    # 切块重叠大小：50字符（约占10%）
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]  # 默认分隔符
)

# 4. 通过分割器获取document 返回一个document对象列表
split_documents = text_splitter.split_documents(documents)

# 5. 获取词的嵌入模型, 大模型分成三类：LLM，聊天模型，嵌入模型
llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)

# 6.将已经切割好的原始文档，以及经过向量模型的处理后的文档保存 一起保存到向量数据库Chroma之中，返回一个向量数据库对象
vector_store = Chroma.from_documents(documents=split_documents,embedding=llm_embeddings)

#---------------------------------以下为 第二部分：索引阶段-----------------------------------------------------------------#
"""
按相似度的分数进行排序，分数值越小，越相似（其实是L2距离）
从向量数据库检索，使用chroma原始API查询 bind(k=1)表示返回相似度最高的第一个默认是4个
1. similarity_search() 方法
--------------------------
作用：基础相似度搜索，只返回文档内容
参数：
- query (str): 查询文本
- k (int): 返回文档数量，默认4
- filter (dict): 元数据过滤条件
- **kwargs: 其他搜索参数

使用场景：
✅ 通用文档检索
✅ FAQ问答系统
✅ 简单的内容查找
❌ 需要质量评估的场景
2. similarity_search_with_score() 方法  
作用：带得分的相似性搜索，返回文档和相似度分数
参数：
- query (str): 查询文本
- k (int): 返回文档数量，默认4
- filter (dict): 元数据过滤条件
- **kwargs: 其他搜索参数

返回值：List[Tuple[Document, float]]
- Document: 匹配的文档对象
- float: 相似度分数（L2距离，越小越相似）
使用场景：
✅ 需要质量控制的检索
✅ 相似度阈值过滤
✅ 结果排序和筛选
✅ 性能评估和调试

"""
# 基础相似度搜索
basic_results = vector_store.similarity_search(
    query="员工晋升流程",
    k=3  # 返回3个最相似的文档
)
print(f"基础搜索返回 {len(basic_results)} 个文档")

# 带分数的相似度搜索
scored_results = vector_store.similarity_search_with_score(
    query="员工晋升流程",
    k=3
)
print(f"带分数搜索返回 {len(scored_results)} 个结果")
for doc, score in scored_results:
    print(f"文档: {doc.page_content[:50]}...")
    print(f"相似度分数: {score:.4f}")

# 带元数据过滤的搜索
filtered_results = vector_store.similarity_search(
    query="晋升",
    k=2,
    filter={"source": "人事管理流程.docx"}  # 根据元数据过滤
)

"""
as_retriever() 方法详解
=====================
作用：将向量数据库转换为标准检索器接口，提供统一的检索API
核心参数说明：
-------------
1. search_type (str): 搜索类型
   - "similarity" (默认): 标准相似性搜索，基于向量距离
   - "similarity_score_threshold": 带阈值的相似性搜索
   - "mmr": 最大边际相关性搜索，平衡相关性和多样性

2. search_kwargs (dict): 搜索参数字典
   - k: 返回文档数量，默认4
   - score_threshold: 相似度阈值（仅限similarity_score_threshold模式）
   - fetch_k: MMR算法初始检索文档数，默认20
   - lambda_mult: MMR多样性参数，0-1之间，越大越多样化
   - filter: 元数据过滤条件

使用场景对比：
-------------
✅ similarity (默认)
- 适用：通用相似性检索
- 场景：FAQ问答、文档检索
- 特点：简单直接，返回最相似的k个文档

✅ similarity_score_threshold  
- 适用：需要质量控制的场景
- 场景：精确问答、高精度检索
- 特点：只返回超过阈值的文档

✅ mmr (最大边际相关性)
- 适用：需要多样性的场景
- 场景：推荐系统、多角度回答
- 特点：平衡相关性和新颖性
"""
# 基础相似性检索（默认配置）
basic_retriever = vector_store.as_retriever()

# 带阈值的质量控制检索
threshold_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,              # 返回5个文档
        "score_threshold": 0.5  # 相似度阈值0.5
    }
)

# MMR多样性检索
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,              # 返回6个文档
        "fetch_k": 20,       # 初始检索20个文档
        "lambda_mult": 0.3   # 多样性权重0.3（较低多样性）
    }
)

# 带元数据过滤的检索
filtered_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"source": "人事管理流程.docx"}  # 根据元数据过滤
    }
)

# 实际使用示例
query = "员工晋升的具体流程是什么？"

# 使用不同检索器
basic_docs = basic_retriever.invoke(query)
threshold_docs = threshold_retriever.invoke(query)
mmr_docs = mmr_retriever.invoke(query)

print(f"基础检索返回文档数: {len(basic_docs)}")
print(f"阈值检索返回文档数: {len(threshold_docs)}")
print(f"MMR检索返回文档数: {len(mmr_docs)}")

#---------------------------------以下为 第三部分：大模型生成阶段-----------------------------------------------------------------#
message = """ 
    仅使用提供的上下文回答下面的问题：
    {question}
    上下文：
    {context}
"""
prompt_template = ChatPromptTemplate.from_messages([('human',message)])
# 定义这个链的时候，还不知道问题是什么，
# 用RunnablePassthrough允许我们将用户的具体问题在实际使用过程中进行动态传入
chain = {"question":RunnablePassthrough(),"context":basic_retriever} | prompt_template | ChatTongyi()
#用大模型生成答案
resp = chain.invoke("员工晋升的具体流程是什么？")
print(resp.content)