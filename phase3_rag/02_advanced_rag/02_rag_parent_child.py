
# 导入LangChain核心组件
import os

from langchain_classic.retrievers import MultiVectorRetriever, ParentDocumentRetriever  # 多向量检索器核心组件
from langchain_community.chat_models import ChatTongyi  # 通义千问大模型
from langchain_community.document_loaders import TextLoader  # 文本加载器，用于加载文档
from langchain_community.embeddings import DashScopeEmbeddings  # 阿里云向量嵌入模型
from langchain_community.vectorstores import Chroma  # Chroma向量数据库
from langchain_core.documents import Document  # 文档对象
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_core.runnables import RunnableMap  # 可运行的映射
from langchain_core.stores import  InMemoryStore  # 内存字节存储
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归文本分割器

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME

"""
父子文档检索(Parent-Child Document Retrieval)深度解析
===============================================

核心概念：
---------
父子文档检索是一种高级的检索增强生成(RAG)技术，通过建立文档的层次化结构来提升检索精度。

架构原理：
--------
1. 父文档(Parent Document)：完整的原始文档片段，存储在内存存储中
2. 子文档(Child Document)：父文档的较小切片，存储在向量数据库中用于相似度检索
3. 检索流程：查询→向量检索子文档→映射回父文档→返回完整上下文

关键技术优势：
------------
1. 检索精度提升：通过子文档的细粒度检索，提高相关性匹配
2. 上下文完整性：返回父文档完整内容，避免信息碎片化
3. 计算效率优化：向量数据库只存储小块文档，减少存储和计算开销

使用场景详解：
------------
1. 技术文档问答：如API文档、产品说明书等长文档的精确检索
2. 法律法规查询：法条条文的精准定位和完整展示
3. 学术论文分析：论文段落的精确检索和全文引用
4. 企业知识库：内部文档的智能检索和知识发现
5. 新闻资讯检索：长篇文章的关键信息提取和完整展示

ParentDocumentRetriever 返回机制详解：
----------------------------------
重要结论：ParentDocumentRetriever 返回的是父文档，不是子文档！

工作机制：
1. 用户查询进入：retriever.invoke("用户问题")
2. 向量检索：在Chroma向量数据库中搜索最相似的子文档块
3. 父文档映射：通过docstore找到子文档对应的父文档
4. 完整返回：返回父文档的全部内容作为检索结果

对比示例：
--------
直接向量检索(vectorstore.similarity_search)：返回子文档块内容
父子检索器(retriever.invoke)：返回完整的父文档内容

这样设计的好处：
1. 保证上下文完整性：避免因文档切片导致的信息缺失
2. 提升回答质量：大模型基于完整文档生成更准确的回答
3. 减少重复检索：同一父文档的多个子块只返回一次完整内容
"""

# 1.获得访问大模型客户端 - 用于生成文档摘要
# 这里的大模型负责理解文档内容并生成简洁准确的摘要
model = ChatTongyi()

# 2.获得嵌入模型 - 用于将文本转换为向量表示
# 向量嵌入是实现语义检索的基础，将摘要转换为数学向量
llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 使用最新的向量嵌入模型
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)  # 从环境变量获取API密钥
)

# 3.初始化文档加载器 - 加载待处理的原始文档
# 这里加载的是DeepSeek的百度百科介绍文档
loader = TextLoader("../Data/deepseek百度百科.txt", encoding="utf-8")
# 加载文档到内存中
docs = loader.load()

"""
文档预处理阶段说明：
-----------------
1. 文档加载：将原始文本文件加载为LangChain Document对象
2. 层次化分割：使用两个不同粒度的分割器创建父子关系
3. 索引构建：父文档存入内存，子文档存入向量数据库
"""

# 创建向量数据库对象
vectorstore = Chroma(
    collection_name="split_parents", embedding_function = llm_embeddings
)

# 创建内存存储对象 - 用于存储完整的父文档
store = InMemoryStore()

"""
分割策略配置详解：
---------------
chunk_size比例关系：父块大小必须是子块大小的整数倍
推荐比例：父块:子块 = 4:1 或 8:1
本例配置：1024:256 = 4:1 (合理的分割比例)

分割器选择说明：
RecursiveCharacterTextSplitter：递归字符分割器
- 优点：保持文本语义完整性，优先按段落、句子分割
- 适用：技术文档、新闻文章等结构化文本
- 参数含义：chunk_size=最大块大小，chunk_overlap=重叠大小
"""

# 子块是父块内容的子集 chunk_size  必须是倍数
#创建主文档分割器 - 生成较大的父文档块
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)

#创建子文档分割器 - 生成较小的子文档块用于向量检索
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)

"""
ParentDocumentRetriever 核心参数详解：
----------------------------------
vectorstore: 向量数据库实例，存储子文档块的向量表示
docstore: 文档存储实例，存储完整的父文档内容
child_splitter: 子文档分割器，决定检索粒度
parent_splitter: 父文档分割器，决定返回的上下文大小
search_kwargs: 检索参数设置

关键参数说明：
search_kwargs={"k": 1} 表示返回top-1最相似的子文档对应的所有父文档
注意：即使多个子文档属于同一父文档，也只返回一次完整父文档（自动去重）

检索去重机制：
当topK=2时，如果两个最相似的子文档(A,B)属于同一个父文档，
ParentDocumentRetriever会智能识别并只返回该父文档一次，避免重复
"""

# 创建父子文档检索器，帮我们通过检索子块，返回父文档块
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store, # 文档存储对象
    child_splitter=child_splitter, # 子文档分割器，子文档存储到向量数据库
    parent_splitter=parent_splitter,# 主文档分割器，主文档存储到内存中
    search_kwargs={"k": 1}  # topK = 1,相似度最高的子文档块
)
# topK = 2,相似度最高的子文档块（A,B） A,B属于同一个父， 父文档块被查询两次，不会去重

#添加文档集 - 执行文档处理和索引构建
retriever.add_documents(docs)
print(f"主文块的数量：{len(list(store.yield_keys()))}")

"""
索引构建过程详解：
---------------
1. 父文档分割：使用parent_splitter将原始文档分割成较大块
2. 父文档存储：将完整父文档存储到InMemoryStore中
3. 子文档生成：对每个父文档使用child_splitter进一步分割
4. 向量索引：将子文档转换为向量并存储到Chroma数据库
5. 关系映射：建立子文档到父文档的映射关系

存储结构：
InMemoryStore：存储 {doc_id: 完整父文档内容}
Chroma VectorStore：存储 {vector: 子文档内容, metadata: {doc_id}}
"""

#这里我们通过向量数据库的similarity_search方法搜索出来的是与用户问题相关的子文档块的内容，
sub_docs = vectorstore.similarity_search("deepseek的应用场景")
#但是使用检索器retriever.invoke的方法来对这个问题进行检索，它会返回该子文档块所属的主文档块的全部内容：
sub_docs = retriever.invoke("deepseek的应用场景")

"""
两种检索方式对比：
----------------
方法1: vectorstore.similarity_search()
- 直接在向量数据库中搜索
- 返回：最相似的子文档块内容（片段）
- 特点：检索速度快，但上下文不完整

方法2: retriever.invoke()  
- 使用父子文档检索器
- 返回：子文档所属的完整父文档内容
- 特点：上下文完整，回答质量更高

实际应用场景选择：
- 简单关键词匹配：可直接使用向量检索
- 复杂问答任务：推荐使用父子文档检索器
- 需要完整上下文：必须使用父子文档检索器
"""

#创建prompt模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

#由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

#创建chain
chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),  # 使用父子检索器获取完整上下文
    "question": lambda x: x["question"]
}) | prompt | model | StrOutputParser()

print("------------模型回复------------------------")

response = chain.invoke({"question": "deepseek的应用场景"})
print(response)

"""
最佳实践建议：
------------
1. 分割参数调优：
   - 技术文档：父块1024-2048字符，子块256-512字符
   - 新闻文章：父块512-1024字符，子块128-256字符
   - 学术论文：父块2048-4096字符，子块512-1024字符

2. 性能优化：
   - 合理设置topK值（通常1-3）
   - 考虑添加chunk_overlap避免关键信息被切割
   - 对于超长文档可考虑多层分割结构

3. 应用场景适配：
   - FAQ系统：适合较小的分割粒度
   - 知识问答：需要较大的上下文窗口
   - 文档摘要：平衡检索精度和上下文完整性

4. 监控指标：
   - 检索准确率：正确父文档的召回率
   - 响应时间：端到端的查询延迟
   - 存储效率：向量数据库和内存使用情况
"""