"""
摘要索引是RAG优化中的关键技术，它通过在索引阶段为每个文档片段生成语义摘要

# 核心价值与解决的问题：
# 1. 提升检索精度：摘要比原始文档更能表达核心语义，检索更精准
# 2. 降低计算成本：向量检索在摘要上进行，比全文检索更高效
# 3. 改善上下文质量：返回的是原始完整文档，但检索依据是高质量摘要
# 注意事项：
# 1.摘要质量至关重要
#  这是一个关于deepseek的介绍文章  太过于简单，丢失了关键信息
# 2.存储开销比较大
#  既要存储摘要的向量还要存储原始文档
# 3.一致性问题，维护一致性
#  通过uuid 将原始文档和摘要一一对应
"""

import os
import uuid  # 用于生成唯一标识符，确保文档-摘要映射的唯一性

# 导入LangChain核心组件
from langchain_classic.retrievers import MultiVectorRetriever  # 多向量检索器核心组件
from langchain_community.chat_models import ChatTongyi  # 通义千问大模型
from langchain_community.document_loaders import TextLoader  # 文本加载器，用于加载文档
from langchain_community.embeddings import DashScopeEmbeddings  # 阿里云向量嵌入模型
from langchain_community.vectorstores import Chroma  # Chroma向量数据库
from langchain_core.documents import Document  # 文档对象
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_core.runnables import RunnableMap  # 可运行的映射
from langchain_core.stores import InMemoryByteStore  # 内存字节存储
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归文本分割器

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME

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

# 4.文档预处理阶段 - 文本分割
# 使用递归字符分割器将长文档切分成适合处理的小块
# chunk_size=1024: 每个文档块最大1024个字符
# chunk_overlap=100: 相邻块之间重叠100个字符，保持语义连贯性
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# 5.摘要生成链构建 - 核心处理逻辑
# 构建一个处理管道：文档内容 → 提示词 → 大模型 → 摘要文本
chain = (
        {"doc": lambda x: x.page_content}  # 提取文档内容
        | ChatPromptTemplate.from_template("总结下面的文档:\n\n{doc}")  # 构造摘要生成提示词
        | model  # 调用大模型生成摘要
        | StrOutputParser()  # 解析模型输出为纯文本
)

print("准备生成文档摘要，请耐心等待...")
# 6.批量生成文档摘要 - 性能优化
# 使用batch方法并行处理多个文档，max_concurrency=5限制并发数避免资源耗尽
summaries = chain.batch(docs, {"max_concurrency": 5})

# 7.向量存储初始化 - 存储摘要向量
# Chroma是轻量级向量数据库，专门用于相似性检索
vectorstore = Chroma(
    collection_name="summaries",  # 指定集合名称
    embedding_function=llm_embeddings  # 使用前面定义的嵌入模型
)

# 8.原始文档存储初始化 - 存储完整文档内容
# 使用内存存储演示，生产环境可用Redis、数据库等持久化存储
store = InMemoryByteStore()

# 9.多向量检索器配置 - 整合向量检索和文档存储
"""
MultiVectorRetriever工作机制详解：
# retriever.invoke(query) 执行双阶段检索：
  阶段1：在向量数据库中检索与查询最相似的摘要向量
  阶段2：通过UUID关联找到对应的原始文档并返回
优势：
✅ 检索精度高：基于语义摘要而非原始文本
✅ 响应速度快：向量检索比全文检索效率更高  
✅ 上下文完整：返回原始完整文档而非摘要片段
✅ 易于维护：摘要和文档分离存储，便于独立更新
"""
id_key = "doc_id"  # 定义文档ID字段名
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,  # 向量存储：存放摘要向量
    byte_store=store,  # 字节存储：存放原始文档
    id_key=id_key,  # ID映射键：连接摘要和文档
)

# 10.文档ID生成 - 建立摘要与原始文档的映射关系
# 为每个文档片段生成全局唯一的UUID，确保一对一映射
doc_ids = [str(uuid.uuid4()) for _ in docs]

# 11.数据入库 - 关键步骤
# 将生成的摘要和原始文档分别存储到对应的存储系统中

# 步骤1：将文档摘要转换为LangChain Document格式并添加到向量数据库
# 这些摘要向量将用于后续的相似性检索
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})  # 摘要内容 + 对应的文档ID
    for i, s in enumerate(summaries)  # 遍历所有生成的摘要
]
# 这些摘要向量是后续检索的基础
retriever.vectorstore.add_documents(summary_docs)

# 步骤2：将原始文档存储到字节存储中
# 采用key-value形式存储，key为文档ID，value为原始文档对象
# list(zip(doc_ids, docs))：将ID和文档配对打包
retriever.docstore.mset(list(zip(doc_ids, docs)))

# 12.问答链构建 - 最终应用接口
# 构建完整的问答处理管道：查询 → 检索 → 生成答案
prompt = ChatPromptTemplate.from_template("根据下面的文档回答问题:\n\n{doc}\n\n问题: {question}")

# 生成问题回答链
# RunnableMap是RunnableParallel另一种写法，实现并行处理
# retriever.invoke将上面对摘要进行检索，但是通过关联ID获得原始文档，
# 最终返回原始文档的过程全部都包含完成了
chain = RunnableMap({
    "doc": lambda x: retriever.invoke(x["question"]),  # 检索相关文档
    "question": lambda x: x["question"]  # 保留原始问题
}) | prompt | model | StrOutputParser()

# 13.实际应用演示 - 企业事件查询
# 测试系统的实际检索和问答能力
query = "deepseek的企业事件"
answer = chain.invoke({"question": query})
print("-------------回答--------------")
print(answer)
