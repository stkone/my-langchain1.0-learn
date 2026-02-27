"""
Multi-Query 多路召回 RAG 实现详解
==================================

Multi-Query（多查询）是一种先进的RAG检索优化技术，通过生成多个查询变体来
提升检索的覆盖率和准确性。

核心原理：
========
1. 单一查询局限性：传统的向量检索可能因为表述差异而遗漏相关文档
2. 多视角覆盖：通过LLM生成同一问题的不同表达方式
3. 并行检索：对每个变体都执行独立的向量检索
4. 结果融合：合并去重所有检索结果，提供更全面的答案

工作步骤：
========
1. 查询理解：LLM分析原始用户查询
2. 变体生成：生成3-5个语义相同但表达不同的查询
3. 并行检索：每个查询变体独立执行向量检索
4. 结果聚合：合并所有检索结果并去除重复
5. 最终返回：提供更丰富的相关文档集合

优势：
====
- 提升召回率：从多个角度检索，减少遗漏
- 增强鲁棒性：对查询表述变化不敏感
- 改善质量：提供更多样化的相关信息
"""

import os

from langchain_classic.retrievers import MultiQueryRetriever  # 核心：多查询检索器
# 导入LangChain核心组件
from langchain_community.chat_models import ChatTongyi  # 通义千问大模型
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings  # 阿里云向量嵌入模型
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME

# ==================== 模型和嵌入配置 ====================
# 1. 初始化大语言模型 - 用于生成查询变体
# 这里的LLM负责将用户的一个查询扩展为多个相关查询
model = ChatTongyi(
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)

# 2. 初始化嵌入模型 - 用于将文档内容转换为向量表示
# 向量嵌入是实现语义相似度计算的基础
llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 使用最新的向量嵌入模型
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)  # 从环境变量获取API密钥
)

# ==================== 文档加载和预处理 ====================
# 加载DeepSeek相关文档
loader = TextLoader("../../Data/deepseek百度百科.txt", encoding="utf-8")
docs = loader.load()

# 创建文档分割器，并分割文档
# 将长文档切分为较小的片段，便于精确检索
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# ==================== 向量数据库初始化 ====================
# 创建向量数据库 - 存储文档片段的向量表示
# 这是传统RAG的核心组件，用于语义相似度检索
vectorstore = Chroma.from_documents(documents=splits, embedding=llm_embeddings)

# 创建基础检索器 - 传统的单查询检索器
# 这是标准的向量检索器，每次只处理一个查询
retriever = vectorstore.as_retriever()

# ==================== 基础检索测试 ====================
# 测试传统单查询检索的效果
print("=== 基础检索测试 ===")
relevant_docs = retriever.invoke('deepseek的应用场景')
print(f"传统检索器检索的文档数量: {len(relevant_docs)}")
print("检索到的相关文档:")
for i, doc in enumerate(relevant_docs[:2]):  # 显示前2个结果
    print(f"{i+1}. {doc.page_content[:100]}...")

# ==================== 传统RAG处理流程 ====================
print("\n=== 传统RAG处理流程 ===")
# 创建标准的RAG提示模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 构建传统RAG处理链
chain = RunnableMap({
    "context": lambda x: relevant_docs,  # 使用检索到的文档作为上下文
    "question": lambda x: x["question"]  # 传递原始问题
}) | prompt | model | StrOutputParser()

# 执行传统RAG回答
response = chain.invoke({"question": "deepseek的应用场景"})
print("传统RAG生成的回答:", response[:200] + "..." if len(response) > 200 else response)

# ==================== Multi-Query 多路召回优化 ====================
print("\n" + "="*50)
print("开始Multi-Query多路召回优化")
print("="*50)

# 启用日志记录，观察LLM生成的查询变体
import logging
logging.basicConfig()
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

"""
Multi-Query 核心实现原理：

第一步：查询变体生成
==================
原始查询："deepseek的应用场景"
↓ LLM分析和扩展
生成变体：
- "deepseek在哪些领域有实际应用？"
- "deepseek的主要使用场景是什么？"  
- "deepseek技术的具体应用场景有哪些？"

第二步：并行向量检索
==================
对每个查询变体都执行独立的向量检索：
变体1 → 检索结果A
变体2 → 检索结果B  
变体3 → 检索结果C

第三步：结果融合去重
==================
合并所有检索结果：A ∪ B ∪ C
去除重复文档，保留唯一结果

第四步：增强的上下文提供
====================
提供比单一查询更丰富、更多样化的文档集合
"""

# ==================== MultiQueryRetriever 核心实现 ====================
# MultiQueryRetriever是对查询的智能优化
# 它会在内部自动生成多个查询变体并执行检索
retrieval_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever,  # 基础检索器
    llm=model,            # 用于生成查询变体的大模型
    # 默认使用内置的提示词模板，生成3个查询变体
)

# 执行多路召回检索
print("\n=== Multi-Query检索过程 ===")
print("正在生成查询变体并执行多路检索...")
unique_docs = retrieval_from_llm.invoke({"question": 'deepseek的应用场景'})

print(f"\nMulti-Query检索到的独特文档数量: {len(unique_docs)}")
print("检索到的文档预览:")
for i, doc in enumerate(unique_docs[:3]):  # 显示前3个结果
    print(f"{i+1}. {doc.page_content[:150]}...")

# ==================== 效果对比分析 ====================
print("\n" + "="*50)
print("检索效果对比分析")
print("="*50)
print(f"传统单查询检索: {len(relevant_docs)} 个文档")
print(f"Multi-Query多路召回: {len(unique_docs)} 个文档")
print(f"提升比例: {((len(unique_docs) - len(relevant_docs)) / len(relevant_docs) * 100):.1f}%")

# ==================== Multi-Query 的核心优势 ====================
"""
Multi-Query 多路召回的核心优势：

1. 克服表述偏差：
   - 用户："deepseek的应用场景" 
   - 可能miss文档中写的"deepseek的使用领域"、"deepseek实践案例"等

2. 提升召回完整性：
   - 从单一角度 → 多角度覆盖
   - 减少因向量相似度计算局限导致的漏检

3. 增强检索鲁棒性：
   - 对同义词、近义词表达更加包容
   - 适应不同的查询习惯和表述方式

4. 保持检索精度：
   - 每个变体都是语义相关的有效查询
   - 通过向量检索保证相关性质量

适用场景：
=======
- 技术文档检索（术语表达多样）
- 知识问答系统（用户表述多变）
- 企业知识库（专业词汇丰富）
- 学术文献搜索（概念表达复杂）

注意事项：
========
- 会增加API调用次数（变体数量倍数）
- 检索时间略有增加
- 需要平衡变体数量和性能
"""

print("\n=== Multi-Query RAG 总结 ===")
print("✓ 通过LLM智能生成多个查询变体")
print("✓ 并行执行多路向量检索")  
print("✓ 融合去重提供更全面的检索结果")
print("✓ 显著提升召回率和检索质量")
print("✓ 是传统RAG的有效增强技术")




