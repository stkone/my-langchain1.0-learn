"""
Post-Retrieval后检索-上下文压缩 (Contextual Compression)

【核心原理】
上下文压缩是RAG流程中Post-Retrieval阶段的关键优化技术，位于检索器(Base Retriever)和LLM之间。
其核心思想是：基础检索器返回的原始文档往往包含大量噪声（无关段落、冗余内容），
直接送入LLM会导致：1) token浪费 2) 关键信息被淹没 3) LLM产生幻觉。

压缩器(BaseDocumentCompressor)通过智能过滤或提取，只保留与查询真正相关的内容，
实现"精准投喂"，显著提升RAG系统的回答质量和成本效率。

【架构位置】
用户查询 → 基础检索器(Base Retriever) → [上下文压缩器] → 精炼文档 → LLM → 回答

【压缩器分类】
1. LLM驱动型：LLMChainExtractor(提取)、LLMChainFilter(过滤) - 精度高、成本高
2. 嵌入驱动型：EmbeddingsFilter - 速度快、成本低、精度中等
3. 组合型：DocumentCompressorPipeline - 多阶段流水线，灵活可配置
"""
import os

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter

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
    通过分隔线和编号将多个文档内容清晰展示，便于对比压缩前后的效果差异
    """
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# ==================== 4. 文档加载与基础检索器构建 ====================
# 【原理】标准RAG文档处理流程：加载 → 切分 → 向量化 → 存储
# 这里使用较大的chunk_size(1024)故意制造"文档包含无关内容"的场景，
# 以展示上下文压缩的价值

documents = TextLoader("../Data/deepseek百度百科.txt", encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,   # 较大的块大小，单个块可能包含多个主题
    chunk_overlap=100  # 保持上下文连贯性
)
texts = text_splitter.split_documents(documents)

# 构建基础向量检索器
retriever = Chroma.from_documents(texts, llm_embeddings).as_retriever()

# 执行基础检索，观察原始结果
docs = retriever.invoke("deepseek的发展历程")
print("-------------------压缩前--------------------------")
pretty_print_docs(docs)


# ==================== 5. 第一种：LLMChainExtractor压缩 ====================
# 【深层原理】
# LLMChainExtractor是基于LLM的智能提取器，其内部工作机制：
#
# 1. Prompt工程：构建专门的提取Prompt，要求LLM从文档中识别并提取与查询相关的句子/段落
#    典型Prompt结构：
#    - 系统指令："你是一个文档分析助手，请从给定文档中提取与问题相关的原文片段"
#    - 输入：文档内容 + 用户问题
#    - 输出格式：要求LLM返回JSON或标记格式的相关片段列表
#
# 2. 逐文档处理：对检索器返回的每个文档单独调用LLM
#    - 输入：document.page_content + query
#    - LLM推理：理解问题意图 → 扫描文档 → 识别相关段落 → 提取原文
#    - 输出：提取的相关文本（保持原文，非摘要）
#
# 3. 结果重组：将提取的片段重新组装为新的Document对象
#    - 保留原始metadata（如source、page等）
#    - page_content替换为提取的精炼内容
#
# 【核心特点】
# - 粒度精细：可以提取文档中的部分段落，而非整篇丢弃
# - 语义理解：利用LLM的推理能力，理解隐含相关性（如"这家公司"指代DeepSeek）
# - 原文保留：提取的是原文片段，非生成式摘要，避免信息失真
#
# 【适用场景】
# - 文档块很长但仅部分相关
# - 需要精确控制送入LLM的上下文
# - 成本可接受（每个文档一次LLM调用）
#
# 【成本权衡】
# - 优点：压缩精度最高，能处理复杂语义关联
# - 缺点：N个文档 = N次LLM调用，延迟和成本较高

print("-------------------第一种：LLMChainExtractor压缩------------------")
compressor = LLMChainExtractor.from_llm(model)
# from_llm() 参数说明：
# - llm: BaseLanguageModel - 用于提取的LLM实例
# - llm_chain: 可选，自定义LLMChain覆盖默认提取逻辑

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,  # 压缩器实例
    base_retriever=retriever     # 基础检索器实例
    # 其他参数：
    # - search_kwargs: 传递给基础检索器的额外参数
)

compressed_docs = compression_retriever.invoke("deepseek的发展历程")
# 【原理】ContextualCompressionRetriever的调用流程：
# 1. 先调用base_retriever.invoke(query)获取原始文档列表
# 2. 对每个文档调用base_compressor.compress_documents(docs, query)
# 3. 返回压缩后的精炼文档列表

print("-------------------压缩后--------------------------")
pretty_print_docs(compressed_docs)


# ==================== 6. 第二种：LLMChainFilter压缩 ====================
# 【深层原理】
# LLMChainFilter是基于LLM的二元分类器，其工作机制与Extractor有本质区别：
#
# 1. 判定模式：对每个文档做"相关/不相关"的二元判断
#    典型Prompt："请判断以下文档是否与问题相关，只回答YES或NO"
#    - 输入：document.page_content + query
#    - LLM输出："YES"或"NO"（或概率分数）
#
# 2. 过滤策略：
#    - 相关(YES)：保留整篇文档，不做任何内容修改
#    - 不相关(NO)：整篇丢弃
#
# 3. 与Extractor的核心差异：
#    ┌─────────────────┬──────────────────┬──────────────────┐
#    │     特性        │ LLMChainExtractor │ LLMChainFilter   │
#    ├─────────────────┼──────────────────┼──────────────────┤
#    │ 处理粒度        │ 段落级提取        │ 文档级过滤        │
#    │ 输出内容        │ 提取的相关片段    │ 完整文档或空      │
#    │ LLM输出长度     │ 较长（提取内容）  │ 极短（YES/NO）    │
#    │ 适用场景        │ 长文档部分相关    │ 短文档整体判断    │
#    │ token消耗       │ 较高              │ 较低              │
#    └─────────────────┴──────────────────┴──────────────────┘
#
# 【内部实现细节】
# - 使用PydanticOutputParser或类似机制约束LLM输出为结构化格式
# - 支持设置confidence_threshold，低于阈值的视为不相关
# - 异步支持：afilter_documents()方法用于高并发场景
#
# 【适用场景】
# - 文档本身较短（<500 tokens），不值得精细提取
# - 需要快速过滤明显无关的文档
# - 对延迟敏感，希望减少LLM输出token数

print("-------------------第二种：LLMChainFilter压缩后--------------------------")
_filter = LLMChainFilter.from_llm(model)
# from_llm() 参数说明：
# - llm: BaseLanguageModel - 用于判定的LLM
# - prompt: 可选，自定义判定Prompt
# - get_input: 可选，自定义输入格式化函数

compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter,
    base_retriever=retriever
)
compressed_docs = compression_retriever.invoke("deepseek的发展历程")
pretty_print_docs(compressed_docs)

# ==================== 7. 第三种：EmbeddingsFilter压缩 ====================
# 【深层原理】
# EmbeddingsFilter是基于向量相似度的轻量级过滤器，完全绕过LLM调用：
#
# 1. 嵌入计算流程：
#    - 步骤1：计算查询(query)的嵌入向量：query_embedding = embed(query)
#    - 步骤2：对每个文档计算嵌入：doc_embedding = embed(doc.page_content)
#    - 步骤3：计算余弦相似度：similarity = cosine_similarity(query_embedding, doc_embedding)
#    - 步骤4：阈值过滤：if similarity >= threshold: keep else: discard
#
# 2. 数学原理：
#    余弦相似度公式：cos(θ) = (A·B) / (||A|| × ||B||)
#    - A·B：向量点积，衡量同向程度
#    - ||A||, ||B||：向量模长，归一化因子
#    - 结果范围：[-1, 1]，实际文本相似度通常[0, 1]
#
# 3. 与LLM-based压缩器的本质区别：
#    - LLM压缩器：基于语义理解和推理，能处理复杂逻辑关联
#    - EmbeddingsFilter：基于向量空间距离，纯数学计算，无"理解"过程
#
# 4. 性能特征：
#    - 延迟：嵌入计算通常<100ms，远低于LLM调用(1-3s)
#    - 成本：嵌入API价格约为LLM的1/100
#    - 批处理：可并行计算多个文档的嵌入，GPU加速友好
#
# 【适用场景】
# - 大规模文档集需要快速初筛
# - 成本敏感的生产环境
# - 作为Pipeline的第一道过滤，配合LLMExtractor精加工
#
# 【阈值调优指南】
# - 0.8+：非常严格，只保留高度相关文档，可能漏掉边缘相关内容
# - 0.6-0.7：平衡模式，推荐默认值
# - 0.4-0.5：宽松模式，保留更多文档，可能包含噪声

print("-------------------第三种：EmbeddingsFilter压缩后--------------------------")
# 对每个检索到的文档进行额外的 LLM 调用既昂贵又缓慢。
# EmbeddingsFilter 通过嵌入文档和查询并仅返回那些与查询具有足够相似嵌入的文档来提供更便宜且更快的选项
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline

embeddings_filter = EmbeddingsFilter(
    embeddings=llm_embeddings,      # 嵌入模型实例
    similarity_threshold=0.6        # 相似度阈值，关键调参点
    # 其他参数：
    # - k: 最多返回的文档数，None表示不限制
    # - similarity_fn: 自定义相似度函数，默认cosine_similarity
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=retriever
)

compressed_docs = compression_retriever.invoke("deepseek的发展历程")
pretty_print_docs(compressed_docs)

# ==================== 8. 第四种：DocumentCompressorPipeline组合压缩 ====================
# 【深层原理】
# DocumentCompressorPipeline实现了责任链模式(Chain of Responsibility)，
# 将多个压缩/转换器按顺序串联，形成多阶段处理流水线：
#
# 输入文档 → [Transformer 1] → [Transformer 2] → ... → [Transformer N] → 输出文档
#
# 【本例流水线详解】
# 阶段1: CharacterTextSplitter - 细粒度切分
#   - 作用：将大文档块进一步切分为300字符的小片段
#   - 原理：使用". "作为分隔符，按句子边界切分，保持语义完整
#   - 输出：文档数量增加，单个文档变短
#
# 阶段2: EmbeddingsRedundantFilter - 去重过滤
#   - 作用：基于嵌入相似度去除内容重复的文档
#   - 原理：
#     a) 计算所有文档对的余弦相似度矩阵（O(n²)复杂度）
#     b) 对相似度>threshold的文档对，标记其中一个为冗余
#     c) 保留相似度较低的文档，确保信息多样性
#   - 算法优化：使用聚类或近似最近邻(ANN)降低计算复杂度
#
# 阶段3: EmbeddingsFilter - 相关性过滤
#   - 作用：基于查询-文档相似度，只保留相关内容
#   - 原理：同第7节所述
#
# 【Pipeline设计模式优势】
# 1. 单一职责：每个transformer只负责一种转换
# 2. 可组合性：灵活搭配不同transformer应对不同场景
# 3. 可扩展性：自定义transformer只需实现BaseDocumentTransformer接口
#
# 【适用场景】
# - 对质量要求极高的生产环境
# - 复杂文档结构需要多阶段处理
# - 需要平衡压缩率、相关性、多样性多个目标

print("-------------------第四种：组合压缩后--------------------------")
# DocumentCompressorPipeline 轻松地按顺序组合多个压缩器

# 阶段1：细粒度切分器
# 【原理】使用CharacterTextSplitter而非RecursiveCharacterTextSplitter，
# 因为前者按固定分隔符切分，更适合已知结构的文本（如按句子）
splitter = CharacterTextSplitter(
    chunk_size=300,      # 目标块大小（字符数）
    chunk_overlap=0,     # 不重叠，避免重复
    separator=". "       # 以". "为分隔符，按句子边界切分
    # 其他参数：
    # - strip_whitespace: 是否去除首尾空白，默认True
)

# 阶段2：冗余过滤器
# 【原理】计算文档间相似度矩阵，使用贪心策略去除冗余：
# 1. 按文档长度排序（优先保留长文档）
# 2. 依次检查每篇文档与已保留文档的相似度
# 3. 若相似度<threshold，则保留；否则丢弃
redundant_filter = EmbeddingsRedundantFilter(
    embeddings=llm_embeddings
    # 参数说明：
# - embeddings: 用于计算文档嵌入的模型
    # - similarity_threshold: 冗余判定阈值，默认0.95（非常高，只去重几乎相同的）
    # - top_n: 最多保留的文档数，None表示不限制
)

# 阶段3：相关性过滤器
relevant_filter = EmbeddingsFilter(
    embeddings=llm_embeddings,
    similarity_threshold=0.6
)

# 组合流水线
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
    # 参数说明：
    # - transformers: List[BaseDocumentTransformer] - 转换器列表，按顺序执行
    # - pipeline_input_key: 可选，指定输入字典的key
)

# 构建压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=retriever
)

compressed_docs = compression_retriever.invoke("deepseek的发展历程")
pretty_print_docs(compressed_docs)

# ==================== 9. 四种压缩器对比总结 ====================
'''
┌──────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│       特性           │Extractor   │Filter      │Embeddings   │Pipeline     │
├──────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ 处理粒度             │ 段落级      │ 文档级      │ 文档级      │ 多阶段      │
│ 核心机制             │ LLM提取     │ LLM判定     │ 向量相似度  │ 流水线组合  │
│ 精度                 │ ★★★★★      │ ★★★★☆      │ ★★★☆☆      │ ★★★★★      │
│ 速度                 │ ★★☆☆☆      │ ★★★☆☆      │ ★★★★★      │ ★★★☆☆      │
│ 成本                 │ 高          │ 中          │ 低          │ 中低        │
│ 适用场景             │ 高精度需求  │ 快速过滤    │ 大规模初筛  │ 生产环境    │
└──────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
'''
