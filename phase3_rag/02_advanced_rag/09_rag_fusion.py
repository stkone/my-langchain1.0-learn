"""
RAG-Fusion: 检索增强生成融合技术

【内容概要】
该文件实现了RAG-Fusion技术，通过查询扩展(Query Expansion)和互逆排序融合(RRF)算法，
将单一用户查询扩展为多个语义等价的子查询，并行检索后融合结果，显著提升召回率和排序质量。

【核心流程】
用户查询 → LLM生成多角度查询(4个) → 并行向量检索(4路) → RRF融合重排序 → 输出最相关文档

【实现原理】

1. 查询扩展(Query Expansion)
   【原理】利用LLM的语义理解能力，将用户的单一查询改写为多个不同表述但语义等价的查询。
   这基于"词汇鸿沟"(Vocabulary Mismatch)问题的解决思路：同一概念可以有多种表达方式，
   不同表述可能召回不同的相关文档。
   
   例如："人工智能的应用" → 
   - "AI在医疗诊断中的应用"
   - "机器学习如何改变金融行业"  
   - "人工智能在制造业的实践案例"
   - "AI技术对未来就业的影响"

2. 并行检索(Parallel Retrieval)
   【原理】使用LangChain的Runnable.map()机制，对每个生成的查询并行执行向量检索。
   这种并行化设计充分利用了向量化计算的批处理能力，4个查询可同时发送给向量数据库，
   相比串行执行显著降低总体延迟。

3. 互逆排序融合 RRF (Reciprocal Rank Fusion)
   【核心公式】RRF_score(d) = Σ(i=1 to n) [ 1 / (k + r_i(d)) ]
   
   参数说明：
   - d: 某个特定文档
   - n: 检索结果列表的数量（如4个查询产生4个结果列表）
   - r_i(d): 文档d在第i个结果列表中的排名（从0开始计数）
   - k: 平滑常数（默认60），防止分母过小导致分数差异过大
   
   【算法机制深度解析】
   
   (1) 排名倒数设计原理：
       - 排名0的文档贡献：1/(60+0) ≈ 0.0167
       - 排名10的文档贡献：1/(60+10) ≈ 0.0143
       - 排名100的文档贡献：1/(60+100) ≈ 0.00625
       这种非线性衰减确保头部文档获得显著更高的权重，同时避免尾部文档分数归零。
   
   (2) 多路累加机制：
       文档的最终分数是各路检索贡献的累加和。这种设计体现"民主投票"思想：
       - 被多路都排在前面的文档 → 高分（共识度高）
       - 只在某一路排名靠前 → 中等分数（可能是该路特有的相关文档）
       - 各路排名都靠后 → 低分（可能相关性较弱）
   
   (3) k值的工程意义：
       k=60是学术界和工业界验证的经验值，平衡了：
       - k太小（如10）：排名差异被过度放大，可能遗漏长尾相关内容
       - k太大（如200）：所有文档分数趋同，失去区分度
       - k=60：在头部区分度和长尾覆盖间取得最佳平衡

【在RAG场景中的核心价值】

1. 提升召回率(Recall)
   【原理】单一查询可能因词汇选择问题漏掉相关文档。多角度查询覆盖不同表述方式，
   显著扩大召回范围。实验证明，RAG-Fusion可比单查询提升15-30%的召回率。

2. 降低单源偏差(Single Source Bias)
   【原理】单一检索结果可能受向量模型偏见影响（如过度关注某些关键词）。
   多路检索+融合算法通过"群体智慧"平滑个体偏差，结果更稳健。

3. 提升最终答案相关性
   【原理】RRF排序将多路都认为重要的文档排在前面，这些文档往往：
   - 与用户意图高度匹配
   - 信息覆盖面广
   - 内容质量高
   送入LLM的上下文质量提升，直接改善生成答案的准确性和完整性。

4. 无需训练的无监督优势
   【原理】RRF是纯算法融合，不需要标注数据或模型训练。这使其可以：
   - 即插即用，零成本部署
   - 适用于冷启动场景
   - 与其他检索技术无缝组合

【使用场景】

1. 查询表述模糊或过于简短
   【触发条件】用户输入少于5个词，或问题不够具体
   【示例】"AI应用" → 扩展为医疗、金融、制造等细分领域查询

2. 领域术语存在多种表达方式
   【触发条件】同一概念有多个同义词或近义词
   【示例】"大型语言模型" ↔ "LLM" ↔ "大模型" ↔ "Foundation Model"

3. 需要高召回率的场景
   【触发条件】宁可多召回再筛选，也不能遗漏关键信息
   【示例】法律检索、医学文献检索、合规审查

4. 混合检索结果融合
   【触发条件】同时使用向量检索和关键词检索(BM25)
   【原理】RRF可融合不同检索系统的结果，无需分数归一化

5. 跨语言检索
   【触发条件】查询和文档使用不同语言
   【示例】中文查询扩展为英文、日文等多语言查询，召回多语言文档

6. 无标注数据的排序优化
   【触发条件】无法获取训练数据，不能使用监督式重排序(Learn-to-Rank)
   【优势】RRF提供了一种零样本的排序增强方案

【计算示例】
假设文档A在4个查询结果中的排名分别为：0, 1, 5, -（未出现）
RRF_score(A) = 1/(60+0) + 1/(60+1) + 1/(60+5) + 0
             = 0.01667 + 0.01639 + 0.01538 + 0
             = 0.04844

假设文档B在4个查询结果中的排名分别为：2, 3, 2, 4
RRF_score(B) = 1/(60+2) + 1/(60+3) + 1/(60+2) + 1/(60+4)
             = 0.01613 + 0.01587 + 0.01613 + 0.01563
             = 0.06376

【结论】虽然文档A有1个排名第0，但文档B在4路中都排名靠前，最终RRF分数更高。
这说明RRF算法倾向于奖励"稳定表现"的文档，而非"单点爆发"的文档。
"""
import os

from langchain_classic import hub
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser

# from langchain.load import dumps, loads
from langchain_core.runnables import chain

from langchain_classic import ChatTongyi

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME

# ==================== 1. 初始化LLM模型 ====================
# 【原理】ChatTongyi用于查询扩展阶段，将单一查询改写为多个子查询
# 这需要LLM具备强大的语义理解和生成能力
model = ChatTongyi(
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
    # 参数说明：
    # - dashscope_api_key: 阿里云DashScope平台的API密钥
    # - model: 默认"qwen-turbo"，查询扩展任务对模型要求不高，轻量级模型即可
    # - temperature: 建议0.7-0.9，保证查询多样性
)

# ==================== 2. 初始化嵌入模型 ====================
# 【原理】用于将文本转换为向量，支持语义相似度检索
llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 使用最新的向量嵌入模型
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)  # 从环境变量获取API密钥
    # 参数说明：
    # - model: text-embedding-v3支持多语言、8192 tokens长文本
)

# ==================== 3. 准备测试数据 ====================
# 【设计意图】构造一个混合数据集，包含：
# - 人工智能相关文档（7条）：用于验证查询扩展和召回效果
# - 无关文档（3条）：用于验证RRF的排序筛选能力
texts = [
    "人工智能在医疗诊断中的应用。",
    "人工智能如何提升供应链效率。",
    "NBA季后赛最新赛况分析。",           # 噪声文档
    "传统法式烘焙的五大技巧。",           # 噪声文档
    "红楼梦人物关系图谱分析。",           # 噪声文档
    "人工智能在金融风险管理中的应用。",
    "人工智能如何影响未来就业市场。",
    "人工智能在制造业的应用。",
    "今天天气怎么样",                    # 噪声文档
    "人工智能伦理：公平性与透明度。"
]

# ==================== 4. 构建向量数据库和检索器 ====================
# 【原理】Chroma.from_texts() 自动完成：文本 → 嵌入向量 → 存储
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=llm_embeddings
    # 参数说明：
    # - texts: 文本列表
    # - embedding: 嵌入模型实例
    # - metadatas: 可选，每条文本的元数据
    # - collection_name: 集合名称
)

# 创建基础检索器
retriever = vectorstore.as_retriever()
# as_retriever() 参数说明：
# - search_type: "similarity"(默认) 或 "mmr"(最大边际相关性)
# - search_kwargs: {"k": 4} 返回最相似的4个文档

# ==================== 5. 获取查询扩展Prompt ====================
# 【原理】从LangChain Hub拉取经过社区验证的RAG-Fusion专用Prompt
# 该Prompt经过精心设计，引导LLM生成多样化的查询变体
prompt = hub.pull("langchain-ai/rag-fusion-query-generation")
# hub.pull() 参数说明：
# - repo: Prompt的完整路径，格式"owner/repo-name"
# - api_url: 可选，自定义Hub地址
# - api_key: 可选，私有Hub的认证密钥

print(prompt)

# 【自定义Prompt示例】如需自定义查询扩展策略，可参考以下模板：
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
#     ("user", "Generate multiple search queries related to: {original_query}"),
#     ("user", "OUTPUT (4 queries):")
# ])

# ==================== 6. 构建查询扩展Chain ====================
# 【原理】使用LangChain管道语法(|)串联各组件，形成数据处理流：
# prompt(填充变量) → model(LLM生成) → StrOutputParser(解析文本) → lambda(分割为列表)

generate_queries = (
    prompt | model | StrOutputParser() | (lambda x: x.split("\n"))
)
# 流程详解：
# 1. prompt: 接收{"original_query": "..."}，填充模板生成完整Prompt
# 2. model: 调用LLM，返回AIMessage对象
# 3. StrOutputParser(): 从AIMessage中提取纯文本内容
# 4. lambda x: x.split("\n"): 按行分割，将多行输出转为查询列表

# 执行查询扩展
original_query = "人工智能的应用"
queries = generate_queries.invoke({"original_query": original_query})
print(f"原始查询：{original_query}, 生成的查询：{queries}")

# ==================== 7. RRF融合算法实现 ====================
@chain
def reciprocal_rank_fusion(results: list[list], k=60):
    """
    互逆排序融合(Reciprocal Rank Fusion)算法实现
    
    【算法原理】
    RRF是一种无监督的文档融合排序算法，通过累加各检索结果中排名的倒数来计算文档的综合相关性分数。
    核心思想：被多路检索都排在前面的文档更可能是真正相关的内容。
    
    【数学公式】
    RRF_score(d) = Σ(i=1 to n) [ 1 / (k + r_i(d)) ]
    其中：
    - d: 文档对象
    - n: 检索结果列表数量（即查询数量）
    - r_i(d): 文档d在第i个结果列表中的排名（从0开始）
    - k: 平滑参数，默认60
    
    Args:
        results: 包含多个排序文档列表的二维列表，形状为[查询数][每查询返回文档数]
        k: 融合公式中的平滑参数（默认60）
           - k值越小：排名差异对分数影响越大，头部优势更明显
           - k值越大：分数分布更平滑，长尾文档有更多机会
           - 推荐值：60（学术界验证的经验值）
    
    Returns:
        List[Tuple[Document, float]]: 按融合分数降序排列的文档列表，
                                       每个元素为(文档对象, RRF分数)元组
    
    【使用示例】
    >>> results = [
    ...     [doc1, doc2, doc3],  # 查询1的结果
    ...     [doc2, doc1, doc4],  # 查询2的结果
    ...     [doc1, doc4, doc5]   # 查询3的结果
    ... ]
    >>> ranked = reciprocal_rank_fusion(results, k=60)
    >>> # doc1在所有结果中都排名靠前，将获得最高RRF分数
    """
    # 初始化融合分数字典
    # key: 序列化后的文档字符串（用于唯一标识）
    # value: 该文档的累计RRF分数
    fused_scores = {}

    # 遍历每个检索结果列表（每个查询对应一个结果列表）
    for docs in results:
        # 遍历当前结果列表中的文档，enumerate提供排名索引
        # rank从0开始，0表示该查询结果中排名第一的文档
        for rank, doc in enumerate(docs):
            # 使用dumps()将文档对象序列化为字符串
            # 【原理】Document对象包含page_content和metadata，需要序列化才能作为字典key
            doc_str = dumps(doc)
            
            # 初始化文档得分（如果是首次出现）
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            
            # 计算并累加RRF分数
            # 【核心】排名越靠前（rank值越小），1/(rank+k)的值越大
            # 例如k=60时：rank0=0.0167, rank10=0.0143, rank100=0.00625
            fused_scores[doc_str] += 1 / (rank + k)

    # 按融合分数降序排序
    # sorted(..., reverse=True): 分数高的排在前面
    # key=lambda x: x[1]: 按字典的value（分数）排序
    reranked_results = [
        (loads(doc), score)  # 使用loads()反序列化还原文档对象
        for doc, score in sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ]

    return reranked_results

# ==================== 8. 构建完整RAG-Fusion Chain ====================
# 【原理】将查询扩展、并行检索、RRF融合三个步骤串联为完整流水线

original_query = "人工智能的应用"

'''
【Chain执行流程详解】

步骤1: generate_queries
输入: {"original_query": "人工智能的应用"}
输出: ["AI在医疗诊断中的应用", "人工智能如何改变金融行业", ...] (4个查询)

步骤2: retriever.map()
【原理】Runnable.map()将retriever转换为可处理列表输入的形式
对于输入列表中的每个查询，并行执行retriever.invoke(query)
输入: ["查询1", "查询2", "查询3", "查询4"]
输出: [[doc1, doc2, doc3, doc4], [doc2, doc3, doc5, doc6], ...] (4×4=16个文档)

步骤3: reciprocal_rank_fusion
输入: 二维文档列表
处理: 计算每个文档的RRF分数并排序
输出: [(doc, score), (doc, score), ...] 按分数降序排列

【性能特征】
- 查询扩展: 1次LLM调用，延迟约1-2秒
- 并行检索: 4路同时执行，延迟取决于最慢的一路（通常<500ms）
- RRF融合: 纯计算操作，延迟<10ms
- 总延迟: 约2-3秒，与单次检索相比增加约1.5-2秒
'''

chain = generate_queries | retriever.map() | reciprocal_rank_fusion


# ==================== 9. 执行RAG-Fusion并输出结果 ====================

# 执行完整Chain
result_list = chain.invoke({"original_query": original_query})

# 提取文档内容和对应分数
contents = [doc[0].page_content for doc in result_list]
scores = [doc[1] for doc in result_list]

# 组合显示
combined_tuples = list(zip(contents, scores))
print("--" * 15, "最相关的文档及其得分：")
for item in combined_tuples:
    print(f"  分数: {item[1]:.5f} | 内容: {item[0][:50]}...")

# ==================== 10. 分析分数计算过程（调试/教学用途） ====================
print("\n" + "--" * 15, "分析分数统计过程：")

# 构建只到检索阶段的Chain，用于观察中间结果
chain1 = generate_queries | retriever.map()
chain1_result = chain1.invoke({"original_query": original_query})

# chain1_result是一个二维列表：
# - 外层列表长度 = 生成的查询数量（如4个）
# - 内层每个列表 = 该查询返回的文档列表（如4个）
print("\n【各查询的检索结果】")
for i, docs in enumerate(chain1_result):
    print(f"\n查询{i+1}返回的文档：")
    for j, doc in enumerate(docs):
        print(f"  rank{j}: {doc.page_content[:40]}...")

# 展平所有文档内容（仅用于展示）
all_contents = [
    doc.page_content
    for group in chain1_result  # 遍历外层列表（每个查询的结果）
    for doc in group            # 遍历内层列表（每个查询的文档）
]
print(f"\n总共检索到 {len(all_contents)} 个文档（去重前）")

# ==================== 11. 使用建议与最佳实践 ====================
'''
【参数调优建议】

1. 生成查询数量
   - 默认：4个（由hub prompt控制）
   - 场景适配：
     * 简单查询：2-3个即可
     * 复杂查询：5-6个以覆盖更多角度
   - 注意：查询越多，召回率提升但延迟增加

2. RRF参数k
   - 默认值：60
   - 调整策略：
     * k减小（如30）：更关注头部文档，适合精确检索场景
     * k增大（如100）：更关注长尾文档，适合探索性检索

3. 每路返回文档数
   - 设置：在retriever.as_retriever(search_kwargs={"k": N})
   - 建议：4-6个，平衡召回覆盖率和RRF计算开销

【与其他技术组合】

RAG-Fusion可与以下技术组合使用：
1. + 上下文压缩：先Fusion扩大召回，再Compression精炼内容
2. + 混合检索：Fusion融合向量检索+BM25检索的结果
3. + 重排序(Rerank)：Fusion粗排后，使用Cross-Encoder精排

【性能优化】

1. 异步执行：使用chain.ainvoke()实现非阻塞调用
2. 缓存查询扩展：对高频查询缓存生成的子查询
3. 并行度控制：根据向量数据库的QPS限制调整并发数
'''
