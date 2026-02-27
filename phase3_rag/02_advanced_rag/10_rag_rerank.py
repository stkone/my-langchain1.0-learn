"""
Rerank(重排序)技术实现

【核心原理】
Rerank使用Cross-Encoder(交叉编码器)模型对候选文档进行精排，计算查询与文档的真实相关性分数。

【Cross-Encoder工作机制】

1. 输入格式
   将查询和文档拼接为单一序列输入模型：
   [CLS] 查询文本 [SEP] 文档文本 [SEP]

2. 编码过程
   - 查询和文档的token一起输入Transformer
   - 通过自注意力机制，查询token和文档token相互关注
   - 捕捉细粒度的语义交互（如"孕妇"与"禁用"的关联）

3. 输出计算
   - 取[CLS]位置的隐藏状态
   - 经过全连接层映射到相关性概率
   - 输出0-1之间的分数，越高表示越相关

【与Bi-Encoder的本质区别】

┌─────────────────┬────────────────────┬────────────────────┐
│     特性        │    Bi-Encoder      │   Cross-Encoder    │
├─────────────────┼────────────────────┼────────────────────┤
│ 编码方式        │ 查询/文档分别编码   │ 查询+文档联合编码   │
│ 注意力机制      │ 无交叉注意力        │ 查询与文档相互注意  │
│ 相似度计算      │ 向量点积/余弦      │ 模型输出概率值      │
│ 速度            │ 快(ms级)           │ 慢(100ms级)        │
│ 准确度          │ 中等               │ 高                 │
│ 可处理文档数    │ 百万级             │ 百级               │
└─────────────────┴────────────────────┴────────────────────┘

【为什么Cross-Encoder更准确？】

Bi-Encoder问题：
- 查询和文档分别编码为固定长度向量
- 编码时不知道对方内容，信息孤立
- 仅通过向量相似度判断，无法捕捉复杂语义关系

Cross-Encoder优势：
- 查询和文档同时输入，token级别交互
- 模型可以看到"孕妇"+"感冒"+"禁用"的完整上下文
- 能判断文档是否真正回答了查询，而非仅表面相似

【典型处理流程】

输入：候选文档列表（已命中片段）
  ↓
for each document:
    拼接：[CLS] query [SEP] doc [SEP]
    输入Cross-Encoder模型
    获取相关性分数
  ↓
按分数降序排序
  ↓
输出：排序后的(文档, 分数)列表
"""

from langchain_core.documents import Document

from common_ai.ai_variable import get_ali_rerank

# ==================== 1. 初始化Rerank模型 ====================
# 获取阿里云Rerank模型实例，基于Cross-Encoder架构
reranker = get_ali_rerank()
# 模型输入：查询文本 + 文档文本
# 模型输出：相关性分数(0-1之间)

# ==================== 2. 定义查询和候选文档 ====================
# 注意：documents是已经命中的候选片段，由上游检索器（如向量检索）返回

query = "孕妇感冒了怎么办"

documents = [
    Document(
        page_content="感冒应该吃999感冒灵",
        metadata={"source": "999感冒灵"},
    ),
    Document(
        page_content="高血压患者感冒了吃什么",
        metadata={"source": "高血压患者"},
    ),
    Document(
        page_content="感冒了可以吃感康，但是孕妇禁用",
        metadata={"source": "感康"},
    ),
    Document(
        page_content="感冒了可以咨询专业医生",
        metadata={"source": "专业建议"},
    ),
]

# ==================== 3. 执行Rerank - 方式一：原生API ====================
# reranker.rerank() 内部处理流程：
# 1. 将query与每个document拼接：[CLS] query [SEP] document [SEP]
# 2. 输入Cross-Encoder模型，通过Transformer编码
# 3. 模型输出层计算相关性概率（0-1之间）
# 4. 按分数降序排序返回

scores = reranker.rerank(documents, query)

print("=" * 50)
print("方式一：rerank() 原生API结果")
print("=" * 50)
for doc, score in scores:
    print(f"分数: {score:.4f} | 来源: {doc.metadata['source']}")
    print(f"内容: {doc.page_content}")
    print("-" * 50)

# ==================== 4. 执行Rerank - 方式二：LangChain标准API ====================
# compress_documents() 与 rerank() 的区别：
# - rerank(): 返回(文档, 分数)元组列表，保留所有文档
# - compress_documents(): 返回文档列表，可能过滤低分文档

scores = reranker.compress_documents(documents, query)

print("\n" + "=" * 50)
print("方式二：compress_documents() LangChain标准API结果")
print("=" * 50)
for doc in scores:
    print(f"来源: {doc.metadata['source']}")
    print(f"内容: {doc.page_content}")
    print("-" * 50)

# ==================== 5. 使用示例 ====================
'''
典型使用流程：

# 1. 获取候选文档（由上游检索器返回）
candidates = retriever.invoke(query)  # 如返回20篇

# 2. Rerank精排
ranked = reranker.rerank(candidates, query)

# 3. 取Top-N用于后续处理
top_docs = [doc for doc, score in ranked[:5]]
'''