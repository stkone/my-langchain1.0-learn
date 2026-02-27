# 导入LangChain核心组件
import os
from typing import List

from langchain_classic.retrievers import MultiVectorRetriever, ParentDocumentRetriever  # 多向量检索器核心组件
from langchain_community.chat_models import ChatTongyi  # 通义千问大模型
from langchain_community.document_loaders import TextLoader  # 文本加载器，用于加载文档
from langchain_community.embeddings import DashScopeEmbeddings  # 阿里云向量嵌入模型
from langchain_community.vectorstores import Chroma  # Chroma向量数据库
from langchain_core.documents import Document  # 文档对象
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_core.runnables import RunnableMap, RunnableParallel  # 可运行的映射
from langchain_core.stores import InMemoryStore, InMemoryByteStore  # 内存字节存储
from langchain_core.utils import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归文本分割器
from openai import BaseModel
from pydantic import Field

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME

"""
假设性问题RAG (Hypothetical Questions RAG) 深度解析
================================================

核心概念：
--------
假设性问题RAG是一种创新的检索增强生成技术，通过预先为文档生成可能的用户问题来提升检索效果。

实现原理：
--------
1. 离线预处理阶段：
   - 将原始文档分割成适当大小的块
   - 使用大模型为每个文档块生成3个假设性问题
   - 将这些问题向量化并存储到向量数据库
   - 原始文档存储到字节存储中，通过doc_id关联

2. 在线检索阶段：
   - 用户提出查询问题
   - 向量数据库检索与查询最相似的假设性问题
   - 通过doc_id映射找到对应的原始文档
   - 使用原始文档回答用户的真实问题

技术优势：
--------
1. 检索精度提升：假设性问题比原始文档更能表达用户可能的查询意图
2. 语义理解增强：大模型生成的问题体现了对文档深层语义的理解
3. 查询匹配优化：将用户查询与预生成的问题进行匹配，提高相关性
4. 上下文完整性：最终返回完整原始文档而非问题片段

使用场景：
--------
1. 企业知识库问答：预生成常见业务问题，提升检索准确性
2. 技术文档查询：为API文档、使用手册生成技术问题
3. 新闻资讯检索：为新闻文章生成读者可能关心的问题
4. 学术文献分析：为研究论文生成学术性问题
5. 产品FAQ系统：预生成用户常见疑问和对应的解答文档

与传统RAG的区别：
----------------
传统RAG：用户查询 → 向量检索原始文档 → 生成回答
假设性问题RAG：用户查询 → 向量检索预生成问题 → 映射原始文档 → 生成回答

这种方式通过"问题-问题"匹配替代"查询-文档"匹配，显著提升了检索的语义准确性。

================================================
假设性问题 vs Query改写的深度对比分析
================================================

核心区别对比：
-------------

| 维度 | 假设性问题RAG | Query改写RAG |
|------|---------------|--------------|
| **处理时机** | 离线预处理 | 在线实时处理 |
| **资源消耗** | 预先消耗大量计算资源 | 每次查询消耗计算资源 |
| **响应速度** | 检索阶段快，无需额外处理 | 需要额外的Query改写步骤 |
| **准确性** | 依赖预生成问题质量 | 依赖改写模型的即时理解能力 |
| **维护成本** | 需要定期重新生成问题 | 相对较低，但每次都要改写 |

技术实现原理对比：
-----------------

假设性问题RAG原理：
1. 离线阶段：大模型深度理解文档 → 生成代表性问题 → 向量化存储
2. 在线阶段：用户查询 → 向量相似度匹配 → 映射原始文档
3. 核心思想：用机器预先学习的问题来代表文档语义

Query改写RAG原理：
1. 在线阶段：用户原始查询 → 大模型理解意图 → 生成多个改写版本
2. 检索阶段：对每个改写版本进行向量检索
3. 核心思想：用机器即时理解用户查询意图

适用场景深度分析：
-----------------

假设性问题RAG适用场景：
✅ 文档内容相对稳定的企业知识库
✅ 用户查询模式可预测的FAQ系统
✅ 需要高检索精度的专业领域问答
✅ 对响应时间要求较高的应用场景
✅ 文档更新频率较低的知识管理系统

❌ 不适用场景：
- 文档频繁更新的内容平台
- 用户查询极其多样化的开放域问答
- 实时性要求极高的新闻资讯检索
- 需要处理大量临时文档的场景

Query改写RAG适用场景：
✅ 文档内容动态变化的场景
✅ 用户查询多样化且难以预测
✅ 需要处理临时上传文档的应用
✅ 对文档预处理要求较低的系统
✅ 查询意图复杂的长尾问题处理

❌ 不适用场景：
- 对响应时间要求极高的系统
- 计算资源受限的边缘设备
- 查询量巨大的高并发场景
- 需要保证检索一致性的专业系统

性能特征对比：
-------------

假设性问题RAG性能特征：
📈 优点：
- 检索阶段响应极快（毫秒级）
- 查询一致性好，相同问题总是得到相似结果
- 可以进行离线优化和质量控制
- 支持批量预处理，提高整体效率

📉 缺点：
- 预处理阶段耗时较长
- 对新文档需要重新生成问题
- 存储开销较大（存储问题向量）
- 难以处理未预见的查询模式

Query改写RAG性能特征：
📈 优点：
- 无需预处理阶段，部署简单
- 能够适应新的查询模式
- 存储开销相对较小
- 对文档更新响应及时

📉 缺点：
- 每次查询都需要额外的改写时间
- 改写质量直接影响检索效果
- 高并发场景下成本较高
- 结果可能存在一定随机性

成本效益分析：
-------------

假设性问题RAG成本结构：
- 一次性预处理成本：较高（需要大量LLM调用）
- 日常运营成本：较低（主要是向量检索）
- 存储成本：中等（需要存储问题向量）
- 维护成本：定期更新（文档变更时）

Query改写RAG成本结构：
- 一次性预处理成本：很低（几乎为零）
- 日常运营成本：较高（每次查询都有LLM调用）
- 存储成本：很低（只需存储原始文档）
- 维护成本：很低（基本无需维护）

质量控制维度：
-------------

假设性问题RAG质量控制：
🔍 可控因素：
- 预生成问题的质量和覆盖面
- 问题生成提示词的设计
- 向量嵌入模型的选择
- 相似度阈值的设定

🔍 难控因素：
- 未预见查询模式的处理
- 文档更新后的适应性
- 新兴概念的理解能力

Query改写RAG质量控制：
🔍 可控因素：
- 改写提示词的优化
- 改写策略的选择（单改写vs多改写）
- 检索结果融合策略
- 大模型改写能力

🔍 难控因素：
- 改写的准确性和相关性
- 改写的一致性
- 复杂查询的理解深度

混合策略建议：
-------------

最佳实践推荐采用混合策略：

1. **分层处理**：
   - 高频核心问题使用假设性问题RAG
   - 长尾复杂问题使用Query改写RAG
   - 根据查询特征动态选择策略

2. **缓存机制**：
   - 对Query改写的优秀结果进行缓存
   - 建立热门查询的假设性问题库
   - 实现两种策略的优势互补

3. **动态切换**：
   - 监控查询效果，自动调整策略
   - 根据响应时间和准确性权衡选择
   - 支持手动配置不同场景的默认策略

技术选型决策矩阵：
-----------------

选择假设性问题RAG当：
- 文档更新频率 < 每月一次
- 查询模式相对固定
- 对响应时间要求 < 100ms
- 有充足的预处理时间和资源
- 需要保证结果一致性

选择Query改写RAG当：
- 文档实时更新或频繁变化
- 查询模式多样化不可预测
- 可接受稍长的响应时间（200-500ms）
- 计算资源充足但存储资源有限
- 需要处理临时或一次性文档

未来发展展望：
-------------

1. **智能化融合**：AI自动判断最优策略
2. **增量更新**：支持文档的增量式问题生成
3. **个性化适配**：根据不同用户群体优化策略
4. **多模态扩展**：支持图像、音频等多媒体文档
5. **联邦学习**：在保护隐私前提下共享问题知识

这种对比分析帮助开发者根据具体业务需求选择最适合的RAG实现策略。
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
loader = TextLoader("../../Data/deepseek百度百科.txt", encoding="utf-8")
# 加载文档到内存中
docs = loader.load()

"""
文档预处理流程详解：
------------------
1. 文档加载：将原始文本文件转换为LangChain Document对象
2. 文本分割：使用RecursiveCharacterTextSplitter将长文档切分成1024字符的块
3. 重叠处理：设置100字符重叠确保语义连贯性
4. 批量处理：为每个文档块准备后续的假设性问题生成
"""

# 初始化递归文本分割器（设置块大小和重叠）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# 初始化Chroma向量数据库（存储生成的问题向量）
vectorstore = Chroma(
    collection_name="hypo-questions", embedding_function=llm_embeddings
)
# 初始化内存存储（存储原始文档）
store = InMemoryByteStore()

id_key = "doc_id" # 文档标识键名

# 配置多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore, #  向量数据库，存储生成的问题向量（调用对话模型生成）
    byte_store=store, # 字节存储，存储原始文档
    id_key=id_key,
)

# 为每个原始文档生成唯一ID
doc_ids = [str(uuid.uuid4()) for _ in docs]

"""
数据结构设计说明：
----------------
1. doc_ids：为每个文档块生成全局唯一标识符
2. vectorstore：存储假设性问题的向量表示，用于相似性检索
3. store：存储原始文档内容，通过doc_id进行关联
4. id_key：定义关联字段名，确保向量和文档的正确映射

关联机制：
doc_id作为桥梁连接：
- 向量数据库中的问题向量 ←→ doc_id
- 内存存储中的原始文档 ←→ doc_id
这样实现了"问题→文档"的间接关联
"""

# 此处使用双括号 {{ 和 }} 是为了在字符串中转义出单个 { 和 }，以确保最终输出的 JSON 格式正确。
prompt = ChatPromptTemplate.from_template(
    """请基于以下文档生成3个假设性问题（必须使用JSON格式）:
    {doc}

    要求：
    1. 输出必须为合法JSON格式，包含questions字段
    2. questions字段的值是包含3个问题的数组
    3. 使用中文提问
    示例格式：
    {{
        "questions": ["问题1", "问题2", "问题3"]
    }}"""
)

"""
提示词设计要点：
--------------
1. 明确任务：要求基于文档内容生成假设性问题
2. 格式约束：强制使用JSON格式确保解析可靠性
3. 数量控制：固定生成3个问题平衡覆盖率和效率
4. 语言规范：要求使用中文确保问题质量
5. 结构清晰：提供标准JSON示例避免格式错误

这种设计确保了大模型输出的一致性和可解析性。
"""

#以下开始用大模型生成假设性问题
#自定义模型输出格式
class HypotheticalQuestions(BaseModel):
    """约束生成假设性问题的格式"""
    questions: List[str] = Field(..., description="List of questions")

"""
Pydantic模型设计说明：
--------------------
1. 类型安全：通过typing.List[str]确保输出为字符串列表
2. 必需字段：Field(...)中的...表示该字段必须存在
3. 描述信息：为字段添加描述便于理解和文档生成
4. 结构化输出：配合with_structured_output实现严格的格式控制

这个模型作为输出解析器，确保大模型严格按照预定格式生成内容。
"""

# 创建假设性问题链
'''
其中的client.with_structured_output可以理解为输出解析器的一种更高级用法
将大模型的输出转换为HypotheticalQuestions所限定的格式，
而HypotheticalQuestions要求的格式是：
定义了一个字段 questions，它具有以下特性：
类型注解：List[str] 表示 questions 字段应该是一个字符串列表。
必需性：Field(...) 中的省略号 ... 表示这个字段是必需的。
描述信息：description="List of questions" 为该字段添加了描述，这对于生成文档或帮助理解模型结构很有用。
'''
chain = (
    {"doc": lambda x: x.page_content}
    | prompt
    # 将LLM输出构建为字符串列表
    | model.with_structured_output(
        HypotheticalQuestions
    )
    # 提取问题列表
    | (lambda x: x.questions)
)

"""
处理链工作流程：
---------------
1. 输入文档内容提取：lambda x: x.page_content获取文档文本
2. 应用提示词模板：将文档内容插入到提示词中
3. 结构化输出解析：使用with_structured_output确保JSON格式正确
4. 结果提取：从HypotheticalQuestions对象中提取questions列表

这种链式处理确保了从原始文档到结构化问题列表的完整转换。
"""

# 批量处理所有文档生成假设性问题（最大并行数5），每个切块后的文档块都对应的生成三个问题
hypothetical_questions = chain.batch(docs, {"max_concurrency": 5})

"""
批处理优化策略：
---------------
1. 并发控制：max_concurrency=5避免过多并发请求导致API限流
2. 批量处理：一次性处理所有文档块提高整体效率
3. 资源管理：合理控制并发数平衡处理速度和系统稳定性
4. 错误处理：批量处理天然具备一定的容错能力

这种方法比逐个处理文档块效率更高，特别适合大规模文档处理场景。
"""

# 将生成的问题转换为带元数据的文档对象
question_docs = []
for i, question_list in enumerate(hypothetical_questions):
    question_docs.extend(
        [Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list]
    )
# 将问题文档存入向量数据库
retriever.vectorstore.add_documents(question_docs)
# 将原始文档存入字节存储（通过ID关联）
retriever.docstore.mset(list(zip(doc_ids, docs)))

"""
数据存储架构详解：
----------------
存储流程：
1. 问题文档构建：
   - 为每个生成的问题创建Document对象
   - 添加metadata包含对应的doc_id
   - 建立问题与原始文档的关联关系

2. 向量数据库存储：
   - 将问题文档添加到vectorstore
   - 系统自动为问题内容生成向量表示
   - 问题向量用于后续的相似性检索

3. 原始文档存储：
   - 使用doc_ids作为键，原始docs作为值
   - 通过mset批量存储提高效率
   - 建立doc_id到完整文档的映射

这种双重存储设计确保了既能高效检索又能获得完整上下文。
"""

prompt1 =  ChatPromptTemplate.from_template("根据下面的文档回答问题:\n\n{doc}\n\n问题: {question}")
# 生成问题回答链
query = "deepseek受到哪些攻击？"
chain = RunnableParallel({
    "doc": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt1 | model | StrOutputParser()

"""
问答链构建解析：
---------------
1. 并行处理设计：
   - RunnableParallel同时处理文档检索和问题传递
   - 提高处理效率，减少响应时间

2. 检索机制：
   - retriever.invoke执行假设性问题检索
   - 通过向量相似性找到最相关的预生成问题
   - 自动映射到对应的原始文档

3. 回答生成：
   - 将检索到的完整文档和原始问题组合
   - 通过大模型生成最终回答
   - StrOutputParser确保返回纯文本结果

这种设计充分利用了假设性问题RAG的优势。
"""

# 生成问题回答
answer = chain.invoke({"question": query})
print("-------------回答--------------")
print(answer)
#  返回的是知识块
retrieved_docs = retriever.invoke(query)
print("-------------检索到的问题--------------")
print(retrieved_docs)

"""
检索过程详解：
-------------
1. 查询处理：
   - 用户问题："deepseek受到哪些攻击？"
   - 系统在向量数据库中检索最相似的假设性问题

2. 相似性匹配：
   - 计算用户查询与预生成问题的语义相似度
   - 找到最相关的假设性问题

3. 文档映射：
   - 通过doc_id找到假设性问题对应的原始文档
   - 返回完整文档内容而非问题片段

4. 最终回答：
   - 基于完整文档内容生成针对性回答
   - 确保回答的准确性和完整性

这种方法比直接检索原始文档具有更高的语义匹配精度。
"""

"""
最佳实践建议：
------------
1. 问题生成质量：
   - 确保生成的问题具有代表性
   - 问题应该覆盖文档的主要信息点
   - 避免生成过于宽泛或模糊的问题

2. 参数调优：
   - 根据文档类型调整chunk_size
   - 合理设置问题生成数量（3-5个较为合适）
   - 控制batch处理的并发数

3. 性能优化：
   - 预生成问题可以离线完成
   - 向量数据库可以持久化存储
   - 考虑使用缓存机制提升重复查询效率

4. 质量评估：
   - 定期检查生成问题的相关性
   - 验证检索结果的准确性
   - 监控用户满意度指标

5. 适用场景选择：
   - 适合文档结构相对稳定的知识库
   - 适用于用户查询模式相对可预测的场景
   - 特别适合FAQ和常见问题解答系统
"""
