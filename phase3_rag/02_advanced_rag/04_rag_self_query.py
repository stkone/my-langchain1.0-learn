"""
Self Query RAG 实现详解

Self Query（自查询）是一种高级RAG检索技术，能够将自然语言查询自动转换为结构化查询，
结合向量检索和元数据过滤，大幅提升检索精度。

核心优势：
1. 自动解析复杂查询条件（如"2023年评分9分以上的AI技术文章"）
2. 智能分离文本语义搜索和结构化过滤条件
3. 减少无关文档的干扰，提高检索相关性
"""

import os

# 导入LangChain核心组件
from langchain_classic.chains.query_constructor.base import get_query_constructor_prompt, StructuredQueryOutputParser
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_community.chat_models import ChatTongyi  # 通义千问大模型
from langchain_community.embeddings import DashScopeEmbeddings  # 阿里云向量嵌入模型
from langchain_community.vectorstores import Chroma  # Chroma向量数据库
from langchain_core.documents import Document  # 文档对象

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME

# ==================== 模型和嵌入配置 ====================
# 1. 初始化大语言模型 - 用于理解和解析自然语言查询
# 这里的LLM负责将用户的自然语言查询转换为结构化查询条件
model = ChatTongyi(
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)

# 2. 初始化嵌入模型 - 用于将文档内容转换为向量表示
# 向量嵌入是实现语义相似度计算的基础
llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 使用最新的向量嵌入模型
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)  # 从环境变量获取API密钥
)

# ==================== 示例文档数据 ====================
# 创建带有丰富元数据的技术文章数据集
# 每个文档包含内容和结构化元数据（年份、评分、领域、作者）
docs = [
    Document(
        page_content="作者A团队开发出基于人工智能的自动驾驶决策系统，在复杂路况下的响应速度提升300%",
        metadata={"year": 2024, "rating": 9.2, "genre": "AI", "author": "A"},
    ),
    Document(
        page_content="区块链技术成功应用于跨境贸易结算，作者B主导的项目实现交易确认时间从3天缩短至30分钟",
        metadata={"year": 2023, "rating": 9.8, "genre": "区块链", "author": "B"},
    ),
    Document(
        page_content="云计算平台实现量子计算模拟突破，作者C构建的新型混合云架构支持百万级并发计算",
        metadata={"year": 2022, "rating": 8.6, "genre": "云", "author": "C"},
    ),
    Document(
        page_content="大数据分析预测2024年全球经济趋势，作者A团队构建的模型准确率超92%",
        metadata={"year": 2023, "rating": 8.9, "genre": "大数据", "author": "A"},
    ),
    Document(
        page_content="人工智能病理诊断系统在胃癌筛查中达到三甲医院专家水平，作者B获医疗科技创新奖",
        metadata={"year": 2024, "rating": 7.1, "genre": "AI", "author": "B"},
    ),
    Document(
        page_content="基于区块链的数字身份认证系统落地20省市，作者C设计的新型加密协议通过国家级安全认证",
        metadata={"year": 2022, "rating": 8.7, "genre": "区块链", "author": "C"},
    ),
    Document(
        page_content="云计算资源调度算法重大突破，作者A研发的智能调度器使数据中心能效提升40%",
        metadata={"year": 2023, "rating": 8.5, "genre": "云", "author": "A"},
    ),
    Document(
        page_content="大数据驱动城市交通优化系统上线，作者B团队实现早晚高峰通行效率提升25%",
        metadata={"year": 2024, "rating": 7.4, "genre": "大数据", "author": "B"},
    )
]

# ==================== 向量数据库初始化 ====================
# 3. 创建向量数据库 - 存储文档向量表示，支持语义相似度检索
vectorstore = Chroma.from_documents(docs, llm_embeddings)

# ==================== 元数据字段定义 ====================
# 定义文档的结构化元数据字段信息
# 这些定义指导LLM如何理解和解析查询中的过滤条件
metadata_field_info = [
    AttributeInfo(
        name="genre",           # 字段名
        description="文章的技术领域，选项:['AI', '区块链', '云', '大数据']",  # 字段描述
        type="string",          # 数据类型
    ),
    AttributeInfo(
        name="year",
        description="文章的出版年份",
        type="integer",
    ),
    AttributeInfo(
        name="author",
        description="署名文章的作者姓名",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="技术价值评估得分（1-10分）",
        type="float"
    )
]

# 文档内容的整体描述，帮助LLM理解文档的性质
document_content_description = "技术文章简述"

# ==================== SelfQueryRetriever 核心实现 ====================
"""
SelfQueryRetriever 工作原理：

1. 查询解析阶段：
   - 接收自然语言查询："查找2023年评分超过9分的AI技术文章"
   - 使用LLM分析查询，自动识别：
     * 文本语义部分："AI技术文章" 
     * 结构化过滤条件：year=2023, rating>9, genre='AI'

2. 检索执行阶段：
   - 向量检索：使用文本语义部分在向量库中查找相关文档
   - 元数据过滤：应用识别出的结构化条件进行精确过滤

3. 结果返回：
   - 返回同时满足语义相关性和结构化条件的文档

相比传统RAG的优势：
- 传统RAG只能做全文语义检索，容易返回不相关的高分文档
- Self Query可以结合语义和结构化过滤，显著提升检索精度
"""

# 初始化SelfQuery检索器
retriever = SelfQueryRetriever.from_llm(
    llm=model,                           # 用于查询解析的大语言模型
    vectorstore=vectorstore,             # 向量数据库
    document_contents=document_content_description,  # 文档内容描述
    metadata_field_info=metadata_field_info,         # 元数据字段信息
    # enable_limit=True,                 # 可选：限制返回结果数量
)

# ==================== 使用示例 ====================
# 示例1：复合条件查询
# 查询："我想了解评分在9分以上的文章"
# Self Query会自动识别：rating > 9
print("---------------------评分在9分以上的文章-------------------------------")
result1 = retriever.invoke("我想了解评分在9分以上的文章")
for doc in result1:
    print(f"文档: {doc.page_content}")
    print(f"元数据: {doc.metadata}\n")

# 示例2：多条件组合查询  
# 查询："作者B在2023年发布的文章"
# Self Query会自动识别：author="B" AND year=2023
print("---------------------作者B在2023年发布的文章-------------------------------")
result2 = retriever.invoke("作者B在2023年发布的文章")
for doc in result2:
    print(f"文档: {doc.page_content}")
    print(f"元数据: {doc.metadata}\n")

# ==================== 内部机制分析 ====================
"""
深入理解Self Query的工作机制：
"""

# 1. 构建查询构造提示模板
# 这个模板指导LLM如何将自然语言转换为结构化查询
prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)

# 2. 创建结构化查询解析器
# 将LLM输出解析为StructuredQuery对象
output_parser = StructuredQueryOutputParser.from_components()

# 3. 构建完整的查询构造链
# 用户查询 -> 提示模板 -> LLM处理 -> 结构化解析
query_constructor = prompt | model | output_parser

# 查看提示词模板内容
print("=== Self Query 提示词模板 ===")
print(prompt.format(query="我想了解评分在9分以上的文章"))
print("=" * 50)

# 查看结构化查询的解析结果
print("=== 结构化查询解析结果 ===")
structured_query = query_constructor.invoke({"query": "作者B在2023年发布的文章"})
print(f"原始查询: 作者B在2023年发布的文章")
print(f"解析结果: {structured_query}")
print(f"- 文本查询部分: {structured_query.query}")
print(f"- 过滤条件: {structured_query.filter}")
print("=" * 50)

# ==================== Self Query 的应用场景 ====================
"""
Self Query 适用场景：

1. 结构化数据检索：
   - 企业知识库（按部门、日期、重要程度过滤）
   - 学术论文检索（按年份、期刊、引用次数过滤）
   - 产品目录搜索（按价格、品牌、规格过滤）

2. 复合条件查询：
   - "最近三个月内关于AI的高评分技术报告"
   - "2023年发布的区块链相关专利，按重要性排序"

3. 精准检索需求：
   - 当普通语义检索返回太多无关结果时
   - 需要结合文本内容和结构化属性的场景

与传统RAG的关系：
- Self Query是RAG的重要优化手段
- 在传统向量检索基础上增加了智能过滤能力
- 特别适合有丰富元数据的文档集合
"""

print("\n=== Self Query RAG 总结 ===")
print("✓ 自动解析自然语言中的结构化查询条件")
print("✓ 结合语义检索和精确过滤，提升检索精度")  
print("✓ 适用于有结构化元数据的文档检索场景")
print("✓ 是传统RAG的有效增强和优化手段")
"""
get_query_constructor_prompt 参数详解：
=====================================

函数签名：
get_query_constructor_prompt(
    document_content_description,  # 文档内容描述
    metadata_field_info,          # 元数据字段信息  
    examples=None,                # 可选示例（few-shot学习）
    allowed_comparators=None,     # 允许的比较操作符
    allowed_operators=None        # 允许的逻辑操作符
)

参数详细说明：

1. document_content_description（必需）
   - 作用：描述文档的整体内容和性质
   - 类型：str
   - 示例："技术文章简述"、"产品说明书"、"法律条文集合"
   - 重要性：帮助LLM理解文档领域和上下文

2. metadata_field_info（必需）
   - 作用：定义文档的结构化元数据字段
   - 类型：List[AttributeInfo]
   - 包含：字段名、描述、数据类型
   - 示例：
     AttributeInfo(
         name="year",
         description="文章发布年份", 
         type="integer"
     )

3. examples（可选）
   - 作用：提供查询转换示例，用于few-shot学习
   - 类型：List[QueryTransformerExample] 或 None
   - 默认：None
   - 用途：提高LLM转换准确性

4. allowed_comparators（可选）
   - 作用：限制可用的比较操作符
   - 默认：["eq", "ne", "lt", "lte", "gt", "gte", "in", "nin"]
   - 说明：
     eq: 等于 (=)    ne: 不等于 (!=)
     lt: 小于 (<)    lte: 小于等于 (<=)  
     gt: 大于 (>)    gte: 大于等于 (>=)
     in: 在...中     nin: 不在...中

5. allowed_operators（可选）
   - 作用：限制可用的逻辑操作符
   - 默认：["and", "or", "not"]
   - 用途：控制复合查询的逻辑关系

使用示例：
========

示例1：基础用法
prompt = get_query_constructor_prompt(
    document_content_description="技术博客文章",
    metadata_field_info=[
        AttributeInfo(name="author", description="作者姓名", type="string"),
        AttributeInfo(name="year", description="发布年份", type="integer"),
        AttributeInfo(name="tags", description="文章标签", type="string"),
    ]
)

示例2：带示例的学习模式
prompt = get_query_constructor_prompt(
    document_content_description="产品说明书",
    metadata_field_info=[
        AttributeInfo(name="category", description="产品类别", type="string"),
        AttributeInfo(name="price", description="产品价格", type="float"),
    ],
    examples=[
        QueryTransformerExample(
            query="便宜的电子产品",
            structured_query={
                "query": "电子产品",
                "filter": {"lt": ["price", 100]}
            }
        )
    ]
)

示例3：限制操作符范围
prompt = get_query_constructor_prompt(
    document_content_description="学术论文",
    metadata_field_info=[
        AttributeInfo(name="year", description="发表年份", type="integer"),
        AttributeInfo(name="citations", description="引用次数", type="integer"),
    ],
    allowed_comparators=["eq", "gt", "gte"],  # 只允许等于、大于、大于等于
    allowed_operators=["and"]  # 只允许AND操作
)
"""

