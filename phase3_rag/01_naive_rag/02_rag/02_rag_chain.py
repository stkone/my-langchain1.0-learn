"""
RAG (Retrieval-Augmented Generation) 是 LangChain 的核心应用场景之一
LangChain为RAG提供了很多预制链，使用这这些预制链条可以进行快速构建

=======================================
RAG核心原理深度解析:
=======================================
RAG系统采用"检索-生成"两阶段架构，解决了大语言模型的三大痛点：
1. 知识时效性问题 - 通过外部知识库接入最新信息
2. 领域专业性不足 - 注入特定领域的专业知识  
3. 事实幻觉问题 - 基于检索到的真实文档生成答案

系统工作流程：
[用户查询] → [向量检索] → [文档召回] → [上下文组装] → [LLM生成] → [最终答案]

=======================================
核心组件解析:
=======================================
create_stuff_documents_chain: 文档处理链
- 功能：将检索到的多个文档内容"填充"(stuff)到Prompt模板中
- 适用场景：文档数量较少(通常<5个)，总长度不超过LLM上下文窗口
- 实现原理：简单的文档拼接策略，将所有相关文档按顺序插入到context变量位置
- 优势：实现简单，保持文档完整性
- 局限：文档过多时容易超出上下文限制

create_retrieval_chain: 检索链
- 功能：组合检索器和文档处理链，构建完整的RAG流程
- 工作机制：
  1. 接收用户输入(input)
  2. 使用检索器(retriever)查找相关文档
  3. 将检索结果传递给文档处理链
  4. 返回包含答案和源文档的完整响应
- 设计思想：责任分离，检索逻辑与文档处理逻辑解耦
"""
import os
import bs4
import langchain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatTongyi

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME

# =======================================
# 第一阶段：模型初始化
# =======================================
# 1.获得访问大模型客户端
# 使用通义千问作为主语言模型，负责最终的答案生成
model = ChatTongyi()

# 2.获得向量嵌入模型  
# 用于将文本转换为向量表示，支持语义相似度计算
# DashScope text-embedding-v3是阿里云提供的高质量中文嵌入模型
llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)

# =======================================
# 第二阶段：文档处理管道 (索引前阶段)
# =======================================
# 3.加载网站内容
# WebBaseLoader专门用于加载网页内容，支持HTML解析和内容过滤
# 使用bs4.SoupStrainer只提取指定ID的内容区域，提高效率并减少噪声
loader = WebBaseLoader(
        web_path="https://www.gov.cn/yaowen/liebiao/202512/content_7050416.htm",
        # bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT"))
        # 分割，将网页中目标内容进行分割
        bs_kwargs={"parse_only":bs4.SoupStrainer(id="UCAP-CONTENT")}
        )
docs = loader.load()

# 4.文本的切割
# RecursiveCharacterTextSplitter是LangChain的核心文本分割器
# 参数详解：
# - chunk_size=500: 每个文本块最大500字符(中文约150-200个汉字)
# - chunk_overlap=100: 块间重叠100字符，确保语义连贯性不被切断
# 策略：优先按段落(\n\n)分割，其次按句子(。！？)分割，最后按字符分割
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(docs)

# 5.存储到向量空间
# Chroma是轻量级的向量数据库，适合原型开发和小规模应用
# 将分割后的文档片段转换为向量并建立索引，支持高效的相似度检索
db = Chroma.from_documents(documents=documents,embedding=llm_embeddings)

# =======================================
# 第三阶段：检索器配置 (索引阶段)
# =======================================
# 6.检索器初始化
# as_retriever()将向量数据库转换为检索器接口
# 默认使用相似度搜索，返回最相关的k个文档(默认k=4)
retriever = db.as_retriever()

# =======================================
# 第四阶段：提示模板设计 (生成阶段准备)
# =======================================
# 7. 初始化模版
# 注意这里的prompt模板中包含 {context} 和 {input} 的模板
# 需要使用{context}，这个变量，来表示上下文，这个变量，会自动从retriever中获取。
# 而human中也限定了变量{input}，链的必须使用这个变量。
system_prompt = """
    您是问答任务的助理。使用以下的上下文来回答问题，
    上下文：<{context}>
    如果你不知道答案，不要其他渠道去获得答案，就说你不知道。
"""
# ChatPromptTemplate.from_messages创建多轮对话模板
# system角色定义助手的行为准则
# human角色接收用户输入，必须使用{input}变量名以匹配链的预期输入
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# =======================================
# 第五阶段：RAG链构建 (核心实现)
# =======================================
# 8. 创建文档处理链 - create_stuff_documents_chain
# =====================================================================
# 🔍 深度解析 create_stuff_documents_chain
# =====================================================================
# 功能定位：文档合并处理器
# - 接收格式：{"input": "用户问题", "context": [Document对象列表]}
# - 处理逻辑：将context中的所有文档内容拼接到Prompt的{context}位置
# - 输出格式：字符串(经过LLM处理后的最终答案)
#
# 实现原理：
# 1. 提取context字段中的Document对象列表
# 2. 遍历所有文档，将其page_content属性连接成单一字符串
# 3. 将拼接后的文本替换Prompt模板中的{context}占位符
# 4. 调用LLM生成最终答案
#
# 适用场景：
# ✅ 文档数量较少(<5个)且总长度可控的场景
# ✅ 需要保持文档完整性的精确问答
# ✅ 对上下文顺序有要求的场景
#
# 局限性：
# ❌ 文档过多时容易超出LLM上下文窗口限制
# ❌ 无法智能筛选最重要的信息片段
# ❌ 缺乏对文档相关性的排序机制
chain1 = create_stuff_documents_chain(model,prompt_template)

# 8. 创建检索链 - create_retrieval_chain  
# =====================================================================
# 🔍 深度解析 create_retrieval_chain
# =====================================================================
# 功能定位：完整的RAG流程协调器
# - 参数1: 检索器(retriever) - 负责文档检索
# - 参数2: 文档处理链(chain1) - 负责文档处理和答案生成
#
# 工作机制详解：
# 1. 接收用户输入 {"input": "问题"}
# 2. 调用retriever.invoke()进行向量相似度检索
# 3. 将检索结果包装成 {"input": "问题", "context": [文档列表]} 格式
# 4. 传递给chain1(create_stuff_documents_chain)进行文档处理
# 5. 返回完整响应 {"input": "原问题", "context": [文档], "answer": "答案"}
#
# 设计优势：
# 🎯 职责分离：检索逻辑与文档处理逻辑完全解耦
# 🎯 接口标准化：统一的输入输出格式，便于组合和扩展
# 🎯 可替换性：可以轻松替换不同的检索器或文档处理策略
#
# 响应结构：
# {
#   "input": "用户原始问题",
#   "context": [Document对象列表],  # 检索到的相关文档
#   "answer": "LLM生成的答案"       # 最终回答
# }
chain2 = create_retrieval_chain(retriever,chain1)

# =======================================
# 第六阶段：执行推理并获取答案
# =======================================
# 9. 用大模型生成答案
# invoke方法触发整个RAG流程：
# 1. 输入问题通过retriever检索相关文档
# 2. 检索到的文档被chain1处理并注入Prompt
# 3. LLM基于增强的上下文生成最终答案
resp = chain2.invoke({"input":"会议说了什么?"})

# 输出结果分析
print(type(resp))  # <class 'dict'> - 返回字典类型包含完整信息
print(resp)        # 完整响应对象，包含input/context/answer三个字段
print("===================")
print(resp["answer"])  # 最终答案文本
