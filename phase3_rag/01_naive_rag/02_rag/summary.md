# LangChain RAG (检索增强生成) 完全指南

> 深入解析LangChain 1.0中RAG系统的核心技术，从基础实现到生产环境部署

## 一、RAG架构概述

RAG（Retrieval-Augmented Generation，检索增强生成）是LangChain的核心应用场景之一，它通过结合信息检索和文本生成，解决大语言模型在以下方面的局限：

- **知识时效性**：接入最新文档数据
- **领域专业性**：注入特定领域知识
- **事实准确性**：减少"幻觉"现象
- **可解释性**：提供答案来源

![RAG架构图](https://langchain-doc.com/images/rag-architecture.png) <!-- 此处应为架构图 -->

LangChain中的RAG实现通常分为三个主要阶段：

1. **文档处理阶段**（索引前阶段）

   - 获取文档 → 分割文档 → 向量化 → 存储到向量数据库
2. **文档搜索阶段**（索引阶段）

   - 将用户查询向量化 → 检索相关文档片段
3. **大模型生成阶段**（生成阶段）

   - 构建提示模板 → 注入上下文 → 生成最终答案

## 二、核心实现方式对比

LangChain提供两种主要的RAG实现方式，各有优劣：


| 实现方式                       | 代码特点                           | 适用场景               | 灵活性     | 学习曲线   |
| ------------------------------ | ---------------------------------- | ---------------------- | ---------- | ---------- |
| **原生构建** (01_navie_rag.py) | 使用Runnable组合各组件             | 需要精细控制流程的场景 | ★★★★★ | ★★★★☆ |
| **预制链** (02_rag_chain.py)   | 使用create_retrieval_chain等预制链 | 快速原型和标准场景     | ★★☆☆☆ | ★★☆☆☆ |

### 2.1 原生构建方式 (01_navie_rag.py)

```python
# 构建基础RAG链
chain = {
    "question": RunnablePassthrough(),
    "context": basic_retriever
} | prompt_template | ChatTongyi()
```

**优势**：

- 完全控制数据流和处理逻辑
- 灵活添加自定义组件
- 易于扩展和调试

**局限**：

- 代码量大，复杂度高
- 需要深入理解LangChain核心概念

### 2.2 预制链方式 (02_rag_chain.py)

```python
# 使用预制链快速构建
chain1 = create_stuff_documents_chain(model, prompt_template)
chain2 = create_retrieval_chain(retriever, chain1)
```

**优势**：

- 代码简洁，开发速度快
- 内置最佳实践
- 减少常见错误

**核心预制链**：

1. `create_stuff_documents_chain`：将检索到的文档注入提示
2. `create_retrieval_chain`：组合检索器和文档处理链

**局限**：

- 定制化能力有限
- 内部逻辑不够透明

## 三、文档处理阶段详解

### 3.1 文档加载策略

LangChain支持多种文档源，每种需特定依赖：

```python
# Web内容加载 (02_rag_chain.py)
loader = WebBaseLoader(
    web_path="https://www.gov.cn/...",
    bs_kwargs={"parse_only": bs4.SoupStrainer(id="UCAP-CONTENT")}
)

# Word文档加载 (01_navie_rag.py)
loader = Docx2txtLoader("../../../Data/人事管理流程.docx")
```

**关键文档加载器**：


| 加载器类型           | 适用场景 | 依赖包             | 特点              |
| -------------------- | -------- | ------------------ | ----------------- |
| `WebBaseLoader`      | 网页内容 | `bs4`, `requests`  | 支持HTML过滤      |
| `Docx2txtLoader`     | Word文档 | `docx2txt`         | 保留基本格式      |
| `PyPDFLoader`        | PDF文档  | `pymupdf`, `pypdf` | 处理扫描PDF能力弱 |
| `TextLoader`         | 纯文本   | 无                 | 简单高效          |
| `UnstructuredLoader` | 复杂格式 | `unstructured`     | 支持表格、图片等  |

**生产建议**：

- 对网页内容，使用`bs4.SoupStrainer`只提取主体内容
- 对大型文档，实现分块加载避免内存溢出
- 始终记录文档元数据（来源、时间等）便于追踪

### 3.2 文本分割核心技术

`RecursiveCharacterTextSplitter`是文档处理的核心组件：

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 切块大小：500字符
    chunk_overlap=50,     # 重叠大小：50字符
    separators=["\n\n", "\n", "。", "！", "..."]  # 分隔符优先级
)
```

**参数深度解析**：


| 参数              | 作用               | 最佳实践                    | 常见错误                         |
| ----------------- | ------------------ | --------------------------- | -------------------------------- |
| `chunk_size`      | 控制文本块大小     | 中文300-800，英文800-1500   | 过大稀释关键信息，过小丢失上下文 |
| `chunk_overlap`   | 块间重叠保证连贯性 | chunk_size的10-20%          | 为节省空间设为0，导致上下文断裂  |
| `separators`      | 分割优先级         | 按文档类型自定义            | 使用默认分隔符处理专业文档       |
| `length_function` | 长度计算方式       | 考虑用token计数替代字符计数 | 忽略不同语言字符编码差异         |

**高级分割策略**：

- **语义感知分割**：使用LLM识别段落边界
- **表格特殊处理**：将表格转为结构化文本
- **标题层次保留**：保留文档结构信息

### 3.3 向量嵌入与存储

#### 嵌入模型选型

```python
# 阿里云DashScope嵌入 (两文件共同使用)
llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)
```

**嵌入模型对比**：


| 模型类型     | 代表产品                           | 优势                 | 适用场景             | 成本      |
| ------------ | ---------------------------------- | -------------------- | -------------------- | --------- |
| **API型**    | OpenAI text-embedding-3, DashScope | 高质量，免维护       | 企业应用，高精度需求 | $/千token |
| **开源模型** | BGE, m3e, Voyage                   | 数据隐私，无调用限制 | 敏感数据，大规模部署 | 部署成本  |
| **轻量模型** | all-MiniLM-L6-v2                   | 低资源需求           | 边缘设备，简单任务   | 低        |

**选型建议**：

- 中文场景：优先考虑DashScope或BGE (BAAI/bge-large-zh-v1.5)
- 多语言场景：OpenAI text-embedding-3-large或BAAI/bge-m3
- 隐私敏感：HuggingFace开源模型+自托管
- 资源受限：all-MiniLM-L6-v2，牺牲精度换取速度

#### 向量数据库选型

```python
# Chroma使用示例
db = Chroma.from_documents(documents=documents, embedding=llm_embeddings)
```

**主流向量数据库对比**：


| 数据库       | 优势                 | 限制                   | 适用规模  | 部署复杂度 |
| ------------ | -------------------- | ---------------------- | --------- | ---------- |
| **Chroma**   | 简单易用，Python原生 | 无持久化，不支持高并发 | <10万向量 | ★☆☆☆☆ |
| **Qdrant**   | 性能优异，过滤能力强 | 需额外服务             | 10万-1亿  | ★★★☆☆ |
| **Milvus**   | 超大规模，分布式     | 资源消耗大             | 1亿+      | ★★★★★ |
| **Pinecone** | 全托管，自动扩展     | 闭源，成本高           | 任意规模  | ★☆☆☆☆ |
| **PGVector** | 与业务数据同库       | 性能受限               | <1000万   | ★★☆☆☆ |

**生产环境选型决策树**：

```
文档规模 < 10万?
├─ 是 → 团队技术能力弱? 
│  ├─ 是 → Pinecone (全托管)
│  └─ 否 → Qdrant (自托管，平衡性能/复杂度)
└─ 否 → 需要与业务数据关联?
   ├─ 是 → PGVector
   └─ 否 → Milvus/Zilliz Cloud
```

## 四、文档检索阶段高级策略

### 4.1 检索方法对比

```python
# 基础相似度搜索
basic_results = vector_store.similarity_search(query="员工晋升流程", k=3)

# 带分数的搜索 (01_navie_rag.py)
scored_results = vector_store.similarity_search_with_score(query="员工晋升流程", k=3)
```


| 检索方法                        | 特点               | 适用场景   | 返回内容         |
| ------------------------------- | ------------------ | ---------- | ---------------- |
| `similarity_search`             | 基础向量相似度     | 通用检索   | 文档列表         |
| `similarity_search_with_score`  | 含相似度分数       | 需质量控制 | (文档, 分数)元组 |
| `max_marginal_relevance_search` | 平衡相关性和多样性 | 多角度问题 | 多样化文档列表   |

### 4.2 检索器高级配置

```python
# 01_navie_rag.py中的多种检索器配置
threshold_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5}
)

mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.3}
)
```

**检索策略应用场景**：


| 策略           | 典型场景        | 配置建议             | 注意事项           |
| -------------- | --------------- | -------------------- | ------------------ |
| **标准相似度** | 事实型问答      | k=3-5                | 避免过多噪声       |
| **阈值过滤**   | 高精度问答      | 阈值0.4-0.6 (L2距离) | 阈值需根据模型调整 |
| **MMR算法**    | 开放性问题      | lambda_mult=0.2-0.8  | 过高降低相关性     |
| **元数据过滤** | 多租户/权限控制 | 提前设计元数据结构   | 过滤条件不宜过复杂 |

### 4.3 高级检索技术

1. **混合检索**：结合关键词和向量检索

   ```python
   from langchain.retrievers import EnsembleRetriever
   from langchain_community.retrievers import BM25Retriever

   # 创建关键词检索器
   keyword_retriever = BM25Retriever.from_documents(documents)
   keyword_retriever.k = 2

   # 创建向量检索器
   vector_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

   # 组合成混合检索器
   ensemble_retriever = EnsembleRetriever(
       retrievers=[keyword_retriever, vector_retriever],
       weights=[0.5, 0.5]
   )
   ```
2. **查询扩展**：使用LLM生成多个查询变体

   ```python
   from langchain.chains import LLMChain
   from langchain.prompts import PromptTemplate

   prompt = PromptTemplate.from_template(
       "生成3个与以下问题相关的搜索查询变体:\n{question}\n变体:"
   )
   query_expander = LLMChain(llm=llm, prompt=prompt)

   expanded_queries = query_expander.predict(question=user_query)
   ```

## 五、生成阶段与链构建

### 5.1 提示工程最佳实践

```python
# 02_rag_chain.py中的提示模板
system_prompt = """
您是问答任务的助理。使用以下的上下文来回答问题，
上下文：<{context}>
如果你不知道答案，不要其他渠道去获得答案，就说你不知道。
"""
```

**增强版提示模板**：

```
你是一个专业{domain}顾问，根据提供的{document_type}回答问题。
要求：
1. 仅使用提供的上下文回答，不要编造信息
2. 如果上下文不包含相关信息，明确说明"根据现有文档，我无法回答这个问题"
3. 用{language}回答，保持{style}风格
4. {special_instructions}

问题：{question}

相关文档：
{context}
```

### 5.2 LangChain核心链类型

除了文件中展示的链，LangChain 1.0还提供以下重要链：

1. **ConversationalRetrievalChain**：对话式RAG

   ```python
   from langchain.chains import ConversationalRetrievalChain

   qa_chain = ConversationalRetrievalChain.from_llm(
       llm=chat_model,
       retriever=retriever,
       return_source_documents=True
   )
   ```
2. **create_history_aware_retriever**：历史感知检索

   ```python
   from langchain.chains import create_history_aware_retriever

   retriever_chain = create_history_aware_retriever(
       llm,
       retriever,
       contextualize_q_prompt
   )
   ```
3. **create_qa_with_sources_chain**：带来源引用的问答

   ```python
   from langchain.chains.combine_documents import create_qa_with_sources_chain

   qa_with_sources_chain = create_qa_with_sources_chain(llm)
   ```
4. **ConstitutionalChain**：确保内容安全

   ```python
   from langchain.chains.constitutional_ai import ConstitutionalChain

   constitutional_chain = ConstitutionalChain.from_llm(
       llm,
       constitutional_principles=[
           ConstitutionalPrinciple(
               critique_request="确保回答尊重所有群体",
               revision_request="重写回答，使其尊重所有群体"
           )
       ]
   )
   ```

### 5.3 链组合与路由

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains import LLMRouterChain, ConversationChain

# 定义不同场景的提示
physics_template = """你是一位物理学专家...
Question: {input}"""

math_template = """你是一位数学专家...
Question: {input}"""

# 创建专业链
physics_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(physics_template))
math_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(math_template))

# 创建路由链
router_chain = LLMRouterChain.from_llm(llm, multi_prompt_router_template)

# 组合成多提示链
multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains={"physics": physics_chain, "math": math_chain},
    default_chain=ConversationChain(llm=llm),
    verbose=True
)
```

## 六、生产环境最佳实践

### 6.1 性能优化

1. **缓存策略**：

   ```python
   from langchain.cache import SQLiteCache

   # 启用LLM调用缓存
   langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

   # 检索结果缓存
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def cached_retrieval(query: str):
       return retriever.get_relevant_documents(query)
   ```
2. **批处理与异步**：

   ```python
   # 异步执行
   results = await asyncio.gather(
       chain.ainvoke(query1),
       chain.ainvoke(query2),
       chain.ainvoke(query3)
   )

   # 批量处理
   batch_results = chain.batch([query1, query2, query3])
   ```
3. **索引优化**：

   - 定期重建向量索引
   - 根据查询模式调整HNSW参数
   - 实现分层缓存（热点数据内存，长尾数据磁盘）

### 6.2安全与合规

1. **敏感信息处理**：

   ```python
   from presidio_analyzer import AnalyzerEngine
   from presidio_anonymizer import AnonymizerEngine

   # 文档预处理：去除PII
   def redact_pii(text):
       analyzer = AnalyzerEngine()
       anonymizer = AnonymizerEngine()
       results = analyzer.analyze(text=text, language="zh")
       return anonymizer.anonymize(text=text, analyzer_results=results).text
   ```
2. **权限控制**：

   ```python
   # 基于元数据的访问控制
   filtered_retriever = vector_store.as_retriever(
       search_kwargs={"filter": {"department": user_department}}
   )
   ```
3. **内容安全**：

   - 输出过滤层，防止有害内容
   - 关键领域启用人工审核
   - 完整审计日志，满足合规要求

### 6.3 部署架构建议

**小规模团队（<10万向量）**：

```
用户 → API网关 (FastAPI) → [应用服务]
    ↓
[向量数据库: Qdrant单节点] + [LLM API]
    ↓
[监控: Prometheus+Grafana]
```

**中大型系统（10万-1亿向量）**：

```
用户 → CDN/负载均衡 → [API层]
    ↓
[应用服务集群] ←→ [缓存: Redis]
    ↓      ↗
[向量数据库集群: Qdrant/Milvus] + [LLM网关]
    ↓
[异步处理] → [文档处理服务] → [对象存储]
    ↓
[监控/告警] + [评估平台]
```

**关键配置**：

- **水平扩展**：应用服务无状态设计，便于扩展
- **读写分离**：高频查询走缓存，更新操作异步处理
- **熔断机制**：LLM/向量DB故障时的降级策略
- **蓝绿部署**：知识库更新时零停机

###
