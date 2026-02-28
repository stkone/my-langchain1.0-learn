
# my-langchain1.0-learn

本工程是一个系统化的 LangChain 1.0 学习项目，涵盖从基础入门到 RAG 进阶的完整知识体系。通过循序渐进的三个阶段学习，帮助开发者掌握大语言模型应用开发的核心技能。

## 项目简介

### 学习目标
- **Phase 1 - 基础入门**：掌握 LangChain 核心概念，包括模型调用、提示词模板、输出解析、链式调用、消息历史等基础知识
- **Phase 2 - 核心知识**：深入理解 Agent 智能体、工具调用、记忆管理和中间件机制
- **Phase 3 - RAG 进阶**：全面学习检索增强生成技术，从基础实现到高级优化策略，以及 RAG 系统评估方法

### 技术栈
- **LangChain 1.0** - 大语言模型应用开发框架
- **Python 3.x** - 编程语言
- **Chroma** - 向量数据库
- **RAGAS** - RAG 系统评估框架

### 项目特点
- 按阶段组织，循序渐进
- 每个章节配有完整示例代码和总结文档
- 包含实战项目：金融助手 RAG 应用
- 涵盖多种 RAG 高级技术：混合搜索、查询分解、重排序等

## 目录详情

### 1. phase1_basic - 基础入门
- **01_frist_program** - 第一个程序
  - `01_model_init.py` - 模型初始化
  - `02_prompt_template.py` - 提示词模板
  - `03_output_parser.py` - 输出解析器
  - `summary.md` - 章节总结
- **02_langchain_basic** - LangChain 基础
  - `01_runnableSequence.py` - RunnableSequence 序列
  - `02_runnableSequence.py` - RunnableSequence 进阶
  - `03_stream.py` - 流式输出
  - `04_batch.py` - 批量处理
  - `summary.md` - 章节总结
- **03_langchain_lamdba** - Lambda 表达式
  - `01_runnable_lambda.py` - Runnable Lambda
  - `02_runnable_parallel.py` - 并行执行
  - `03_runnable_passthrough.py` - 透传处理
  - `summary.md` - 章节总结
- **04_message_history** - 消息历史
  - `01_no_memory.py` - 无记忆对话
  - `02_chat_message_history.py` - 聊天消息历史
  - `03_runnable_with_message_history.py` - 带历史记录的 Runnable
  - `summary.md` - 章节总结
- **05_project_demo** - 项目演示
  - `01_project_demo1.py` - 项目示例1
  - `summary.md` - 章节总结

### 2. phase2_core - 核心知识
- **01_agent_tools** - Agent 与工具
  - `01_agent.py` - Agent 基础
  - `02_agent_with_tools.py` - 带工具的 Agent
  - `03_agent_with_tools.py` - 工具调用进阶
  - `summary.md` - 章节总结
- **02_agent_memory** - Agent 记忆
  - `01_agent_memory.py` - Agent 记忆基础
  - `02_enrich_memory.py` - 丰富记忆功能
  - `summary.md` - 章节总结
- **03_middleware_basics** - 中间件基础
  - `01_middleWare_basics.py` - 中间件基础1
  - `02_middleWare_basics.py` - 中间件基础2
  - `03_middleWare_example.py` - 中间件示例1
  - `04_middleWare_example.py` - 中间件示例2
  - `summary.md` - 章节总结

### 3. phase3_rag - RAG 进阶
- **01_naive_rag** - 基础 RAG
  - **01_search** - 搜索模块
    - `01_bm25.py` - BM25 搜索算法
    - `02_hybrid.py` - 混合搜索
    - `summary.md` - 章节总结
  - **02_rag** - RAG 实现
    - `01_navie_rag.py` - 基础 RAG 实现
    - `02_rag_chain.py` - RAG 链式调用
    - `summary.md` - 章节总结
- **02_advanced_rag** - 高级 RAG 技术
  - `01_rag_summary.py` - RAG 总结
  - `02_rag_parent_child.py` - 父子文档检索
  - `03_rag_pre_questions.py` - 预设问题优化
  - `04_rag_self_query.py` - 自查询检索
  - `05_rag_multi_query.py` - 多查询检索
  - `06_rag_decomposition.py` - 查询分解
  - `07_rag_hybrid_search.py` - 混合搜索策略
  - `08_rag_contextual_compression.py` - 上下文压缩
  - `09_rag_fusion.py` - 融合检索
  - `10_rag_rerank.py` - 重排序优化
  - `summary.md` - 章节总结
- **03_rag_assessment** - RAG 评估
  - `01_rags_assessment.py` - RAGAS 评估基础
  - `02_rags_assessment.py` - RAGAS 评估进阶
  - `03_ragas_precision.py` - RAGAS 精确度评估
  - `04_ragas_2023年智能汽车AI挑战赛_v1.py` - 竞赛评估 v1
  - `05_ragas_2023年智能汽车AI挑战赛_v2.py` - 竞赛评估 v2
  - `06_ragas_2023年智能汽车AI挑战赛_v3.py` - 竞赛评估 v3
  - `summary.md` - 章节总结
- **04_rag_project** - RAG 实战项目
  - **01_financial_assistant** - 金融助手项目
    - `financial_assistant.py` - 金融助手主程序
    - `summary.md` - 项目总结
    - `my_chroma_db/` - Chroma 向量数据库
    - `my_pdfs/` - PDF 文档目录
    - `processed/` - 处理后数据目录
  - `requirements.txt` - 项目依赖
- **Data** - 数据文件
  - `deepseek百度百科.txt` - DeepSeek 百科资料
  - `train.json` - 训练数据
  - `人事管理流程.docx` - 人事管理流程文档
  - `采购招标文件.docx` - 采购招标文件
  - `领克汽车用户操作手册.pdf` - 汽车用户手册
