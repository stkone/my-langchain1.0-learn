"""
针对不同分块大小对结果的影响的评估
============================================================
RAG系统评估脚本 - 基于Ragas框架的端到端评估
============================================================

【脚本功能】
本脚本实现了一个完整的RAG系统评估流程，包括：
1. 文档加载与分块
2. 向量数据库构建与检索
3. RAG链构建与答案生成
4. 使用Ragas进行多维度评估

【核心评估指标】
- Context Precision: 检索精确率（检索到的文档中有多少是相关的）
- Context Recall: 检索召回率（相关文档有多少被检索到）
- Faithfulness: 忠实度（答案是否基于检索文档，无幻觉）
- Answer Relevancy: 回答相关性（答案与问题的匹配度）
- F1 Score: 综合检索质量的F1分数

【技术栈】
- LangChain: RAG链构建、文档处理、向量存储
- Ragas: RAG系统评估框架
- FAISS: 高效向量检索库
- DashScope: 阿里云大模型和Embedding服务
============================================================
"""

import os

from datasets import Dataset
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME

# ============================================================
# 阶段一：模型初始化
# ============================================================

model = ChatTongyi()

llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 使用阿里云DashScope的向量嵌入模型
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)  # 从环境变量获取API密钥
)

# 向量数据库实例，延迟初始化
vectordb = None

# ============================================================
# 阶段二：文档加载与预处理
# ============================================================

# 【步骤2.1】加载PDF文档
# PyPDFLoader会将PDF按页解析，每页生成一个Document对象
# Document结构：{page_content: "文本内容", metadata: {source: "文件路径", page: 页码}}
file_path = "../Data/领克汽车用户操作手册.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print("文档个数：", len(docs))

# 【步骤2.2】文档分块（Chunking）
# 【核心原理】分块策略直接影响RAG系统的检索质量和生成质量：
#
# ┌─────────────────────────────────────────────────────────────┐
# │  分块大小(chunk_size)的权衡：                                │
# │  • 小块(128-256): 检索精度高，但可能丢失上下文，召回率降低   │
# │  • 大块(512-1024): 上下文完整，但可能引入噪声，精确率降低    │
# │                                                             │
# │  重叠(chunk_overlap)的作用：                                │
# │  • 防止语义边界被切断，保证跨块信息的连续性                  │
# │  • 通常设置为chunk_size的10%-25%，这里使用20%                │
# └─────────────────────────────────────────────────────────────┘
#
# 【实验记录】不同chunk_size的评估结果对比：
# chunk_size=128:  faithfulness=0.9650, context_precision=0.9044, F1=0.8773
# chunk_size=256:  faithfulness=0.9697, context_precision=0.7775, F1=0.8295
# chunk_size=512:  faithfulness=0.9570, context_precision=0.7680, F1=0.8241
#
# 【结论】较小的chunk_size(128)在精确率上表现更好，但F1分数需结合召回率综合判断

chunk_size = 512
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=int(chunk_size * 0.20),  # 20%重叠，保证语义连续性
)
split_docs = text_splitter.split_documents(docs)
print("分块个数：", len(split_docs))

# ============================================================
# 阶段三：向量数据库构建
# ============================================================

# 【步骤3.1】构建或加载FAISS向量索引
# 【原理】FAISS(Facebook AI Similarity Search)是高效的向量相似度搜索库：
# - 将文本通过Embedding模型转换为向量
# - 使用倒排文件(IVF)或HNSW等索引结构加速最近邻搜索
# - 支持余弦相似度、欧氏距离等多种度量方式
#
# 【优化策略】索引持久化：
# - 首次运行时创建索引并保存到本地
# - 后续运行直接加载，避免重复计算Embedding

index_folder_path = "data/faiss_index"
index_name = "c_default_" + str(chunk_size)
index_file_path = os.path.join(index_folder_path, f"{index_name}.faiss")

if os.path.exists(index_file_path):
    print("索引文件已存在，直接加载...")
    vectordb = FAISS.load_local(
        index_folder_path,
        llm_embeddings,
        index_name,
        allow_dangerous_deserialization=True  # 允许加载本地序列化文件
    )
else:
    print("索引文件不存在，创建并保存索引...")
    # 【用法】FAISS.from_documents()：
    # - 自动调用Embedding模型将所有文档转换为向量
    # - 构建索引结构，支持高效检索
    vectordb = FAISS.from_documents(split_docs, llm_embeddings)
    vectordb.save_local(index_folder_path, index_name)
    print("向量化完成....")

# ============================================================
# 阶段四：评估数据集准备
# ============================================================

# 【步骤4.1】配置检索参数
topK_doc_count = 10  # 每次检索返回的文档数量

# 【步骤4.2】定义评估问题集
# 这些问题应该覆盖实际业务场景中的典型查询
questions = [
    "如何使用安全带？",
    "车辆如何保养？",
    "座椅太热怎么办？"
]

# 【步骤4.3】定义标准答案（Ground Truth）
# 【重要】ground_truth是评估的基准，应由领域专家编写
# Ragas通过对比answer与ground_truth来评估回答质量
ground_truths = [
    '''调节座椅到合适位置，缓慢拉出安全带，将锁舌插入锁扣中，直到听见“咔哒”声。
    使腰部安全带应尽可能低的横跨于胯部。确保肩部安全带斜跨整个肩部，穿过胸部。
    将前排座椅安全带高度调整至合适的位置。
    请勿将座椅靠背太过向后倾斜。
    请在系紧安全带前检查锁扣插口是否存在异物（如：食物残渣等），若存在异物请及时取出。
    为确保安全带正常工作，请务必将安全带插入与之匹配的锁扣中。
    乘坐时，安全带必须拉紧，防止松垮，并确保其牢固贴身，无扭曲。
    切勿将安全带从您的后背绕过、从您的胳膊下面绕过或绕过您的颈部。安全带应远离您的面部和颈部，但不得从肩部滑落。
    如果安全带无法正常使用，请联系Lynk & Co领克中心进行处理。''',
    "为了保持车辆处于最佳状态，建议您定期关注车辆状态，包括定期保养、洗车、内部清洁、外部清洁、轮胎的保养、低压蓄电池的保养等。",
    '''有三种方式：1、通过中央显示屏，设置座椅加热强度或关闭座椅加热功能，
    在中央显示屏中点击座椅进入座椅加热控制界面，可在“关-低-中-高”之间循环。
    2、登录Lynk & Co App，按下前排座椅加热图标图标可以打开/关闭前排座椅加热。
    3、在中央显示屏中唤起空调控制界面然后点击舒适选项，降低座椅加热时间。'''
]

# ============================================================
# 阶段五：RAG链构建
# ============================================================

# 【步骤5.1】设计系统提示词模板
# 【原理】system_prompt定义了LLM的行为规范：
# - 明确指定使用提供的上下文回答问题（防止幻觉）
# - 限制知识来源，禁止从预训练知识中回答
# - {context}占位符将在运行时被检索到的文档填充

system_prompt = """
您是问答任务的助理。使用以下的上下文来回答问题，
上下文：<{context}>
如果你不知道答案，不要其他渠道去获得答案，就说你不知道。
"""

# 【用法】ChatPromptTemplate.from_messages()：
# - 构建结构化的对话提示词
# - system消息设定角色和行为规范
# - human消息包含用户输入（{input}为占位符）
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# ============================================================
# 阶段六：评估器类封装
# ============================================================

class QAEvaluator:
    """
    RAG问答评估器

    封装了RAG链的构建、答案生成和Ragas评估的完整流程
    """

    def __init__(self, retriever):
        """
        初始化评估器

        【原理】LangChain的链式组合模式：
        ┌─────────────────────────────────────────────────────────────┐
        │  create_stuff_documents_chain:                              │
        │  • 将所有检索到的文档"塞"(stuff)进提示词的context中         │
        │  • 调用LLM生成回答                                          │
        │                                                             │
        │  create_retrieval_chain:                                    │
        │  • 协调检索和生成两个步骤                                   │
        │  • 输入: user_query → 检索 → 填充prompt → LLM → 输出      │
        └─────────────────────────────────────────────────────────────┘

        【用法】链的执行流程：
        1. retriever根据input检索相关文档
        2. document_chain将文档填充到prompt_template的{context}
        3. LLM根据填充后的prompt生成answer
        """
        # 创建文档处理链：负责将文档列表格式化为prompt并调用LLM
        document_chain = create_stuff_documents_chain(model, prompt_template)

        # 创建检索链：整合检索器和文档链
        # 执行顺序: input → retriever → document_chain → answer
        self.chain = create_retrieval_chain(retriever, document_chain)
        self.retriever = retriever

    def generate_answers(self, questions):
        """
        批量生成答案和检索上下文

        【步骤】
        1. 遍历每个问题，调用RAG链生成回答
        2. 收集生成的答案(answer)和检索到的文档(contexts)
        3. 这两个字段是Ragas评估的核心输入

        【返回值】
        - answers: List[str]，RAG生成的回答列表
        - contexts: List[List[str]]，每个问题对应的检索文档列表
        """
        answers = []
        contexts = []
        for question in questions:
            print("问题：", question)

            # 【用法】chain.invoke({"input": question})：
            # - 执行完整的RAG流程
            # - 返回dict包含：answer(生成回答), context(检索文档列表)
            response = self.chain.invoke({"input": question})

            print("大模型答复：", response["answer"], "\n")
            answers.append(response["answer"])

            # 提取检索到的文档内容，用于Ragas评估
            contexts.append([doc.page_content for doc in response["context"]])
            print("大模型回答时参考的上下文：", contexts, "\n")
            print("==" * 35)
        return answers, contexts

    def evaluate(self, questions, answers, contexts, ground_truths):
        """
        使用Ragas框架执行评估

        【原理】Ragas评估流程：
        ┌─────────────────────────────────────────────────────────────┐
        │  1. 构建评估数据集（包含question/answer/contexts/ground_truth）│
        │  2. 对每个样本，使用LLM作为评判员计算各指标分数               │
        │  3. 指标计算方式：                                          │
        │     • faithfulness: LLM判断answer中的每个陈述是否能在       │
        │       contexts中找到依据                                    │
        │     • answer_relevancy: 使用Embedding计算answer与question   │
        │       的语义相似度                                          │
        │     • context_precision: LLM判断检索到的文档是否相关        │
        │     • context_recall: LLM判断ground_truth中的信息是否被     │
        │       contexts覆盖                                          │
        └─────────────────────────────────────────────────────────────┘

        【参数】
        - questions: 问题列表
        - answers: RAG生成的回答列表
        - contexts: 检索到的文档片段（二维列表）
        - ground_truths: 标准答案列表
        """
        # 组装评估数据
        evaluate_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

        # 转换为Hugging Face Dataset格式
        evaluate_dataset = Dataset.from_dict(evaluate_data)

        # 【用法】ragas.evaluate()：
        # - dataset: 评估数据集
        # - llm: 用于评估的LLM（作为评判员）
        # - embeddings: 用于语义相似度计算的Embedding模型
        # - metrics: 要计算的评估指标列表
        evaluate_result = evaluate(
            evaluate_dataset,
            llm=model,
            embeddings=llm_embeddings,
            metrics=[
                faithfulness,       # 忠实度：答案是否基于文档
                answer_relevancy,   # 回答相关性
                context_recall,     # 上下文召回率
                context_precision,  # 上下文精确率
            ]
        )
        return evaluate_result


# ============================================================
# 阶段七：执行评估与结果分析
# ============================================================

def exec_eval(retriever):
    """
    执行完整评估流程

    【流程】
    1. 初始化QAEvaluator
    2. 生成答案和检索上下文
    3. 使用Ragas评估
    """
    qa_evaluator = QAEvaluator(retriever)
    answers, contexts = qa_evaluator.generate_answers(questions)
    return qa_evaluator.evaluate(questions, answers, contexts, ground_truths)


def calc_f1(evaluate_result):
    """
    计算检索质量的F1分数

    【原理】F1分数是精确率和召回率的调和平均：
    # ┌─────────────────────────────────────────────────────────────┐
    # │  F1 = 2 * (Precision * Recall) / (Precision + Recall)       │
    # │                                                             │
    # │  为什么使用F1？                                             │
    # │  • Precision高但Recall低：漏掉了相关文档                    │
    # │  • Recall高但Precision低：引入了太多噪声                    │
    # │  • F1平衡两者，综合反映检索质量                             │
    # └─────────────────────────────────────────────────────────────┘

    【参数】
    - evaluate_result: Ragas返回的评估结果字典

    【返回值】
    - F1分数（保留4位小数）
    """
    context_precisions = evaluate_result["context_precision"]
    context_recalls = evaluate_result["context_recall"]

    print("context_precisions=", context_precisions)
    print("context_recalls=", context_recalls)

    # 计算平均值
    context_precision_score = sum(context_precisions) / len(context_precisions)
    context_recall_score = sum(context_recalls) / len(context_recalls)

    # 计算F1分数
    f1_score = (2 * context_precision_score * context_recall_score) / (
        context_precision_score + context_recall_score
    )
    return round(f1_score, 4)


# ============================================================
# 主执行流程
# ============================================================

# 【步骤7.1】从向量数据库创建检索器
# 【用法】as_retriever(search_kwargs={"k": N})：
# - 将FAISS向量存储转换为LangChain Retriever接口
# - search_kwargs["k"]: 指定返回最相似的N个文档
faiss_retriever = vectordb.as_retriever(search_kwargs={"k": topK_doc_count})

# 【步骤7.2】执行评估
evaluate_result = exec_eval(faiss_retriever)

# 【步骤7.3】输出结果
print("chunk_size=[", chunk_size, "]评估结果：", evaluate_result,
      " ，f1分数：", calc_f1(evaluate_result))

# ============================================================
# 【结果解读指南】
# ┌─────────────────────────────────────────────────────────────┐
# │  faithfulness低 → 模型产生幻觉，回答包含文档外的信息        │
# │  answer_relevancy低 → 回答偏离问题核心                      │
# │  context_precision低 → 检索器返回了太多不相关文档           │
# │  context_recall低 → 检索器漏掉了关键信息                    │
# │  F1低 → 检索质量整体不佳，需优化分块策略或Embedding模型     │
# └─────────────────────────────────────────────────────────────┘
# ============================================================
