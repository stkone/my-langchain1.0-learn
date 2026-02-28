"""
RAGAS评估实验v3：基于LCEL + 重排序(Rerank)的进阶实现

【实验目标】
使用RAGAS框架评估重排序策略在智能汽车问答场景下的效果，
对比混合检索与重排序后检索的性能差异。

【核心流程】
1. 文档加载与分块 → 2. 向量索引构建/加载 → 3. 混合检索器配置 → 4. 重排序压缩 → 5. 问答生成 → 6. RAGAS评估

【与v2的核心差异】
- 引入重排序(Rerank)机制：
  * v2: 仅使用混合检索(EnsembleRetriever)的结果
  * v3: 在混合检索后增加DashScopeRerank重排序，提升结果质量
- 使用RAGAS的Wrapper类封装LLM和Embedding：
  * LangchainLLMWrapper: 包装LangChain LLM供RAGAS使用
  * LangchainEmbeddingsWrapper: 包装LangChain Embedding供RAGAS使用

【RAGAS评估指标说明】
1. Context Precision（上下文精确率）:
   - 定义：检索到的相关文档块占所有检索文档块的比例
   - 与"命中个数计算方式"的差异：RAGAS使用LLM判断每个文档块是否与ground_truth语义相关，
     而非简单的关键词匹配；适用于评估检索质量而非仅仅是命中数量
   - 典型值：0.48~0.90（重排序后通常能提升precision）

2. Context Recall（上下文召回率）:
   - 定义：ground_truth中能被检索到的信息比例，别名hit_rate
   - 计算方式：RAGAS通过LLM分析ground_truth的每个陈述句，判断其是否被检索到的上下文覆盖
   - 典型值：0.70~1.00（理想情况下应接近1.00）

3. Faithfulness（忠实度）:
   - 定义：答案是否能从检索到的上下文中推理得出
   - 注意：不评估答案正确性，只评估答案与上下文的一致性
   - 典型值：0.96~0.98

4. Answer Relevancy（答案相关性）:
   - 定义：答案与问题的相关程度
   - 注意：不评估答案正确性，只评估是否回答了问题
   - 典型值：0.82~0.92
"""
import os
import time

# ============================================
# 【第1步】导入LCEL新版链式API组件
# ============================================
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# LLMChainExtractor: 基于LLM的文档压缩器（v3中未使用但保留导入）
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from langchain_community.chat_models import ChatTongyi

# DashScopeRerank: 通义千问重排序模型
# 原理：使用Cross-Encoder架构对query和document进行联合编码，计算相关性得分
# 与Bi-Encoder的区别：Cross-Encoder能捕捉query和doc的交互信息，排序更准确
# 用法：作为ContextualCompressionRetriever的base_compressor使用
from langchain_community.document_compressors import DashScopeRerank

from langchain_community.document_loaders import PyPDFLoader

# EnsembleRetriever: 混合检索器
# ContextualCompressionRetriever: 上下文压缩检索器（v3中用于重排序）
# MultiVectorRetriever: 多向量检索器（v3中导入但未使用）
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever, \
    ContextualCompressionRetriever, MultiVectorRetriever

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ============================================
# 【第2步】导入RAGAS评估框架及Wrapper类
# ============================================
from datasets import Dataset
from langchain_core.prompts import ChatPromptTemplate
from ragas import evaluate

# RAGAS Wrapper类：将LangChain组件包装为RAGAS可用的格式
# 原理：RAGAS需要特定接口的LLM和Embedding，Wrapper提供适配层
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from ragas.metrics import (
    faithfulness,        # 忠实度：答案是否可从上下文中推理得出
    answer_relevancy,    # 答案相关性：答案是否与问题相关
    context_recall,      # 上下文召回率：ground_truth中有多少被检索到
    context_precision,   # 上下文精确率：检索结果中有多少是相关的
)

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME, ALI_TONGYI_EMBEDDING

# ============================================
# 【第3步】初始化LLM和Embedding模型及RAGAS包装器
# ============================================
# ChatTongyi: 通义千问对话模型封装
model = ChatTongyi(
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)

# DashScopeEmbeddings: 通义千问向量嵌入模型封装
llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 使用最新的向量嵌入模型
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)  # 从环境变量获取API密钥
)

# LangchainLLMWrapper: 将LangChain LLM包装为RAGAS可用的格式
# 用法：RAGAS内部调用此wrapper与LLM交互进行指标计算
vllm = LangchainLLMWrapper(model)

# LangchainEmbeddingsWrapper: 将LangChain Embedding包装为RAGAS可用的格式
# 用法：RAGAS内部调用此wrapper计算语义相似度（用于answer_relevancy）
vllm_e = LangchainEmbeddingsWrapper(llm_embeddings)

# ============================================
# 【第4步】文档加载与分块
# ============================================
vectordb = None

# 加载PDF文档
file_path = "../data/领克汽车用户操作手册.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print("文档个数：", len(docs))

# 文档分块策略：RecursiveCharacterTextSplitter
# 原理：递归地按字符分割文本，优先按段落、句子、单词的顺序尝试分割
# 参数说明：
#   - chunk_size=128: 每个分块的目标字符数
#   - chunk_overlap=int(chunk_size * 0.20): 相邻分块的重叠字符数（20%重叠）
chunk_size = 128
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=int(chunk_size * 0.20),
)
split_docs = text_splitter.split_documents(docs)
print("分块个数：", len(split_docs))

# ============================================
# 【第5步】向量索引构建或加载
# ============================================
# FAISS: Facebook开源的向量相似度搜索库
# 原理：将文档embedding后构建索引，支持高效的近似最近邻(ANN)搜索
# 注意：v3中使用vllm_e（LangchainEmbeddingsWrapper）加载索引，与v2略有不同
index_folder_path = "data/faiss_index"
index_name = "c_default_"+str(chunk_size)
index_file_path = os.path.join(index_folder_path, f"{index_name}.faiss")

# 检查索引文件是否存在，避免重复构建
if os.path.exists(index_file_path):
    print("索引文件已存在，直接加载...")
    # 使用vllm_e（LangchainEmbeddingsWrapper）加载索引
    vectordb = FAISS.load_local(index_folder_path, vllm_e, index_name, allow_dangerous_deserialization=True)
else:
    print("索引文件不存在，创建并保存索引...")
    # 创建向量存储：将分块文档转为embedding并构建FAISS索引
    vectordb = FAISS.from_documents(split_docs, llm_embeddings)
    # 保存索引到本地，便于下次快速加载
    vectordb.save_local(index_folder_path, index_name)
    print("向量化完成....")


# ============================================
# 【第6步】实验结果记录与参数配置
# ============================================
# 历史实验结果记录（topK=10，chunk_size=128时）：
# - mix_retriever<0.2,0.8>: f1=0.6497, recall=1.0000, precision=0.4811（高召回低精度）
# - faiss_retriever: f1=0.8773, recall=0.8519, precision=0.9044（最佳平衡）
# - compression_retriever(LLMChainExtractor): f1=0.7513, recall=0.7037, precision=0.8057
#
# v3目标：通过重排序(Rerank)进一步提升precision，同时保持较高的recall
# 预期效果：重排序能过滤掉低质量检索结果，提升context_precision
topK_doc_count = 10

# ============================================
# 【第7步】测试问题与标准答案定义
# ============================================
# 测试集：3个典型用户问题，覆盖车辆使用、保养、功能操作等场景
# ground_truths: 人工编写的标准答案，用于RAGAS评估计算recall和precision
questions = ["如何使用安全带？", "车辆如何保养？", "座椅太热怎么办？"]
ground_truths = [
    '''调节座椅到合适位置，缓慢拉出安全带，将锁舌插入锁扣中，直到听见"咔哒"声。
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
    在中央显示屏中点击座椅进入座椅加热控制界面，可在"关-低-中-高"之间循环。
    2、登录Lynk & Co App，按下前排座椅加热图标图标可以打开/关闭前排座椅加热。
    3、在中央显示屏中唤起空调控制界面然后点击舒适选项，降低座椅加热时间。'''
]

# ============================================
# 【第8步】自定义Prompt模板
# ============================================
# ChatPromptTemplate: 支持多角色消息定义
#   - system: 系统指令，定义助手行为
#   - human: 用户输入
# 占位符说明：
#   - {context}: 将被检索到的文档内容填充
#   - {input}: 用户问题
system_prompt = """
您是问答任务的助理。使用以下的上下文来回答问题，
上下文：<{context}>
如果你不知道答案，不要其他渠道去获得答案，就说你不知道。
"""
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)
# ============================================
# 【第9步】问答评估器类定义
# ============================================
class QAEvaluator:
    """
    RAG问答评估器：封装问答生成和RAGAS评估流程
    
    使用LCEL链式API：
    - create_stuff_documents_chain: 创建文档处理链
    - create_retrieval_chain: 创建检索链（组合检索器和文档链）
    """
    def __init__(self, retriever):
        # 创建文档处理链：将检索到的文档填充到prompt的{context}位置
        document_chain = create_stuff_documents_chain(model, prompt_template)
        
        # 创建检索链：组合检索器和文档处理链
        # 注意：v3中的retriever可以是普通检索器或重排序后的压缩检索器
        self.chain = create_retrieval_chain(retriever,document_chain)
        self.retriever = retriever

    def generate_answers(self, questions):
        """
        生成答案并收集上下文
        
        流程：
        1. 对每个问题调用检索链
        2. 提取生成的答案和检索到的上下文
        3. 收集所有结果用于后续RAGAS评估
        
        返回：
        - answers: LLM生成的答案列表
        - contexts: 每个问题对应的检索文档内容列表（二维列表）
        """
        answers = []
        contexts = []
        for question in questions:
            print("问题：", question)
            response = self.chain.invoke({"input": question})
            print("大模型答复：", response["answer"], "\n")
            answers.append(response["answer"])
            # 获取上下文：LCEL返回的字段名为"context"
            contexts.append([doc.page_content for doc in response["context"]])
            print("大模型回答时参考的上下文：", contexts, "\n")
            print("=="*35)
        return answers, contexts

    def evaluate(self, questions, answers, contexts, ground_truths):
        """
        使用RAGAS框架执行评估
        
        与v2的差异：
        - v2直接使用model和llm_embeddings
        - v3使用vllm和vllm_e（RAGAS Wrapper类包装后的版本）
        
        评估数据格式要求：
        - question: 问题列表
        - answer: 生成的答案列表
        - contexts: 检索到的上下文列表（每个元素是文档字符串列表）
        - ground_truth: 标准答案列表
        """
        # 构建评估数据集（HuggingFace datasets格式）
        evaluate_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        evaluate_dataset = Dataset.from_dict(evaluate_data)
        
        # 执行RAGAS评估
        # 使用Wrapper包装后的LLM和Embedding
        evaluate_result = evaluate(
            evaluate_dataset,
            llm=vllm,            # LangchainLLMWrapper包装后的LLM
            embeddings=vllm_e,   # LangchainEmbeddingsWrapper包装后的Embedding
            metrics=[
                faithfulness,        # 忠实度
                answer_relevancy,    # 答案相关性
                context_recall,      # 上下文召回率
                context_precision,   # 上下文精确率
            ]
        )
        return evaluate_result

def exec_eval(retriever):
    """
    执行完整评估流程：问答生成 + RAGAS评估
    
    参数：
    - retriever: 文档检索器（可以是普通检索器或重排序后的压缩检索器）
    
    返回：
    - RAGAS评估结果（包含各指标得分）
    """
    qa_evaluator = QAEvaluator(retriever)
    answers, contexts = qa_evaluator.generate_answers(questions)
    return qa_evaluator.evaluate(questions, answers, contexts, ground_truths)

def calc_f1(evaluate_result):
    """
    计算F1分数（基于Context Precision和Context Recall）
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    说明：
    - F1分数综合考量检索的精确率和召回率
    - 当Precision和Recall都较高时，F1才会高
    - 用于权衡检索质量，避免单一指标的偏差
    """
    context_precisions = evaluate_result["context_precision"]
    context_recalls = evaluate_result["context_recall"]
    print("context_precisions=",context_precisions)
    print("context_recalls=",context_recalls)
    # 计算平均值
    context_precision_score = sum(context_precisions) / len(context_precisions)
    context_recall_score = sum(context_recalls) / len(context_recalls)
    # 计算F1分数
    f1_score = (2 * context_precision_score * context_recall_score) / (context_precision_score + context_recall_score)
    return round(f1_score,4)

# ============================================
# 【第10步】重排序检索器配置与评估
# ============================================

# --------------------------------------------
# 基础检索器配置
# --------------------------------------------
# 创建向量检索器：基于向量相似度检索语义相关的文档
faiss_retriever = vectordb.as_retriever(search_kwargs={"k": topK_doc_count})

# 创建全文检索器：基于BM25算法的关键词匹配
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k=topK_doc_count

# 创建混合检索器：结合BM25和FAISS的优势
# 参数说明：
#   - retrievers: [bm25_retriever, faiss_retriever]
#   - weight: [0.2, 0.8]（BM25权重0.2，FAISS权重0.8）
mix_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],weight=[0.2, 0.8])

# --------------------------------------------
# 重排序模型配置（v3核心特性）
# --------------------------------------------
# DashScopeRerank: 通义千问重排序模型
# 原理（Cross-Encoder）：
#   - 将query和document拼接为[CLS] query [SEP] doc [SEP]格式
#   - 通过Transformer联合编码，捕捉query和doc的交互信息
#   - 输出相关性得分，用于重新排序检索结果
# 与Bi-Encoder的区别：
#   - Bi-Encoder：分别编码query和doc，计算向量相似度（速度快但精度较低）
#   - Cross-Encoder：联合编码query和doc（精度高但速度较慢）
# 参数说明：
#   - model: 重排序模型名称
#   - dashscope_api_key: API密钥
#   - top_n: 最终返回的文档数量
rerank_retriever = DashScopeRerank(
        model=ALI_TONGYI_EMBEDDING, 
        dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
        top_n=10
)

# --------------------------------------------
# 上下文压缩检索器（用于重排序）
# --------------------------------------------
# ContextualCompressionRetriever: 上下文压缩检索器
# 在v3中的用法：
#   - base_retriever: 混合检索器（先召回候选文档）
#   - base_compressor: 重排序模型（对候选文档重新排序并筛选）
# 工作流程：
#   1. base_retriever检索初始文档（如top 20）
#   2. base_compressor（重排序模型）对文档重新打分
#   3. 返回top_n个最相关的文档
combin_retriever = ContextualCompressionRetriever(
    base_compressor=rerank_retriever, 
    base_retriever=mix_retriever
)

# --------------------------------------------
# 执行评估
# --------------------------------------------
mix_evaluate_result = exec_eval(combin_retriever)
print("重排序后mix_retriever评估结果：", mix_evaluate_result," ，f1分数：",calc_f1(mix_evaluate_result))
