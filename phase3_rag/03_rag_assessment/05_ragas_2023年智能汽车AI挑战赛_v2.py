"""
RAGAS评估实验v2：基于LCEL（LangChain Expression Language）的新版链式API实现

【实验目标】
使用RAGAS框架评估不同检索策略在智能汽车问答场景下的效果，
对比FAISS向量检索、BM25全文检索、混合检索的性能差异。

【核心流程】
1. 文档加载与分块 → 2. 向量索引构建/加载 → 3. 多检索器配置 → 4. 问答生成 → 5. RAGAS评估

【与v1的核心差异】
- 使用LCEL新API替代经典RetrievalQA：
  * v1: RetrievalQA.from_chain_type（封装度高但灵活性低）
  * v2: create_stuff_documents_chain + create_retrieval_chain（组合式，更灵活）
- 支持自定义prompt模板，可精细控制问答行为

【RAGAS评估指标说明】
1. Context Precision（上下文精确率）:
   - 定义：检索到的相关文档块占所有检索文档块的比例
   - 与"命中个数计算方式"的差异：RAGAS使用LLM判断每个文档块是否与ground_truth语义相关，
     而非简单的关键词匹配；适用于评估检索质量而非仅仅是命中数量
   - 典型值：0.48~0.90（高召回时往往较低，如0.48；高精度时可达0.90）

2. Context Recall（上下文召回率）:
   - 定义：ground_truth中能被检索到的信息比例，别名hit_rate
   - 计算方式：RAGAS通过LLM分析ground_truth的每个陈述句，判断其是否被检索到的上下文覆盖
   - 典型值：0.70~1.00（理想情况下应接近1.00）

3. Faithfulness（忠实度）:
   - 定义：答案是否能从检索到的上下文中推理得出
   - 注意：不评估答案正确性，只评估答案与上下文的一致性
   - 典型值：0.96~1.00

4. Answer Relevancy（答案相关性）:
   - 定义：答案与问题的相关程度
   - 注意：不评估答案正确性，只评估是否回答了问题
   - 典型值：0.82~0.92
"""
import os
import time

from datasets import Dataset

# ============================================
# 【第1步】导入LCEL新版链式API组件
# ============================================
# LCEL（LangChain Expression Language）是LangChain的新版链式API
# 特点：使用管道操作符(|)组合组件，更灵活、可组合、可观测

# create_stuff_documents_chain: 创建文档处理链
# 原理：将所有检索到的文档"塞"(stuff)进prompt的context占位符中
# 用法：传入LLM和prompt模板，返回一个可执行的Runnable
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# create_retrieval_chain: 创建检索链
# 原理：将检索器与文档处理链组合，自动处理检索→格式化→生成的完整流程
# 与v1的区别：v1的RetrievalQA是黑盒封装，v2的create_retrieval_chain是透明组合
from langchain_classic.chains.retrieval import create_retrieval_chain

# EnsembleRetriever: 混合检索器，支持多路检索结果融合
# 原理：同时调用多个子检索器，按权重加权融合得分，默认使用加权融合而非RRF
from langchain_classic.retrievers import EnsembleRetriever

from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings

# BM25Retriever: 基于BM25算法的全文检索器
# 原理：计算query与文档的词项匹配得分，擅长精确关键词匹配
from langchain_community.retrievers import BM25Retriever

from langchain_community.vectorstores import FAISS

# ChatPromptTemplate: 对话式prompt模板
# 用法：支持system/human/ai等多角色消息定义，更灵活地控制对话行为
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================
# 【第2步】导入RAGAS评估框架
# ============================================
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # 忠实度：答案是否可从上下文中推理得出
    answer_relevancy,    # 答案相关性：答案是否与问题相关
    context_recall,      # 上下文召回率：ground_truth中有多少被检索到
    context_precision,   # 上下文精确率：检索结果中有多少是相关的
)

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME

# ============================================
# 【第3步】初始化LLM和Embedding模型
# ============================================
# ChatTongyi: 通义千问对话模型封装
# 用法：直接调用invoke/generate等方法进行对话
model = ChatTongyi(
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)

# DashScopeEmbeddings: 通义千问向量嵌入模型封装
# 用法：调用embed_query/embed_documents将文本转为向量
# 原理：将文本映射到高维向量空间，语义相似的文本向量距离更近
llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 使用最新的向量嵌入模型
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)  # 从环境变量获取API密钥
)


# ============================================
# 【第4步】文档加载与分块
# ============================================
vectordb = None

# 加载PDF文档
file_path = "../Data/领克汽车用户操作手册.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print("文档个数：", len(docs))

# 文档分块策略：RecursiveCharacterTextSplitter
# 原理：递归地按字符分割文本，优先按段落、句子、单词的顺序尝试分割
# 参数说明：
#   - chunk_size=128: 每个分块的目标字符数（v2使用更小的分块，粒度更细）
#   - chunk_overlap=int(chunk_size * 0.20): 相邻分块的重叠字符数（20%重叠）
# 与v1的区别：v1使用chunk_size=256，v2使用128，更细粒度可能提高检索精度
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
# 用法：
#   - from_documents: 从文档创建新索引
#   - load_local: 从本地加载已有索引
#   - save_local: 将索引保存到本地
index_folder_path = "../Data/faiss_index"
index_name = "c_default_"+str(chunk_size)  # 根据chunk_size命名索引，便于区分
index_file_path = os.path.join(index_folder_path, f"{index_name}.faiss")

# 检查索引文件是否存在，避免重复构建
if os.path.exists(index_file_path):
    print("索引文件已存在，直接加载...")
    # allow_dangerous_deserialization=True: 允许加载pickle序列化的索引（生产环境需谨慎）
    vectordb = FAISS.load_local(index_folder_path, llm_embeddings, index_name, allow_dangerous_deserialization=True)
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
# - compression_retriever: f1=0.7513, recall=0.7037, precision=0.8057
#
# topK=4实验（对比不同topK的影响）：
# - faiss_retriever: f1=0.875, recall=0.7778, precision=1.0000（高精度但召回略降）
# - mix_retriever: f1=0.619, recall=0.8519, precision=0.4861
#
# 结论：
# 1. 纯FAISS向量检索在本场景下表现最佳，f1分数最高
# 2. 混合检索虽然recall高，但precision较低，拉低了f1
# 3. 较小的chunk_size(128)相比v1的256，能提供更细粒度的检索
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
# 【第8步】自定义Prompt模板（LCEL特性）
# ============================================
# 与v1的核心差异：v1使用默认prompt，v2可自定义prompt模板
# ChatPromptTemplate: 支持多角色消息定义
#   - system: 系统指令，定义助手行为
#   - human: 用户输入
# 占位符说明：
#   - {context}: 将被检索到的文档内容填充（由create_stuff_documents_chain处理）
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
# 【第9步】问答评估器类定义（LCEL版本）
# ============================================
class QAEvaluator:
    """
    RAG问答评估器：封装问答生成和RAGAS评估流程（LCEL版本）
    
    与v1的核心差异：
    - v1使用RetrievalQA（黑盒封装）
    - v2使用create_stuff_documents_chain + create_retrieval_chain（透明组合）
    
    LCEL链式调用流程：
    1. create_stuff_documents_chain: 创建文档处理链（LLM + Prompt）
    2. create_retrieval_chain: 将检索器与文档链组合为完整RAG链
    3. 执行时：query → 检索文档 → 填充prompt → LLM生成
    """
    def __init__(self, retriever):
        # 创建文档处理链：将检索到的文档填充到prompt的{context}位置
        # 参数说明：
        #   - model: 用于生成答案的LLM
        #   - prompt_template: 包含{context}和{input}占位符的prompt模板
        document_chain = create_stuff_documents_chain(model, prompt_template)
        
        # 创建检索链：组合检索器和文档处理链
        # 原理：
        #   1. 接收input（用户问题）
        #   2. 调用retriever检索相关文档
        #   3. 将文档传递给document_chain生成答案
        #   4. 返回答案和检索到的上下文
        self.chain = create_retrieval_chain(retriever,document_chain)
        self.retriever = retriever

    def generate_answers(self, questions):
        """
        生成答案并收集上下文
        
        流程：
        1. 对每个问题调用检索链
        2. 提取生成的答案和检索到的上下文
        3. 收集所有结果用于后续RAGAS评估
        
        与v1的差异：
        - v1: response['result'] + response['source_documents']
        - v2: response["answer"] + response["context"]（LCEL标准字段名）
        
        返回：
        - answers: LLM生成的答案列表
        - contexts: 每个问题对应的检索文档内容列表（二维列表）
        """
        answers = []
        contexts = []
        for question in questions:
            print("问题：", question)
            # invoke: 执行检索链
            # 输入格式：{"input": question}（与prompt模板中的{input}对应）
            response = self.chain.invoke({"input": question})
            print("大模型答复：", response["answer"], "\n")
            answers.append(response["answer"])
            # 获取上下文：LCEL返回的字段名为"context"（v1为"source_documents"）
            contexts.append([doc.page_content for doc in response["context"]])
            print("大模型回答时参考的上下文：", contexts, "\n")
            print("=="*35)
        return answers, contexts

    def evaluate(self, questions, answers, contexts, ground_truths):
        """
        使用RAGAS框架执行评估
        
        评估数据格式要求：
        - question: 问题列表
        - answer: 生成的答案列表
        - contexts: 检索到的上下文列表（每个元素是文档字符串列表）
        - ground_truth: 标准答案列表
        
        RAGAS评估原理：
        1. 使用指定的LLM作为评判器（非生成答案的LLM）
        2. 使用embedding模型计算语义相似度（用于answer_relevancy）
        3. 对每个指标独立计算，最终返回各指标的平均值
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
        # 参数说明：
        #   - evaluate_dataset: 评估数据集
        #   - llm: 用于评估的LLM（RAGAS内部使用）
        #   - embeddings: 用于语义相似度计算的embedding模型
        #   - metrics: 要计算的评估指标列表
        evaluate_result = evaluate(
            evaluate_dataset,
            llm=model,
            embeddings=llm_embeddings,
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
    - retriever: 文档检索器（FAISS、BM25、Ensemble等）
    
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
# 【第10步】多检索器效果对比实验
# ============================================

# --------------------------------------------
# 实验1：FAISS向量检索器
# --------------------------------------------
# 原理：基于向量相似度（余弦相似度）检索语义相关的文档
# 优势：擅长理解语义，能召回语义相关但关键词不同的文档
# 用法：vectordb.as_retriever(search_kwargs={"k": n}) 返回top-n个文档
faiss_retriever = vectordb.as_retriever(search_kwargs={"k": topK_doc_count})
evaluate_result = exec_eval(faiss_retriever)
print("faiss_retriever评估结果：", evaluate_result," ，f1分数：",calc_f1(evaluate_result))
time.sleep(60)  # 暂停避免API限流

# --------------------------------------------
# 实验2：BM25全文检索器
# --------------------------------------------
# 原理：基于BM25算法计算query与文档的词项匹配得分
# 优势：擅长精确关键词匹配，对专业术语效果好
# 用法：from_documents创建后设置k值控制返回数量
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k=topK_doc_count

# --------------------------------------------
# 实验3：混合检索器（EnsembleRetriever）
# --------------------------------------------
# 原理：同时调用多个子检索器，按权重加权融合得分
# 参数说明：
#   - retrievers: 子检索器列表
#   - weight: 各检索器的权重，默认加权融合（非RRF）
# 优势：结合向量检索的语义理解和BM25的关键词匹配
# 实验配置：BM25权重0.2，FAISS权重0.8（更依赖语义检索）
mix_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],weight=[0.2, 0.8])
mix_evaluate_result = exec_eval(mix_retriever)
print("mix_retriever评估结果：", mix_evaluate_result," ，f1分数：",calc_f1(mix_evaluate_result))

# --------------------------------------------
# 实验4：上下文压缩检索器（已注释）
# --------------------------------------------
# 注：v2版本中压缩检索器实验被注释，v3中引入重排序替代
# LLMChainExtractor: 基于LLM的文档压缩器
# 原理：对每个检索到的文档，调用LLM提取与query相关的片段
# 风险：每个文档都会调用一次LLM，token消耗较大
# compressor = LLMChainExtractor.from_llm(model)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=mix_retriever
# )
# compression_evaluate_result = exec_eval(compression_retriever)