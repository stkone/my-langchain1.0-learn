"""
RAGAS评估实验v1：基于RetrievalQA的经典链式API实现

【实验目标】
使用RAGAS框架评估不同检索策略在智能汽车问答场景下的效果，
对比FAISS向量检索、BM25全文检索、混合检索及上下文压缩检索的性能差异。

【核心流程】
1. 文档加载与分块 → 2. 向量索引构建/加载 → 3. 多检索器配置 → 4. 问答生成 → 5. RAGAS评估

【技术特点】
- 使用LangChain经典的RetrievalQA链式API（较旧版本）
- 支持多种检索器：FAISS、BM25、Ensemble混合检索、ContextualCompression压缩检索
- 评估指标：Faithfulness、Answer Relevancy、Context Recall、Context Precision

【RAGAS评估指标说明】
1. Context Precision（上下文精确率）:
   - 定义：检索到的相关文档块占所有检索文档块的比例
   - 与"命中个数计算方式"的差异：RAGAS使用LLM判断每个文档块是否与ground_truth语义相关，
     而非简单的关键词匹配；适用于评估检索质量而非仅仅是命中数量
   - 典型值：0.38~0.90（高召回时往往较低，如0.38；高精度时可达0.90）

2. Context Recall（上下文召回率）:
   - 定义：ground_truth中能被检索到的信息比例，别名hit_rate
   - 计算方式：RAGAS通过LLM分析ground_truth的每个陈述句，判断其是否被检索到的上下文覆盖
   - 典型值：0.67~1.00（理想情况下应接近1.00）

3. Faithfulness（忠实度）:
   - 定义：答案是否能从检索到的上下文中推理得出
   - 注意：不评估答案正确性，只评估答案与上下文的一致性
   - 典型值：0.74~1.00

4. Answer Relevancy（答案相关性）:
   - 定义：答案与问题的相关程度
   - 注意：不评估答案正确性，只评估是否回答了问题
   - 典型值：0.88~0.92
"""
import os
import time

# ============================================
# 【第1步】导入LangChain核心组件
# ============================================
# RetrievalQA: 经典的检索问答链，将检索器与LLM组合完成RAG流程
# 原理：接收query → 检索相关文档 → 将文档+query拼接为prompt → LLM生成答案
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# LLMChainExtractor: 基于LLM的文档压缩器，从每个文档中提取与查询相关的内容
# 原理：对每个检索到的文档，调用LLM判断并提取与query相关的片段
# 风险：每个文档都会调用一次LLM，token消耗较大
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import PyPDFLoader

# EnsembleRetriever: 混合检索器，支持多路检索结果融合
# 原理：同时调用多个子检索器，按权重加权融合得分，默认使用加权融合而非RRF
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever

from langchain_community.embeddings import DashScopeEmbeddings

# BM25Retriever: 基于BM25算法的全文检索器
# 原理：计算query与文档的词项匹配得分，擅长精确关键词匹配
from langchain_community.retrievers import BM25Retriever

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ============================================
# 【第2步】导入RAGAS评估框架
# ============================================
# RAGAS: 自动化RAG系统评估框架，使用LLM作为评判器
# 特点：无需人工标注即可评估，但依赖LLM的判断能力
from datasets import Dataset
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
#   - chunk_size=256: 每个分块的目标字符数
#   - chunk_overlap=50: 相邻分块的重叠字符数，保证语义连续性
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=50,
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
index_name = "4"
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
# 历史实验结果记录（topK=10时）：
# - mix_retriever<0.5,0.5>: f1=0.5381, recall=0.6667, precision=0.4511
# - mix_retriever<0.7,0.3>: f1=0.5530, recall=1.0000, precision=0.3822（高召回低精度）
# - mix_retriever<0.2,0.8>: f1=0.5530, recall=1.0000, precision=0.3822
# - faiss_retriever: f1=0.8664, recall=1.0000, precision=0.7644（最佳平衡）
# - compression_retriever: f1=0.8313, recall=0.8333, precision=0.8293
#
# topK调整实验：
# - topK=4: f1较低，recall=0.6667（检索文档不足导致信息缺失）
# - topK=7: f1中等，recall=0.6667
# - topK=14: recall=1.0000但precision降至0.3144（噪声文档增多）
#
# 结论：topK值需要在recall和precision之间权衡
topK_doc_count = 4

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
    '''您好，如果您的座椅太热，1、通过中央显示屏，设置座椅加热强度或关闭座椅加热功能，
    在中央显示屏中点击座椅进入座椅加热控制界面，可在"关-低-中-高"之间循环。
    2、登录Lynk & Co App，按下前排座椅加热图标图标可以打开/关闭前排座椅加热。
    3、在中央显示屏中唤起空调控制界面然后点击舒适选项，降低座椅加热时间。'''
]

# ============================================
# 【第8步】问答评估器类定义
# ============================================
class QAEvaluator:
    """
    RAG问答评估器：封装问答生成和RAGAS评估流程
    
    使用RetrievalQA链（经典API）：
    - 原理：将检索器与LLM封装为一个完整的问答链
    - chain_type="stuff": 将所有检索到的文档直接"塞"进prompt（简单直接但可能超长）
    - return_source_documents=True: 返回检索到的原始文档，用于后续评估
    """
    def __init__(self, llm, retriever, embeddings):
        # RetrievalQA.from_chain_type: 工厂方法创建检索问答链
        # 参数说明：
        #   - llm: 用于生成答案的语言模型
        #   - chain_type: 文档处理方式，"stuff"表示直接拼接所有文档
        #   - retriever: 文档检索器
        #   - return_source_documents: 是否返回检索到的源文档
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        self.embeddings = embeddings

    def generate_answers(self, questions):
        """
        生成答案并收集上下文
        
        流程：
        1. 对每个问题调用检索问答链
        2. 提取生成的答案(result)和检索到的文档(source_documents)
        3. 收集所有结果用于后续RAGAS评估
        
        返回：
        - answers: LLM生成的答案列表
        - contexts: 每个问题对应的检索文档内容列表（二维列表）
        """
        answers = []
        contexts = []
        for question in questions:
            print("问题：", question)
            # invoke: 执行检索问答链
            # 内部流程：检索 → 构建prompt → LLM生成
            response = self.chain.invoke(question)
            print("大模型答复：", response['result'], "\n")
            answers.append(response['result'])
            # 提取检索到的文档内容，用于RAGAS的contexts字段
            contexts.append([doc.page_content for doc in response['source_documents']])
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
            embeddings=self.embeddings,
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
    qa_evaluator = QAEvaluator(model, retriever, llm_embeddings)
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
    return f1_score

# ============================================
# 【第9步】多检索器效果对比实验
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
time.sleep(10)  # 暂停避免API限流

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
mix_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],weight=[0.2, 0.8])
mix_evaluate_result = exec_eval(mix_retriever)
print("mix_retriever评估结果：", mix_evaluate_result," ，f1分数：",calc_f1(mix_evaluate_result))
time.sleep(10)

# --------------------------------------------
# 实验4：上下文压缩检索器
# --------------------------------------------
# LLMChainExtractor: 基于LLM的文档压缩器
# 原理：对每个检索到的文档，调用LLM提取与query相关的片段
# 注意：每个文档都会调用一次LLM，token消耗较大，生产环境需谨慎使用
compressor = LLMChainExtractor.from_llm(model)

# ContextualCompressionRetriever: 上下文压缩检索器
# 原理：
#   1. 先通过base_retriever检索初始文档
#   2. 再通过base_compressor压缩/过滤文档
#   3. 返回压缩后的文档列表
# 作用：减少无关内容，提高context_precision
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=mix_retriever
)
compression_evaluate_result = exec_eval(compression_retriever)
print("compression_retriever评估结果：", compression_evaluate_result," ，f1分数：",calc_f1(compression_evaluate_result))