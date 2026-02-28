import os
import time

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import PyPDFLoader

from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.embeddings import DashScopeEmbeddings

from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME


model = ChatTongyi(
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)


llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 使用最新的向量嵌入模型
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)  # 从环境变量获取API密钥
)

vectordb = None

file_path = "../Data/领克汽车用户操作手册.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print("文档个数：", len(docs))
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=50,
)
split_docs = text_splitter.split_documents(docs)
print("分块个数：", len(split_docs))

index_folder_path = "data/faiss_index"
index_name = "4"
index_file_path = os.path.join(index_folder_path, f"{index_name}.faiss")
# 检查索引文件是否存在
if os.path.exists(index_file_path):
    print("索引文件已存在，直接加载...")
    vectordb = FAISS.load_local(index_folder_path, llm_embeddings, index_name, allow_dangerous_deserialization=True)
else:
    print("索引文件不存在，创建并保存索引...")
    # 创建向量存储
    vectordb = FAISS.from_documents(split_docs, llm_embeddings)
    # 保存索引
    vectordb.save_local(index_folder_path, index_name)
    print("向量化完成....")

'''
初始为10
检索器=mix_retriever<0.5,0.5>，指标为{'faithfulness': 0.9333, 'answer_relevancy': 0.8924,
 'context_recall': 0.6667, 'context_precision': 0.4511}  f1分数：0.5381
检索器=mix_retriever<0.7,0.3>，指标为{'faithfulness': 0.9333, 'answer_relevancy': 0.8846, 
 'context_recall': 1.0000, 'context_precision': 0.3822}   f1分数：0.5530
检索器=mix_retriever<0.2,0.8>，指标为{'faithfulness': 1.0000, 'answer_relevancy': 0.9112, 
'context_recall': 1.0000, 'context_precision': 0.3822}   f1分数：0.5530
检索器=faiss_retriever，指标为{'faithfulness': 0.8889, 'answer_relevancy': 0.9230, 
 'context_recall': 1.0000, 'context_precision': 0.7644} f1分数： 0.8664
检索器=compression_retriever，指标为{'faithfulness': 0.7381, 'answer_relevancy': 0.9092, 
 'context_recall': 0.8333, 'context_precision': 0.8293} f1分数： 0.8313
'''
'''调整为4，指标为 {'faithfulness': 0.6667, 'answer_relevancy': 0.8955,
 'context_recall': 0.6667, 'context_precision': 0.3889}'''
'''调整为7，指标为 {'faithfulness': 0.8000, 'answer_relevancy': 0.8926,
 'context_recall': 0.6667, 'context_precision': 0.4517}'''
'''调整为14，指标为 {'faithfulness': 0.8413, 'answer_relevancy': 0.8792,
 'context_recall': 1.0000, 'context_precision': 0.3144}'''
topK_doc_count = 4

questions = ["如何使用安全带？", "车辆如何保养？", "座椅太热怎么办？"]
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
    '''您好，如果您的座椅太热，1、通过中央显示屏，设置座椅加热强度或关闭座椅加热功能，
    在中央显示屏中点击座椅进入座椅加热控制界面，可在“关-低-中-高”之间循环。
    2、登录Lynk & Co App，按下前排座椅加热图标图标可以打开/关闭前排座椅加热。
    3、在中央显示屏中唤起空调控制界面然后点击舒适选项，降低座椅加热时间。'''
]

class QAEvaluator:
    def __init__(self, llm, retriever, embeddings):
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        self.embeddings = embeddings

    def generate_answers(self, questions):
        answers = []
        contexts = []
        for question in questions:
            print("问题：", question)
            response = self.chain.invoke(question)
            print("大模型答复：", response['result'], "\n")
            answers.append(response['result'])
            contexts.append([doc.page_content for doc in response['source_documents']])
        return answers, contexts

    def evaluate(self, questions, answers, contexts, ground_truths):
        evaluate_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        evaluate_dataset = Dataset.from_dict(evaluate_data)
        evaluate_result = evaluate(
            evaluate_dataset,
            llm=llm,
            embeddings=self.embeddings,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ]
        )

        return evaluate_result

def exec_eval(retriever):
    qa_evaluator = QAEvaluator(llm, retriever, embeddings)
    answers, contexts = qa_evaluator.generate_answers(questions)
    return qa_evaluator.evaluate(questions, answers, contexts, ground_truths)

def calc_f1(evaluate_result):
    context_precisions = evaluate_result["context_precision"]
    context_recalls = evaluate_result["context_recall"]
    print("context_precisions=",context_precisions)
    print("context_recalls=",context_recalls)
    context_precision_score = sum(context_precisions) / len(context_precisions)
    context_recall_score = sum(context_recalls) / len(context_recalls)
    f1_score = (2 * context_precision_score * context_recall_score) / (context_precision_score + context_recall_score)
    return f1_score

# 创建三个不同的检索器，评估三个检索器的效果

# 创建向量检索器，擅长根据语义相似度查找相关文档
faiss_retriever = vectordb.as_retriever(search_kwargs={"k": topK_doc_count})
evaluate_result = exec_eval(faiss_retriever)
print("faiss_retriever评估结果：", evaluate_result," ，f1分数：",calc_f1(evaluate_result))
time.sleep(10)
# 创建全文检索器，擅长根据全文查找相关文档
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k=topK_doc_count
# 创建混合检索器
mix_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],weight=[0.2, 0.8])
mix_evaluate_result = exec_eval(mix_retriever)
print("mix_retriever评估结果：", mix_evaluate_result," ，f1分数：",calc_f1(mix_evaluate_result))
time.sleep(10)
# 创建文档压缩器：LLMChainExtractor ，它将遍历最初返回的文档，并仅从每个文档中提取与查询相关的内容。
compressor = LLMChainExtractor.from_llm(llm)
# 创建上下文压缩检索器：需要传入一个文档压缩器和基本检索器
# 上下文压缩检索器将查询传递到基本检索器，获取初始文档并将它们传递到文档压缩器。
# 文档压缩器获取文档列表，并通过减少文档内容或完全删除文档来缩短文档列表。
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=mix_retriever
)
compression_evaluate_result = exec_eval(compression_retriever)
print("compression_retriever评估结果：", compression_evaluate_result," ，f1分数：",calc_f1(compression_evaluate_result))