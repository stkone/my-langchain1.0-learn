"""
RAG (Retrieval-Augmented Generation) 是 LangChain 的核心应用场景之一
LangChain为RAG提供了很多预制链，使用这这些预制链条可以进行快速构建
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

#1.获得访问大模型客户端
model = ChatTongyi()

#2.获得向量数据库对象
llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)

#3.加载网站
loader = WebBaseLoader(
        web_path="https://www.gov.cn/yaowen/liebiao/202512/content_7050416.htm",
        # bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT"))
        # 分割，将网页中目标内容进行分割
        bs_kwargs={"parse_only":bs4.SoupStrainer(id="UCAP-CONTENT")}
        )
docs = loader.load()

#4.文本的切割
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(docs)

#5.存储到向量空间
db = Chroma.from_documents(documents=documents,embedding=llm_embeddings)

#6.检索器
retriever = db.as_retriever()

# 7. 初始化模版
#注意这里的prompt模板中包含 {context} 和 {input} 的模板
#需要使用{context}，这个变量，来表示上下文，这个变量，会自动从retriever中获取。
#而human中也限定了变量{input}，链的必须使用这个变量。
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


# 8. 创建链，文档链  create_stuff_documents_chain
chain1 = create_stuff_documents_chain(model,prompt_template)
# 8. 创建链 搜索链 create_retrieval_chain  参数1:是检索器  参数2:是文档链
chain2 = create_retrieval_chain(retriever,chain1)
# 9. 用大模型生成答案
resp = chain2.invoke({"input":"会议说了什么?"})

print(type(resp))
print(resp)
print("===================")
print(resp["answer"])
