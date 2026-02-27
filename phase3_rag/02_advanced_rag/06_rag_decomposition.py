"""
RAG 问题分解 (Query Decomposition) 示例
======================================
【核心原理】
问题分解是一种高级的RAG技术，它将复杂的用户查询拆解成多个简单的子问题，
分别检索和回答，最后整合得到完整的答案。

【为什么需要问题分解？】
1. 复杂问题的答案往往分散在多个文档中，单次检索难以获取全部信息
2. 向量相似度搜索基于距离计算，可能遗漏某些关键但语义距离较远的文档
3. 将大问题拆成小问题，每个子问题更具体，检索精度更高

【工作流程】
用户问题 → 分解为子问题列表 → 逐个检索子问题 → 生成子答案 → 整合为最终答案
"""
import os
from typing import List

from langchain_classic.retrievers.multi_query import LineListOutputParser
# 导入LangChain核心组件
from langchain_community.chat_models import ChatTongyi  # 通义千问大模型
from langchain_community.embeddings import DashScopeEmbeddings  # 阿里云向量嵌入模型
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate, BasePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from common_ai.ai_variable import ALI_TONGYI_API_KEY_OS_VAR_NAME

# ==================== 模型和嵌入配置 ====================
# 【配置说明】
# 本示例需要两个核心组件：
# 1. LLM (大语言模型): 用于问题分解和答案生成
# 2. Embedding模型: 用于文档向量化和语义检索

# 1. 初始化大语言模型 - 用于生成子问题和回答子问题
# 【原理】LLM具有强大的语言理解能力，可以分析问题的结构，识别其中的多个维度，
# 并将复杂问题拆解成逻辑上独立的子问题
model = ChatTongyi(
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)

# 2. 初始化嵌入模型 - 用于将文档内容转换为向量表示
# 【原理】向量嵌入将文本映射到高维空间，语义相似的文本在向量空间中距离更近
# 这是实现语义检索的基础，比传统的关键词匹配更能理解文本含义
llm_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 使用最新的向量嵌入模型
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)  # 从环境变量获取API密钥
)

# ==================== 工具函数 ====================
def pretty_print_docs(docs):
    """
    格式化打印文档列表
    【作用】将检索到的文档以清晰的格式输出，便于查看和调试
    """
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# ==================== 文档数据准备 ====================
# 【场景设计】
# 使用"番茄炒蛋"作为示例，包含三个维度的信息：
# 1. 食材/原料文档 - 回答"需要什么材料"
# 2. 步骤文档 - 回答"怎么做"
# 3. 技巧文档 - 回答"有什么注意事项"
#
# 【问题分解的价值】
# 当用户问"新手如何制作番茄炒蛋？"时，这个问题包含多个子维度：
# - 需要什么食材？
# - 具体步骤是什么？
# - 有什么注意事项？
# 单次检索可能只能命中其中一个文档，而分解后可以分别检索每个维度

documents = [
    Document(page_content="番茄炒蛋的食材：\n\n- 新鲜鸡蛋：3-4个（根据人数调整）\n- 番茄：2-3个中等大小\n- 盐：适量\n- 白糖：一小勺（可选，用于提鲜）\n- 食用油：适量\n- 葱花：少许（可选，用于增香）\n\n这些是最基本的材料，当然也可以根据个人口味添加其他调料或配料。"),
    Document(page_content="番茄炒蛋的步骤：鸡蛋打入碗中，加入少许盐，用筷子或打蛋器充分搅拌均匀；\n   - 番茄洗净后切成小块备用。\n\n3. **炒鸡蛋**：锅内倒入适量食用油加热至温热状态，然后将搅拌好的鸡蛋液缓缓倒入锅中。待鸡蛋凝固时轻轻翻动几下，让其受热均匀直至完全熟透，随后盛出备用。\n\n4. **炒番茄**：在同一锅里留下的底油中放入切好的番茄块，中小火慢慢翻炒至出汁，可根据个人口味加一点点白糖提鲜。\n\n5. **合炒**：当番茄炒至软烂并开始释放大量汤汁时，再把之前炒好的鸡蛋倒回锅里，快速与番茄混合均匀，同时加入适量的盐调味。如果喜欢的话还可以撒上一些葱花增加香气。\n\n6. **完成**：最后检查一下味道是否合适，确认无误后即可关火装盘享用美味的番茄炒蛋啦！"),
    Document(page_content="技巧与注意事项：1. **选材**：选择新鲜的鸡蛋和成熟的番茄。新鲜的食材是做好这道菜的基础。\n2. **打蛋液**：将鸡蛋打入碗中后加入少许盐（根据个価口味调整），然后充分搅拌均匀。这样做可以让蛋更加松软且味道更佳。\n3. **处理番茄**：番茄最好先用开水稍微焯一下皮，然后去皮切块。这样可以去除表皮的硬质部分，让番茄更容易入味，并且口感更好。\n4. **热锅冷油**：先用中小火把锅烧热，再倒入适量食用油，待油温五成热时下蛋液。这样的做法可以使蛋快速凝固形成漂亮的形状而不易粘锅。\n5. **分步烹饪**：通常建议先炒鸡蛋至半熟状态取出备用；接着利用剩下的底油继续翻炒番茄至出汁，最后再将之前炒好的鸡蛋倒回锅里与番茄混合均匀加热即可。\n6. **调味品**：除了基本的盐之外，还可以根据喜好添加少量糖来提鲜或者一点酱油增色添香。注意调味料不宜过多以免掩盖了食材本身的味道。\n7. **出锅前加葱花**：如果喜欢的话，在即将完成时撒上一些葱花不仅能增加菜品色泽还能增添香气。")
]

# ==================== 向量数据库和检索器 ====================
# 【原理】Chroma是一个轻量级的向量数据库，用于存储文档的向量表示
# 并支持基于相似度的语义检索

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=llm_embeddings,
    collection_name="decomposition"
)

# 创建基础检索器，每次只返回最相关的1个文档
# 【注意】这里设置k=1是为了演示问题分解的必要性：
# 单次检索只能得到1个文档，无法覆盖问题的全部维度
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

print("-------------检索到的文档（拆解前）--------------")
# 【对比实验】先看普通检索的结果
# 用户问题"新手如何制作番茄炒蛋？"包含多个维度，但单次检索只能命中一个文档
# 从实际来说，完整的答案应该包括：原材料、步骤、注意事项
pretty_print_docs(retriever.invoke("新手如何制作番茄炒蛋？"))


print("-------------检索到的文档（开始拆解）--------------")

# ==================== 子问题生成Prompt ====================
# 【原理】这是问题分解的核心Prompt，指导LLM将复杂问题拆解为子问题
# 
# 【Prompt设计要点】
# 1. 明确角色：AI语言模型助理
# 2. 明确任务：将输入问题分解成3个子问题
# 3. 说明目的：克服基于距离的相似性搜索的局限性
# 4. 格式要求：用换行符分隔，只输出子问题本身
#
# 【为什么这样设计？】
# - 限定3个子问题：避免过多导致效率降低，过少则覆盖不全
# - 换行分隔：便于后续用LineListOutputParser解析
# - 强调"克服相似性搜索局限"：让LLM理解分解的目的

template = """
            你是一名AI语言模型助理。你的任务是将输入问题分解成3个子问题，通过一个个解决这些子问题从而解决完整的问题。
            子问题需要在矢量数据库中检索相关文档。通过分解用户问题生成子问题，你的目标是帮助用户克服基于距离的相似性搜索的一些局限性。
            请提供这些用换行符分隔的子问题本身，不需要额外内容。
            原始问题: {question}"""

DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=template,
)

print("-------------测试大模型对问题的拆解，实际业务中可不用--------------")

# 【LineListOutputParser的作用】
# 将模型输出的文本按换行符分割成字符串列表
# 例如："问题1\n问题2\n问题3" → ["问题1", "问题2", "问题3"]
# 这是连接LLM输出和程序处理的重要桥梁

chain = DEFAULT_QUERY_PROMPT | model | LineListOutputParser()
result = chain.invoke({"question": "新手如何制作番茄炒蛋？"})
print("问题拆解：", result)
print("-------------完成测试大模型对问题的拆解--------------")

# ==================== 子问题回答Prompt ====================
# 【原理】这个Prompt用于指导LLM基于检索到的文档回答每个子问题
#
# 【输入变量说明】
# - question: 原始的主问题（提供上下文）
# - sub_question: 当前要回答的子问题
# - documents: 为该子问题检索到的相关文档
#
# 【设计思路】
# 1. 明确主问题和子问题的关系，让LLM理解上下文
# 2. 提供检索到的文档作为回答依据
# 3. 要求直接给出答案，避免冗余输出

DEFAULT_SUB_QUESTION_PROMPT = PromptTemplate(
    input_variables=["question", "sub_question", "documents"],
    template="""要解决主要问题{question}，需要先解决子问题{sub_question}。
    以下是为支持您的推理而提供的参考文档：{documents}。请直接给出当前子问题的答案。不需要额外内容。""",
)

# ==================== 问题分解检索器类 ====================
# 【核心类设计】DecompositionQueryRetriever
# 继承自BaseRetriever，实现自定义的检索逻辑
#
# 【类属性说明】
# - retriever: 基础的向量检索器，用于检索子问题相关文档
# - make_sub_chain: 生成子问题的处理链 (Prompt → LLM → OutputParser)
# - resolve_sub_chain: 解决子问题的处理链 (Prompt → LLM)

class DecompositionQueryRetriever(BaseRetriever):
    """
    问题分解检索器
    
    【核心功能】
    1. 将用户问题分解为多个子问题
    2. 对每个子问题分别检索相关文档
    3. 基于检索结果生成子问题的答案
    4. 将所有子问题和答案整合为文档返回
    
    【使用场景】
    适用于复杂查询，单个问题包含多个维度，需要综合多个文档才能回答完整的情况
    """
    
    # 向量数据库检索器 - 用于检索子问题相关文档
    retriever: BaseRetriever
    # 生成子问题链 - Prompt → LLM → LineListOutputParser
    make_sub_chain: Runnable
    # 解决子问题链 - Prompt → LLM
    resolve_sub_chain: Runnable

    @classmethod
    def from_llm(
            cls,
            retriever: BaseRetriever,
            llm: BaseLanguageModel,
            prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,
            sub_prompt: BasePromptTemplate = DEFAULT_SUB_QUESTION_PROMPT
    ) -> "DecompositionQueryRetriever":
        """
        类方法：从LLM和检索器创建实例
        
        【参数说明】
        - retriever: 基础向量检索器
        - llm: 大语言模型，用于生成子问题和回答
        - prompt: 生成子问题的Prompt模板
        - sub_prompt: 回答子问题的Prompt模板
        
        【返回】
        DecompositionQueryRetriever实例
        """
        output_parser = LineListOutputParser()
        
        # 构建处理链：
        # make_sub_chain: 输入问题 → Prompt填充 → LLM生成 → 解析为列表
        # resolve_sub_chain: 输入变量 → Prompt填充 → LLM生成答案
        
        return cls(
            retriever=retriever,
            make_sub_chain=prompt | llm | output_parser,
            resolve_sub_chain=sub_prompt | llm
        )

    def generate_queries(self, question: str) -> List[str]:
        """
        生成子问题列表
        
        【原理】调用make_sub_chain，让LLM将复杂问题分解为多个子问题
        
        【参数】
        - question: 用户的原始问题
        
        【返回】
        - List[str]: 子问题字符串列表
        """
        response = self.make_sub_chain.invoke({"question": question})
        lines = response
        print(f"生成子问题: {lines}")
        return lines

    def retrieve_documents(self, query: str, sub_queries: List[str]) -> List[Document]:
        """
        检索并回答子问题
        
        【原理】
        1. 对每个子问题，使用基础检索器查找相关文档
        2. 将子问题和检索到的文档传入LLM，生成该子问题的答案
        3. 将所有子问题及其答案封装为Document对象返回
        
        【参数】
        - query: 原始主问题（用于提供上下文）
        - sub_queries: 子问题列表
        
        【返回】
        - List[Document]: 包含子问题和答案的文档列表
        """
        # 使用RunnableLambda创建动态处理链
        # 每个子问题都会：检索文档 → 调用LLM生成答案
        sub_llm_chain = RunnableLambda(
            lambda sub_query: self.resolve_sub_chain.invoke(
                {
                    "question": query,           # 主问题（上下文）
                    "sub_question": sub_query,   # 当前子问题
                    "documents": [doc.page_content for doc in self.retriever.invoke(sub_query)]  # 检索结果
                }
            )
        )
        
        # 【批量执行】使用batch方法并行处理所有子问题，提高效率
        responses = sub_llm_chain.batch(sub_queries)
        
        # 将子问题和对应的答案合并，封装为Document
        # 格式："子问题\n答案内容"
        documents = [
            Document(page_content=sub_query + "\n" + response.content)
            for sub_query, response in zip(sub_queries, responses)
        ]
        return documents

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        实现BaseRetriever的抽象方法
        
        【处理流程】
        1. 生成子问题
        2. 检索并回答每个子问题
        3. 返回整合后的文档列表
        
        【参数】
        - query: 用户查询
        - run_manager: 回调管理器（用于追踪和监控）
        
        【返回】
        - List[Document]: 相关文档列表
        """
        # 步骤1：生成子问题
        sub_queries = self.generate_queries(query)
        
        # 步骤2：解决子问题（检索+回答）
        documents = self.retrieve_documents(query, sub_queries)
        
        return documents


# ==================== 使用问题分解检索器 ====================
print("首先使用 DecompositionQueryRetriever 来分解问题------------>")

# 创建问题分解检索器实例
# 【参数】
# - llm: 用于生成子问题和回答的大模型
# - retriever: 基础向量检索器

decompositionQueryRetriever = DecompositionQueryRetriever.from_llm(
    llm=model, 
    retriever=retriever
)

# 调用检索器
# 【内部流程】
# 1. LLM将"番茄炒蛋怎么制作？"分解为3个子问题
#    例如：["番茄炒蛋需要什么食材？", "番茄炒蛋的步骤是什么？", "番茄炒蛋有什么技巧？"]
# 2. 对每个子问题分别检索文档
# 3. 基于检索结果生成每个子问题的答案
# 4. 返回整合后的文档

decomposition_docs = decompositionQueryRetriever.invoke("番茄炒蛋怎么制作？")

print("-------------检索到的文档（拆解后）--------------")
pretty_print_docs(decomposition_docs)

# ==================== 最终答案生成 ====================
# 【最后一步】基于问题分解检索到的所有文档，生成完整答案

# 创建最终回答的Prompt模板
template = """
请根据以下文档回答问题:
### 文档:
{context}
### 问题:
{question}
"""

# 由模板生成ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(template)

# 构建处理链：Prompt → LLM
chain = prompt | model

print("-------------回答--------------")
question = "新手如何制作番茄炒蛋？"

# 调用链生成最终答案
# 【输入】
# - context: 问题分解检索到的所有文档内容
# - question: 用户的原始问题
#
# 【输出】
# 基于完整信息的综合答案，包含食材、步骤、技巧等所有维度

response = chain.invoke({
    "context": [doc.page_content for doc in decomposition_docs], 
    "question": question
})
print(response.content)


# ==================== 问题分解RAG总结 ====================
"""
【问题分解RAG - 完整流程总结】

┌─────────────────────────────────────────────────────────────────────────────┐
│                         问题分解RAG工作流程                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   步骤1: 问题分解 (Query Decomposition)                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 输入: 用户复杂问题 (如"新手如何制作番茄炒蛋？")                         │   │
│   │         ↓                                                           │   │
│   │ 处理: LLM分析问题结构，识别多个维度                                     │   │
│   │         ↓                                                           │   │
│   │ 输出: 子问题列表 (如["需要什么食材？", "步骤是什么？", "有什么技巧？"])   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│   步骤2: 子问题检索 (Sub-query Retrieval)                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 对每个子问题:                                                        │   │
│   │   - 使用向量检索器检索相关文档                                         │   │
│   │   - 基于检索结果生成子问题答案                                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│   步骤3: 答案整合 (Answer Integration)                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 输入: 所有子问题及其答案                                              │   │
│   │         ↓                                                           │   │
│   │ 处理: LLM综合所有信息，生成完整答案                                    │   │
│   │         ↓                                                           │   │
│   │ 输出: 结构化的最终答案                                                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

【核心原理】

1. 为什么需要问题分解？
   ┌────────────────────────────────────────────────────────────┐
   │ 传统RAG的局限:                                               │
   │   - 单次检索只能获取有限数量的文档                             │
   │   - 复杂问题涉及多个维度，单次查询难以覆盖全部                  │
   │   - 向量相似度可能遗漏关键但语义距离较远的文档                   │
   │                                                             │
   │ 问题分解的优势:                                              │
   │   ✓ 每个子问题更具体，检索精度更高                             │
   │   ✓ 多维度覆盖，信息更全面                                    │
   │   ✓ 克服单一查询的语义局限                                    │
   └────────────────────────────────────────────────────────────┘

2. 关键组件说明:
   ┌────────────────────────────────────────────────────────────┐
   │ 组件                    │ 作用                              │
   ├────────────────────────────────────────────────────────────┤
   │ LineListOutputParser    │ 将LLM输出按行解析为列表            │
   │ DEFAULT_QUERY_PROMPT    │ 指导LLM分解问题的Prompt            │
   │ DEFAULT_SUB_QUESTION_PROMPT │ 指导LLM回答子问题的Prompt     │
   │ DecompositionQueryRetriever │ 自定义检索器，封装分解逻辑     │
   └────────────────────────────────────────────────────────────┘

3. 代码结构:
   ┌────────────────────────────────────────────────────────────┐
   │ 配置层: 模型初始化、Prompt定义                                │
   │     ↓                                                       │
   │ 数据层: 文档准备、向量数据库构建                               │
   │     ↓                                                       │
   │ 核心层: DecompositionQueryRetriever类实现                    │
   │     ↓                                                       │
   │ 应用层: 检索调用、最终答案生成                                 │
   └────────────────────────────────────────────────────────────┘

【适用场景】

✓ 复杂查询：问题包含多个子任务或维度
✓ 信息分散：答案需要综合多个文档
✓ 高精度要求：需要全面且准确的回答
✓ 多跳推理：需要多步骤推理才能回答的问题

【注意事项】

⚠ 子问题数量：过多会降低效率，过少则覆盖不全（建议3-5个）
⚠ 成本考量：每个子问题都涉及一次LLM调用，成本会累积
⚠ 并行优化：使用batch方法并行处理子问题，提高效率

【与其他RAG技术对比】

┌────────────────┬─────────────────────────────────────────────────────────────┐
│ 技术            │ 核心思想                                                     │
├────────────────┼─────────────────────────────────────────────────────────────┤
│ 基础RAG         │ 单次检索 + 生成答案                                          │
│ 多查询RAG       │ 生成多个查询变体，分别检索后合并                               │
│ 问题分解RAG     │ 将问题拆解为子问题，分别解决后整合 ← 本示例                   │
│ 递归RAG         │ 迭代分解问题，直到可以直接回答                                 │
└────────────────┴─────────────────────────────────────────────────────────────┘
"""