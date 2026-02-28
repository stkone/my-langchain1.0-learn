"""
金融报告RAG问答系统

基于MinerU文档解析 + ChromaDB向量存储 + DashScope大模型的RAG实现
适用于财务年报等结构化文档的智能问答场景。

核心流程：
1. PDF文档解析(MinerU) → 2. 智能分块 → 3. 向量化存储(ChromaDB) → 4. 检索问答
"""
import glob
import os
import json
import subprocess
from datetime import datetime
import chromadb
import torch
import doclayout_yolo.nn.tasks
import dashscope
from http import HTTPStatus

# ============================================
# 第0步：环境初始化与依赖配置
# ============================================
# 注册YOLOv10模型为安全全局对象，避免PyTorch反序列化警告
# 原理：MinerU使用YOLOv10进行文档版面分析，需要序列化/反序列化模型权重
torch.serialization.add_safe_globals([doclayout_yolo.nn.tasks.YOLOv10DetectionModel])

# DashScopeEmbeddings: 通义千问Embedding模型封装
# 原理：将文本映射到高维向量空间，语义相似的文本向量距离更近
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

# MinerU文档解析工具
# do_parse: 执行PDF解析的核心函数
# read_fn: 读取PDF文件的辅助函数
from mineru.cli.common import do_parse, read_fn
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量（DASHSCOPE_API_KEY等）
load_dotenv()


# ============================================
# 第1部分：MinerURAGSystem - 文档处理与向量存储系统
# ============================================
class MinerURAGSystem:
    """
    基于MinerU的RAG文档处理系统
    
    职责：
    1. PDF文档解析（MinerU智能版面分析）
    2. 文档智能分块（保留标题层级、表格结构）
    3. 向量嵌入生成（DashScope Embedding）
    4. 向量数据库存储与检索（ChromaDB）
    
    与标准RAG的区别：
    - 使用MinerU替代PyPDFLoader，能识别文档结构（标题、表格、正文）
    - 分块策略考虑文档层级，标题与内容保持关联
    - 表格单独处理，保留HTML格式和元数据
    """
    def __init__(self, persist_directory="./chroma_db"):
        # DashScopeEmbeddings: 通义千问文本嵌入模型
        # 用法：
        #   - embed_documents(texts): 批量编码文档列表
        #   - embed_query(text): 编码单个查询
        # 参数：
        #   - model: 嵌入模型版本，默认text-embedding-v2
        #   - dashscope_api_key: API密钥，从环境变量读取
        self.embedding_model = DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-v2"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        
        # ChromaDB持久化客户端
        # 原理：将向量索引保存到本地磁盘，支持增量更新
        # 用法：path指定存储目录，重启后数据不丢失
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 获取或创建集合（相当于关系型数据库的表）
        # 集合名称"documents"，用于存储文档块及其向量
        self.collection = self.client.get_or_create_collection("documents")

    def process_documents(self, pdf_directory, output_dir="./processed"):
        """
        批量处理PDF文档：解析 → 结构化提取 → 生成JSON
        
        流程：
        1. 遍历PDF目录，逐个读取PDF二进制数据
        2. 调用MinerU的do_parse进行智能版面分析
        3. 收集生成的_content_list.json文件路径
        
        参数：
        - pdf_directory: PDF文件所在目录
        - output_dir: 解析结果输出目录
        
        返回：
        - processed_files: 解析生成的JSON文件路径列表
        
        MinerU解析输出说明：
        - 每个PDF生成一个目录，包含auto子目录
        - _content_list.json: 文档内容结构化列表（文本、表格、标题层级）
        - _middle.json: 中间处理结果
        - images/: 提取的图片文件
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        processed_files = []
        
        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                input_path = os.path.join(pdf_directory, filename)
                output_path = os.path.join(output_dir, filename.replace('.pdf', ''))
                pdf_file_name = Path(input_path).stem
                
                # 读取PDF二进制数据
                pdf_bytes = read_fn(input_path)
                
                # MinerU智能解析
                # 参数说明：
                #   - output_dir: 解析结果输出目录
                #   - pdf_file_names: PDF文件名列表
                #   - pdf_bytes_list: PDF二进制数据列表
                #   - p_lang_list: 文档语言列表，["ch"]表示中文
                #   - backend: 解析后端，"pipeline"使用完整处理流程
                do_parse(
                    output_dir=output_path,
                    pdf_file_names=[pdf_file_name],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=["ch"],
                    backend="pipeline"
                )
                
                # 收集解析生成的内容列表文件
                content_file = os.path.join(output_path, pdf_file_name, "auto", f"{pdf_file_name}_content_list.json")
                if os.path.exists(content_file):
                    processed_files.append(content_file)
                    
        return processed_files

    def _split_into_chunks(self, content_data, chunk_size, overlap):
        """
        智能文档分块：保留标题层级，单独处理表格
        
        分块策略：
        1. 标题识别：text_level>0表示标题，单独成块并标记has_title
        2. 正文累积：按chunk_size累积文本，超长时切分新块
        3. 表格处理：表格单独成块，保留HTML格式和元数据（页码、标题、脚注）
        4. 大表格切分：超长的表格按行切分，保留表头
        
        与RecursiveCharacterTextSplitter的区别：
        - 理解文档结构（标题层级），非纯文本切割
        - 表格作为独立块，保留结构化信息
        - 标题与后续内容保持语义关联
        
        参数：
        - content_data: MinerU解析后的内容列表
        - chunk_size: 目标块大小（字符数）
        - overlap: 块间重叠（本实现未使用，保留参数兼容性）
        
        返回：
        - chunks: 分块结果列表，每个块包含text和metadata
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for item in content_data:
            if item.get('type') == 'text':
                text = item.get('text', '')
                if not text:
                    continue
                    
                # 标题识别：text_level>0表示各级标题
                if item.get('text_level', 0) > 0:
                    # 先保存当前累积的文本块
                    if current_chunk:
                        chunks.append({
                            'text': ' '.join(current_chunk), 
                            'metadata': {'chunk_type': 'text', 'has_title': True}
                        })
                    # 标题作为新块的开头，标记为二级标题格式
                    current_chunk = [f"## {text}"]
                    current_length = len(text)
                    
                # 正文累积：检查是否超过chunk_size
                elif current_length + len(text) > chunk_size and current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk), 
                        'metadata': {'chunk_type': 'text'}
                    })
                    current_chunk = [text]
                    current_length = len(text)
                else:
                    current_chunk.append(text)
                    current_length += len(text)

            elif item.get('type') == 'table':
                # 遇到表格时，先保存当前累积的文本块
                if current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk), 
                        'metadata': {'chunk_type': 'text'}
                    })
                    current_chunk = []
                    current_length = 0

                # 表格单独处理
                table_body = item.get('table_body', '')
                if table_body:
                    # 小表格直接作为一个块
                    if len(table_body) <= chunk_size:
                        chunks.append({
                            'text': table_body,
                            'metadata': {
                                'chunk_type': 'table',
                                'page_idx': item.get('page_idx', 0),
                                'has_caption': len(item.get('table_caption', [])) > 0,
                                'has_footnote': len(item.get('table_footnote', [])) > 0
                            }
                        })
                    # 大表格按行切分
                    else:
                        table_chunks = self._split_table_into_chunks(table_body, chunk_size)
                        for i, chunk in enumerate(table_chunks):
                            chunks.append({
                                'text': chunk,
                                'metadata': {
                                    'chunk_type': 'table',
                                    'page_idx': item.get('page_idx', 0),
                                    'table_chunk_index': i,
                                    'total_table_chunks': len(table_chunks)
                                }
                            })
                            
        # 保存最后累积的文本块
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk), 
                'metadata': {'chunk_type': 'text'}
            })
            
        return chunks

    def _split_table_into_chunks(self, table_html, chunk_size):
        """
        表格按行切分：保留表头，每块包含表头+部分数据行
        
        原理：
        - 表格的语义完整性要求每块都包含表头（列定义）
        - 按行累积，超过chunk_size时切分新块
        - 保持HTML表格格式完整性
        
        参数：
        - table_html: 表格HTML字符串
        - chunk_size: 目标块大小
        
        返回：
        - chunks: 切分后的表格HTML片段列表
        """
        chunks = []
        
        # 提取表格内容（去掉<table>标签）
        if table_html.startswith('<table>'):
            table_content = table_html[7:-8]  # 去掉<table>和</table>
        else:
            table_content = table_html
            
        # 按行分割，rows[0]为空，rows[1]为表头
        rows = table_content.split('<tr>')
        table_head = rows[1]  # 表头行
        current_chunk = '<table>' + table_head
        print('表头:', current_chunk)

        current_chunk_size = len(current_chunk)
        
        # 从第2行开始处理数据行
        for row in rows[2:]:
            row_content = '<tr>' + row
            
            # 检查添加当前行后是否超过chunk_size
            if current_chunk_size + len(row_content) > chunk_size and current_chunk != '<table>':
                # 关闭当前表格块
                current_chunk += '</table>'
                chunks.append(current_chunk)
                # 新开一个块，包含表头
                current_chunk = '<table>' + table_head + row_content
                current_chunk_size = len(current_chunk)
            else:
                current_chunk += row_content
                current_chunk_size += len(row_content)

            print('行:', row_content)
            
        # 保存最后一个表格块
        if current_chunk != '<table>' + table_head:
            current_chunk += '</table>'
            chunks.append(current_chunk)
            
        return chunks

    def chunk_and_embed(self, content_files, chunk_size=512, overlap=50):
        """
        批量处理解析后的JSON文件：分块 → 准备元数据 → 生成ID
        
        流程：
        1. 读取MinerU生成的_content_list.json
        2. 调用_split_into_chunks进行智能分块
        3. 为每个块生成唯一ID和复合元数据
        
        参数：
        - content_files: JSON文件路径列表
        - chunk_size: 分块大小，默认512字符
        - overlap: 重叠大小（本实现未使用）
        
        返回：
        - documents: 文本块内容列表
        - metadatas: 元数据列表（包含source、chunk_type等）
        - ids: 唯一标识符列表
        """
        documents = []
        metadatas = []
        ids = []
        
        for file_path in content_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
                
            # 文档级元数据
            doc_metadata = {
                'source': os.path.basename(file_path),
                'file_type': 'pdf',
                'processing_date': datetime.now().isoformat()
            }
            
            # 智能分块
            text_chunks = self._split_into_chunks(content_data, chunk_size, overlap)
            
            # 为每个块生成记录
            for i, chunk in enumerate(text_chunks):
                documents.append(chunk['text'])
                # 合并文档级元数据和块级元数据
                metadatas.append({**doc_metadata, **chunk['metadata']})
                # 生成唯一ID：文件名_chunk_序号
                ids.append(f"{os.path.basename(file_path)}_chunk_{i}")
                
        return documents, metadatas, ids

    def build_vector_store(self, documents, metadatas, ids):
        """
        构建向量数据库：清空旧数据 → 生成嵌入 → 批量存储
        
        流程：
        1. 检查并清空集合中的旧数据（全量更新策略）
        2. 调用Embedding模型生成向量
        3. 批量添加到ChromaDB集合
        
        注意：本实现采用全量更新策略，每次重建整个索引
        生产环境建议改为增量更新：仅添加/修改/删除变更的文档
        
        参数：
        - documents: 文本块列表
        - metadatas: 元数据列表
        - ids: 唯一标识符列表
        """
        # 清空旧数据（全量更新）
        if self.collection.count() > 0:
            print(f"集合不为空，正在删除 {self.collection.count()} 个旧文档...")
            existing_ids = self.collection.get(include=[])['ids']
            self.collection.delete(ids=existing_ids)
            print("旧文档删除完毕。")
        else:
            print("集合为空，无需删除。")

        # 生成向量嵌入
        # embed_documents: 批量编码，比逐个编码更高效
        embeddings = self.embedding_model.embed_documents(documents)
        
        # 批量添加到向量数据库
        # ChromaDB自动维护向量索引，支持相似度检索
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"成功存储 {len(documents)} 个文档块到向量数据库")

    def query_documents(self, query_text, n_results=5):
        """
        向量相似度检索：将查询转为向量 → 检索最相似的文档块
        
        原理：
        1. 使用相同的Embedding模型编码查询文本
        2. ChromaDB计算查询向量与文档向量的相似度（余弦相似度）
        3. 返回相似度最高的n_results个文档块
        
        参数：
        - query_text: 用户查询文本
        - n_results: 返回结果数量，默认5
        
        返回：
        - results: 包含documents、metadatas、distances的字典
          - documents: 检索到的文本块内容
          - metadatas: 对应的元数据
          - distances: 向量距离（越小表示越相似）
        """
        # 编码查询文本（注意：使用embed_documents而非embed_query，因为输入是列表）
        query_embedding = self.embedding_model.embed_documents([query_text])[0]
        
        # 执行向量检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        return results


# ==============================================================================
# 第2部分：RAGQASystem - 问答生成系统
# ==============================================================================
class RAGQASystem:
    """
    RAG问答生成系统
    
    职责：
    1. 接收用户问题，调用MinerURAGSystem检索相关文档
    2. 构建结构化Prompt（包含检索到的上下文）
    3. 调用DashScope大模型生成答案
    
    与MinerURAGSystem的关系：
    - MinerURAGSystem负责文档处理和检索（RAG的"R"部分）
    - RAGQASystem负责答案生成（RAG的"G"部分）
    """
    def __init__(self, mineru_rag_system):
        self.rag_system = mineru_rag_system
        # 设置DashScope API密钥，用于调用通义千问模型
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    def _build_prompt(self, question, context_docs):
        """
        构建RAG Prompt：将检索到的文档片段组装为结构化提示词
            
        Prompt设计原则：
        1. 角色定义：明确指定"财务报告分析师"角色，引导专业回答风格
        2. 上下文约束：强调"仅依据提供的信息"，减少幻觉风险
        3. 不确定性处理：明确指导"信息不足时"的回应方式
        4. 结构化格式：使用---分隔符清晰区分上下文和问题
            
        参数：
        - question: 用户问题
        - context_docs: 检索到的文档片段列表
            
        返回：
        - prompt: 完整的提示词字符串
        """
        # 将文档片段格式化为带编号的文本块
        context_text = "\n\n".join([f"文档片段 {i + 1}:\n{doc}" for i, doc in enumerate(context_docs)])
            
        # 构建结构化Prompt
        prompt = f"""请你扮演一个专业的财务报告分析师。
                    根据下面提供的几段从《厦门灿坤实业股份有限公司2019年年度报告》中摘录的文字，
                    严谨且仅依据这些信息来回答用户的问题。
                    如果提供的信息不足以回答问题，请明确告知"根据现有信息无法回答"。
    
                    ---
                    相关文档内容：
                    {context_text}
                    ---
    
                    用户问题：{question}
                    """
        return prompt

    def _call_llm(self, prompt):
        """
        调用DashScope通义千问模型生成答案
        
        参数：
        - prompt: 构建好的提示词
        
        返回：
        - answer: 模型生成的答案，或错误提示
        
        模型选择说明：
        - qwen-turbo: 快速、成本低，适合简单问答
        - qwen-plus: 平衡性能与成本
        - qwen-max: 最强性能，适合复杂推理
        """
        print("\n--- 正在向DashScope LLM发送请求... ---")

        # 调用通义千问API
        # Generation.call: DashScope的统一生成接口
        # 参数说明：
        #   - model: 模型名称
        #   - messages: 对话消息列表，支持多轮对话
        #   - result_format: 返回格式，'message'表示返回标准消息格式
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=[{'role': 'user', 'content': prompt}],
            result_format='message'
        )

        if response.status_code == HTTPStatus.OK:
            # 提取模型回答
            # response.output.choices[0]['message']['content'] 包含生成的文本
            answer = response.output.choices[0]['message']['content']
            print("--- LLM响应成功 ---")
            return answer
        else:
            # 错误处理：打印诊断信息并返回友好提示
            print(f"请求失败：request_id={response.request_id}, status_code={response.status_code}, "
                  f"code={response.code}, message={response.message}")
            return "抱歉，调用大语言模型时出错，无法生成答案。"

    def generate_answer(self, question, context):
        """
        生成答案的完整流程：构建Prompt → 调用LLM → 返回答案
        
        参数：
        - question: 用户问题
        - context: 检索到的文档片段列表
        
        返回：
        - answer: 生成的答案
        - context: 原始上下文（用于调试和展示）
        """
        prompt = self._build_prompt(question, context)
        # 打印Prompt便于调试，生产环境可关闭
        print("\n--- 发送给LLM的最终Prompt ---\n", prompt)
        answer = self._call_llm(prompt)
        return answer, context

    def ask_question(self, question):
        """
        完整的RAG问答流程入口
        
        流程：
        1. 向量检索：调用MinerURAGSystem.query_documents获取相关文档
        2. 空结果处理：未检索到文档时返回提示
        3. 答案生成：调用generate_answer构建Prompt并生成答案
        
        参数：
        - question: 用户问题
        
        返回：
        - answer: 生成的答案
        - results: 完整的检索结果（包含documents、metadatas、distances）
        """
        # 步骤1：向量检索
        results = self.rag_system.query_documents(question)
        
        # 步骤2：检查检索结果
        if not results or not results.get('documents') or not results['documents'][0]:
            return "抱歉，没有找到相关的文档信息。", []

        # 步骤3：生成答案
        answer, context = self.generate_answer(question, results['documents'][0])
        return answer, results


# ==============================================================================
# 第3部分：主调用流程 - 完整RAG pipeline演示
# ==============================================================================
if __name__ == '__main__':
    """
    完整RAG流程演示：
    
    步骤1: 初始化RAG系统
    步骤2: 文档解析（或加载已解析的JSON）
    步骤3: 文本分块与向量化准备
    步骤4: 构建向量数据库
    步骤5: 初始化问答系统
    步骤6: 执行问答并展示结果
    
    工作流程：
    - 首次运行：PDF → MinerU解析 → 分块 → 向量化 → 存储 → 问答
    - 后续运行：直接加载已解析的JSON → 分块 → 向量化 → 存储 → 问答
    """

    # ============================================
    # 步骤1: 初始化RAG系统
    # ============================================
    rag_system = MinerURAGSystem(persist_directory="./my_chroma_db")

    # ============================================
    # 步骤2: 文档解析（智能路径处理）
    # ============================================
    pdf_folder = 'my_pdfs'  # PDF源文件目录
    # 已解析文档的目录（避免重复解析）
    folder_path = 'processed/年度报告/2019年__年度报告/auto'
    processed_files = []

    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print(f"请在当前目录下创建'{pdf_folder}'文件夹并放入PDF文件。")
    else:
        flag = True
        # 优先检查是否已有解析好的JSON文件
        for file in os.listdir(folder_path):
            if file.endswith('.json'):
                print(f"在当前目录下存在json文件。")
                # 递归查找所有_content_list.json文件
                pattern = os.path.join(folder_path, '**', '*_content_list.json')
                processed_files = glob.glob(pattern, recursive=True)
                processed_files = [f for f in processed_files if f]  # 过滤空字符串
                print(f"\n读取了{len(processed_files)}个文件")
                flag = False
                break

        # 如果没有找到JSON文件，则解析PDF
        if flag:
            processed_files = rag_system.process_documents(pdf_directory=pdf_folder)
            print(f"解析完成，生成了 {len(processed_files)} 个内容文件。")

    # ============================================
    # 步骤3: 文本分块与向量化准备
    # ============================================
    print(f"\n开始对{len(processed_files)}个文件进行文本分块与向量化...")
    documents, metadatas, ids = rag_system.chunk_and_embed(content_files=processed_files)
    print(f"已将文档分割成 {len(documents)} 个块。")

    # ============================================
    # 步骤4: 构建并填充向量数据库
    # ============================================
    if documents:
        print("\n开始构建并填充向量数据库...")
        rag_system.build_vector_store(documents, metadatas, ids)

        # ============================================
        # 步骤5: 初始化问答系统
        # ============================================
        qa_system = RAGQASystem(mineru_rag_system=rag_system)
        print("\n问答系统已准备就绪。")

        # ============================================
        # 步骤6: 执行问答
        # ============================================
        question = "2019年度归属于上市公司股东的净利润？"
        # question = "2019年一季度归属于上市公司股东的净利润？"

        print(f"\n--- 正在查询问题 --- \n{question}")
        answer, search_results = qa_system.ask_question(question)

        # ============================================
        # 展示问答结果
        # ============================================
        print("\n\n==================== 问答结果 ====================")
        print("--- 最终回答 ---")
        print(answer)
        print("\n--- 检索到的相关信息 ---")
        if search_results and search_results.get('documents') and search_results['documents'][0]:
            for i, doc in enumerate(search_results['documents'][0]):
                distance = search_results['distances'][0][i]
                metadata = search_results['metadatas'][0][i]
                print(f"\n--- 相关片段 {i + 1} (距离: {distance:.4f}) ---")
                print(f"来源: {metadata.get('source')}")
                print(f"类型: {metadata.get('chunk_type')}")
                print("内容:")
                print(doc)
        else:
            print("没有找到相关的文档片段。")
        print("\n==================================================")
