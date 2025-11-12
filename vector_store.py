from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config


class VectorStore:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
            cls._instance = Chroma(
                collection_name="comment",
                embedding_function=embeddings,
                persist_directory=Config.VECTOR_DIR
            )
        return cls._instance

def delete_text_by_metadata(filter: dict):
    """根据元数据删除文本"""
    vector_store = VectorStore()
    results = vector_store.get(where=filter)
    if results and "ids" in results:
        vector_store.delete(ids=results["ids"])


def process_text(text: str, metadata: dict = None):
    """处理、切分并存储文本"""
    # 1. 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # 每块长度（字符数）
        chunk_overlap=50,  # 块之间的重叠量
        separators=["\n\n", "\n", "。", " ；", " "]  # 切分优先级
    )

    # 2. 将文本转换为 Document 并切分
    doc = Document(page_content=text, metadata=metadata or {})
    chunks = text_splitter.split_documents([doc])  # 输入必须是文档列表

    # 3. 存储到向量库
    vector_store = VectorStore()
    vector_store.add_documents(chunks)  # 存储所有切分后的块
