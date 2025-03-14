from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
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
    # 获取符合过滤条件的文档ID
    results = vector_store.get(where=filter)
    if results and "ids" in results:
        vector_store.delete(ids=results["ids"])

def process_text(text: str, metadata: dict = None):
    """处理并存储文本"""
    # 将文本转换为Document对象
    doc = Document(page_content=text, metadata=metadata or {})

    # 存储到向量库
    vector_store = VectorStore()
    vector_store.add_documents([doc])  # 注意：add_documents接受的是Document列表

    # 如果需要持久化，直接使用Chroma的持久化机制
    # 最新版本的Chroma会自动将数据持久化到persist_directory