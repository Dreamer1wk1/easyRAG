import os
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config

logger = logging.getLogger(__name__)


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


def get_text_splitter():
    """
    获取优化后的文本分割器
    
    优化点：
    1. chunk_size 从 200 增大到 500（约 250 汉字），保持语义完整性
    2. chunk_overlap 从 50 增大到 100，更好的上下文连贯
    3. 修复分隔符，添加完整的中文标点支持
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",              # 1. 段落分隔（最高优先级）
            "\n",                # 2. 换行
            "。",                # 3. 句号
            "！",                # 4. 感叹号
            "？",                # 5. 问号
            "；",                # 6. 分号
            "……",               # 7. 省略号
            "...",               # 8. 英文省略号
            "，",                # 9. 逗号（尽量不在此处切）
            " ",                 # 10. 空格
            ""                   # 11. 最后按字符切
        ]
    )


def delete_text_by_metadata(filter: dict):
    """根据元数据删除文本"""
    vector_store = VectorStore()
    results = vector_store.get(where=filter)
    if results and "ids" in results:
        vector_store.delete(ids=results["ids"])


def process_text(text: str, metadata: dict = None):
    """
    处理、切分并存储文本
    
    根据 CHUNK_STRATEGY 配置选择分块策略：
    - "char": 字符分块（默认，快速）
    - "semantic": 语义分块（精准，需要额外依赖）
    - "hybrid": 混合策略（根据文本长度自动选择）
    """
    strategy = getattr(Config, 'CHUNK_STRATEGY', 'char')
    min_chunk_length = getattr(Config, 'MIN_CHUNK_LENGTH', 300)
    
    # 短文本不分块，直接存储
    if len(text) < min_chunk_length:
        doc = Document(page_content=text, metadata=metadata or {})
        vector_store = VectorStore()
        vector_store.add_documents([doc])
        logger.debug(f"短文本直接存储，长度: {len(text)}")
        return
    
    # 根据策略选择分块方式
    if strategy == "semantic":
        chunks = _semantic_split(text, metadata)
    elif strategy == "hybrid":
        chunks = _hybrid_split(text, metadata)
    else:  # 默认 char
        chunks = _char_split(text, metadata)
    
    # 存储到向量库
    if chunks:
        vector_store = VectorStore()
        vector_store.add_documents(chunks)
        logger.debug(f"文本分块完成，策略: {strategy}, 块数: {len(chunks)}")


def _char_split(text: str, metadata: dict = None) -> list:
    """字符分块（快速）"""
    text_splitter = get_text_splitter()
    doc = Document(page_content=text, metadata=metadata or {})
    return text_splitter.split_documents([doc])


def _semantic_split(text: str, metadata: dict = None) -> list:
    """语义分块（精准）"""
    try:
        from text_splitter import get_semantic_splitter
        splitter = get_semantic_splitter()
        doc = Document(page_content=text, metadata=metadata or {})
        return splitter.split_documents([doc])
    except ImportError:
        logger.warning("语义分块依赖未安装，降级到字符分块")
        return _char_split(text, metadata)
    except Exception as e:
        logger.warning(f"语义分块失败，降级到字符分块: {e}")
        return _char_split(text, metadata)


def _hybrid_split(text: str, metadata: dict = None) -> list:
    """混合分块（根据文本长度自动选择）"""
    try:
        from text_splitter import get_hybrid_splitter
        splitter = get_hybrid_splitter()
        return splitter.split_text(text, metadata)
    except ImportError:
        logger.warning("混合分块依赖未安装，降级到字符分块")
        return _char_split(text, metadata)
    except Exception as e:
        logger.warning(f"混合分块失败，降级到字符分块: {e}")
        return _char_split(text, metadata)
