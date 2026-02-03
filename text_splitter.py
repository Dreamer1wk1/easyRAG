"""
文本分块器模块
提供多种分块策略：字符分块、语义分块、混合分块
"""
import logging
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config

logger = logging.getLogger(__name__)

# 全局单例
_semantic_splitter = None
_hybrid_splitter = None


def get_char_splitter() -> RecursiveCharacterTextSplitter:
    """
    获取字符分块器
    
    特点：快速、无额外依赖
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",              # 1. 段落分隔
            "\n",                # 2. 换行
            "。",                # 3. 句号
            "！",                # 4. 感叹号
            "？",                # 5. 问号
            "；",                # 6. 分号
            "……",               # 7. 省略号
            "...",               # 8. 英文省略号
            "，",                # 9. 逗号
            " ",                 # 10. 空格
            ""                   # 11. 字符
        ]
    )


class SemanticTextSplitter:
    """
    基于语义的文本分割器
    
    原理：计算相邻句子的嵌入相似度，在相似度低的地方切分
    优点：保持语义完整性
    缺点：需要额外计算嵌入向量，速度较慢
    """
    
    def __init__(self, breakpoint_threshold_type: str = "percentile",
                 breakpoint_threshold_amount: float = 0.5):
        """
        初始化语义分块器
        
        Args:
            breakpoint_threshold_type: 断点阈值类型
                - "percentile": 百分位数（推荐，稳定）
                - "standard_deviation": 标准差
                - "interquartile": 四分位距
            breakpoint_threshold_amount: 阈值大小，越小切分越细
        """
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self._splitter = None
        self._initialized = False
    
    def _lazy_init(self):
        """延迟初始化"""
        if self._initialized:
            return
        
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            
            embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
            self._splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.breakpoint_threshold_amount
            )
            self._initialized = True
            logger.info("语义分块器初始化完成")
            
        except ImportError as e:
            raise ImportError(
                "语义分块需要安装 langchain-experimental: "
                "pip install langchain-experimental>=0.0.47"
            ) from e
    
    def split_text(self, text: str) -> List[str]:
        """切分文本，返回字符串列表"""
        self._lazy_init()
        return self._splitter.split_text(text)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档列表"""
        self._lazy_init()
        return self._splitter.split_documents(documents)


class HybridTextSplitter:
    """
    混合文本分割器
    
    根据文本长度自动选择最佳分块策略：
    - 短文本（<300字）: 不分块
    - 中等文本（300-1000字）: 字符分块（速度优先）
    - 长文本（>1000字）: 语义分块（质量优先）
    """
    
    def __init__(self):
        self._char_splitter = get_char_splitter()
        self._semantic_splitter = None
        self._semantic_available = None
    
    def _get_semantic_splitter(self) -> Optional[SemanticTextSplitter]:
        """获取语义分块器（带缓存和可用性检查）"""
        if self._semantic_available is False:
            return None
        
        if self._semantic_splitter is None:
            try:
                self._semantic_splitter = SemanticTextSplitter()
                self._semantic_splitter._lazy_init()  # 触发初始化检查
                self._semantic_available = True
            except ImportError:
                logger.warning("语义分块依赖未安装，混合模式将仅使用字符分块")
                self._semantic_available = False
                return None
        
        return self._semantic_splitter
    
    def split_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        根据文本特征选择分块策略
        
        Args:
            text: 待分块的文本
            metadata: 元数据
            
        Returns:
            分块后的文档列表
        """
        text_len = len(text)
        min_chunk_length = getattr(Config, 'MIN_CHUNK_LENGTH', 300)
        
        # 短文本：不分块
        if text_len < min_chunk_length:
            logger.debug(f"短文本不分块: {text_len} < {min_chunk_length}")
            return [Document(page_content=text, metadata=metadata or {})]
        
        # 中等文本（<1000字）：字符分块（速度优先）
        if text_len < 1000:
            logger.debug(f"中等文本使用字符分块: {text_len}")
            doc = Document(page_content=text, metadata=metadata or {})
            return self._char_splitter.split_documents([doc])
        
        # 长文本（>=1000字）：尝试语义分块（质量优先）
        semantic_splitter = self._get_semantic_splitter()
        if semantic_splitter:
            try:
                logger.debug(f"长文本使用语义分块: {text_len}")
                doc = Document(page_content=text, metadata=metadata or {})
                return semantic_splitter.split_documents([doc])
            except Exception as e:
                logger.warning(f"语义分块失败，降级到字符分块: {e}")
        
        # 降级：字符分块
        logger.debug(f"降级到字符分块: {text_len}")
        doc = Document(page_content=text, metadata=metadata or {})
        return self._char_splitter.split_documents([doc])


def get_semantic_splitter() -> SemanticTextSplitter:
    """获取语义分块器单例"""
    global _semantic_splitter
    if _semantic_splitter is None:
        _semantic_splitter = SemanticTextSplitter()
    return _semantic_splitter


def get_hybrid_splitter() -> HybridTextSplitter:
    """获取混合分块器单例"""
    global _hybrid_splitter
    if _hybrid_splitter is None:
        _hybrid_splitter = HybridTextSplitter()
    return _hybrid_splitter
