"""
Reranker 重排序服务
使用 bge-reranker 模型对检索结果进行精排
"""
import logging
from typing import List, Dict, Optional

from config import Config

logger = logging.getLogger(__name__)


class RerankerService:
    """
    基于 BGE-Reranker 的重排序服务
    支持 bge-reranker-v2-m3（多语言）和 bge-reranker-large（中文）
    """
    
    def __init__(self, model_name: str = None, use_fp16: bool = None):
        """
        初始化 Reranker 服务
        
        Args:
            model_name: 模型名称，支持:
                - BAAI/bge-reranker-v2-m3 (多语言，推荐)
                - BAAI/bge-reranker-large (中文)
                - BAAI/bge-reranker-base (轻量)
            use_fp16: 是否使用 FP16 加速
        """
        # 使用配置文件中的默认值
        self.model_name = model_name or getattr(Config, 'RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
        self.use_fp16 = use_fp16 if use_fp16 is not None else getattr(Config, 'RERANKER_USE_FP16', True)
        self.reranker = None
        self._initialized = False
        
    def _lazy_init(self):
        """延迟初始化，首次调用时加载模型"""
        if self._initialized:
            return
            
        try:
            from FlagEmbedding import FlagReranker
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"初始化 Reranker 模型: {self.model_name}, device: {device}, fp16: {self.use_fp16}")
            
            self.reranker = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
                device=device
            )
            self._initialized = True
            logger.info("Reranker 模型加载完成")
            
        except ImportError as e:
            logger.error(f"FlagEmbedding 未安装，请运行: pip install FlagEmbedding")
            raise ImportError("请安装 FlagEmbedding: pip install FlagEmbedding") from e
        except Exception as e:
            logger.error(f"Reranker 模型加载失败: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前 K 个结果
            
        Returns:
            按相关性排序的结果列表，每个元素包含:
            - index: 原始索引
            - score: 重排序分数 (0-1)
            - text: 文档文本
        """
        if not documents:
            return []
        
        # 延迟初始化
        self._lazy_init()
        
        try:
            # 构建 query-document 对
            pairs = [[query, doc] for doc in documents]
            
            # 计算相关性分数
            scores = self.reranker.compute_score(pairs, normalize=True)
            
            # 如果只有一个文档，scores 是标量
            if isinstance(scores, (int, float)):
                scores = [scores]
            
            # 构建结果
            results = [
                {
                    "index": i,
                    "score": float(scores[i]),
                    "text": documents[i]
                }
                for i in range(len(documents))
            ]
            
            # 按分数降序排序
            results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.debug(f"Rerank 完成: query='{query[:50]}...', docs={len(documents)}, top_k={top_k}")
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Rerank 失败: {e}")
            # 降级：返回原始顺序
            return [
                {"index": i, "score": 0.5, "text": doc}
                for i, doc in enumerate(documents[:top_k])
            ]
    
    def is_available(self) -> bool:
        """检查 Reranker 是否可用"""
        try:
            self._lazy_init()
            return self._initialized
        except:
            return False


# 全局单例
_reranker_service: Optional[RerankerService] = None


def get_reranker_service(model_name: str = None) -> RerankerService:
    """获取 Reranker 服务单例"""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService(model_name=model_name)
    return _reranker_service
