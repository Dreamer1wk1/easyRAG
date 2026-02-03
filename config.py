import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # 星火API配置
    SPARK_APPID = os.getenv("SPARK_APPID")
    SPARK_API_SECRET = os.getenv("SPARK_API_SECRET")
    SPARK_API_KEY = os.getenv("SPARK_API_KEY")
    SPARK_URL = "wss://spark-api.xf-yun.com/v4.0/chat"
    SPARK_DOMAIN = "4.0Ultra"

    # 向量数据库配置
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    VECTOR_DIR = os.getenv("VECTOR_DIR", "./comment_vectors")

    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_USE_HTTPS = os.getenv("CHROMA_USE_HTTPS", "False").lower() == "true"
    CHROMA_AUTH = os.getenv("CHROMA_AUTH", "")

    # 文本分块配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))           # 每块最大字符数（约 250 汉字）
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))     # 块之间的重叠字符数
    CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "char")       # 分块策略: char/semantic/hybrid
    MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "300"))  # 低于此长度不分块
    
    # Reranker 配置
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANKER_USE_FP16 = os.getenv("RERANKER_USE_FP16", "True").lower() == "true"

    # Nacos配置
    NACOS_SERVER_ADDR = os.getenv("NACOS_SERVER_ADDR", "127.0.0.1:8848")
    NACOS_NAMESPACE = os.getenv("NACOS_NAMESPACE", "public")
    NACOS_USERNAME = os.getenv("NACOS_USERNAME", "")
    NACOS_PASSWORD = os.getenv("NACOS_PASSWORD", "")


    # 服务配置
    SERVICE_NAME = os.getenv("SERVICE_NAME", "easyrag-service")
    SERVICE_PORT = int(os.getenv("SERVICE_PORT", "5000"))
    SERVICE_WEIGHT = float(os.getenv("SERVICE_WEIGHT", "1.0"))
    SERVICE_CLUSTER = os.getenv("SERVICE_CLUSTER", "DEFAULT")
    SERVICE_GROUP = os.getenv("SERVICE_GROUP", "DEFAULT_GROUP")
    SERVICE_EPHEMERAL = os.getenv("SERVICE_EPHEMERAL", "True").lower() == "true"