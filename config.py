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
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    VECTOR_DIR = "./comment_vectors"

    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_USE_HTTPS = os.getenv("CHROMA_USE_HTTPS", "False").lower() == "true"
    CHROMA_AUTH = os.getenv("CHROMA_AUTH", "")

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