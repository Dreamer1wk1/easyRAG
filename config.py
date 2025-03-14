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