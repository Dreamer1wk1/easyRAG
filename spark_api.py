import json
import ssl
import websocket
from threading import Thread
from datetime import datetime
from time import mktime
from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time
import hashlib
import hmac
import base64
from config import Config


class SparkAPI:
    def __init__(self):
        self.ws = None
        self.response = ""
        self.connected = False

    def _create_url(self):
        """生成鉴权URL"""
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = f"host: {urlparse(Config.SPARK_URL).netloc}\n"
        signature_origin += f"date: {date}\nGET {urlparse(Config.SPARK_URL).path} HTTP/1.1"

        signature_sha = hmac.new(
            Config.SPARK_API_SECRET.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()

        authorization = base64.b64encode(
            f'api_key="{Config.SPARK_API_KEY}", algorithm="hmac-sha256", headers="host date request-line", signature="{base64.b64encode(signature_sha).decode()}"'.encode()
        ).decode()

        return f"{Config.SPARK_URL}?{urlencode({'authorization': authorization, 'date': date, 'host': urlparse(Config.SPARK_URL).netloc})}"

    def _on_message(self, ws, message):
        """处理返回消息"""
        data = json.loads(message)
        if data['header']['code'] != 0:
            print(f"Error: {data['header']['message']}")
            return

        for text in data["payload"]["choices"]["text"]:
            self.response += text['content']

        if data['payload']['choices']['status'] == 2:
            self.connected = False
            ws.close()

    def _on_error(self, ws, error):
        """处理WebSocket错误"""
        print("WebSocket error:", error)
        self.connected = False

    def _on_close(self, ws):
        """处理WebSocket关闭"""
        print("WebSocket connection closed")
        self.connected = False

    def _on_open(self, ws):
        """处理WebSocket连接建立"""
        self.connected = True
        print("WebSocket connection opened")

    def get_response(self, query: str) -> str:
        """获取大模型回复"""
        self.response = ""

        # 创建WebSocket连接
        ws = websocket.WebSocketApp(
            self._create_url(),
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )

        # 启动WebSocket线程
        ws_thread = Thread(target=ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}})
        ws_thread.start()

        # 等待连接建立
        while not self.connected:
            pass

        # 发送请求数据
        data = json.dumps({
            "header": {"app_id": Config.SPARK_APPID, "uid": "1234"},
            "parameter": {"chat": {"domain": Config.SPARK_DOMAIN, "temperature": 0.5}},
            "payload": {"message": {"text": [{"role": "user", "content": query}]}}
        })
        ws.send(data)

        # 等待响应完成
        ws_thread.join()
        return self.response