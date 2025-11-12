import socket
import logging
import threading
import time
from nacos import NacosClient
from config import Config


class NacosService:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(NacosService, cls).__new__(cls)
            cls._instance._init_nacos_client()
        return cls._instance

    def _init_nacos_client(self):
        """初始化 Nacos 客户端"""
        try:
            params = {
                'server_addresses': Config.NACOS_SERVER_ADDR,
                'namespace': Config.NACOS_NAMESPACE
            }
            if getattr(Config, 'NACOS_USERNAME', None):
                params['username'] = Config.NACOS_USERNAME
            if getattr(Config, 'NACOS_PASSWORD', None):
                params['password'] = Config.NACOS_PASSWORD

            self.client = NacosClient(**params)
            logging.info(f'连接 Nacos 成功: {Config.NACOS_SERVER_ADDR}')
        except Exception as e:
            logging.error(f'连接 Nacos 失败: {e}')
            self.client = None

    def register(self):
        """注册服务到 Nacos 并启动心跳线程"""
        if not self.client:
            return False

        ip = socket.gethostbyname(socket.gethostname())
        try:
            self.client.add_naming_instance(
                service_name=Config.SERVICE_NAME,
                ip=ip,
                port=Config.SERVICE_PORT,
                weight=Config.SERVICE_WEIGHT,
                cluster_name=Config.SERVICE_CLUSTER,
                group_name=Config.SERVICE_GROUP,
                ephemeral=Config.SERVICE_EPHEMERAL
            )
            logging.info(f'服务已注册: {Config.SERVICE_NAME} ({ip}:{Config.SERVICE_PORT})')

            if Config.SERVICE_EPHEMERAL:
                # 启动后台心跳线程
                def heartbeat_loop():
                    while True:
                        try:
                            self.client.send_heartbeat(
                                service_name=Config.SERVICE_NAME,
                                ip=ip,
                                port=Config.SERVICE_PORT,
                                cluster_name=Config.SERVICE_CLUSTER,
                                group_name=Config.SERVICE_GROUP
                            )
                            logging.info(f'💓 心跳成功: {ip}:{Config.SERVICE_PORT}')
                            time.sleep(5)
                        except Exception as e:
                            logging.error(f'❌ Nacos 心跳失败: {e}')
                            time.sleep(10)

                t = threading.Thread(target=heartbeat_loop, daemon=True)
                t.start()
                logging.info("Nacos 心跳线程已启动")

            return True
        except Exception as e:
            logging.error(f'Nacos 注册失败: {e}')
            return False

    def deregister(self):
        """注销服务"""
        if not self.client:
            return
        try:
            ip = socket.gethostbyname(socket.gethostname())
            self.client.remove_naming_instance(
                service_name=Config.SERVICE_NAME,
                ip=ip,
                port=Config.SERVICE_PORT,
                cluster_name=Config.SERVICE_CLUSTER,
                group_name=Config.SERVICE_GROUP
            )
            logging.info(f'服务已注销: {Config.SERVICE_NAME}')
        except Exception as e:
            logging.error(f'Nacos 注销失败: {e}')
