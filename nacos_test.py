#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Nacos服务注册测试脚本"""

import os
import sys
import logging
from nacos import NacosClient
import socket

# 设置日志级别
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('nacos_test')

# 获取配置
NACOS_SERVER_ADDR = os.getenv("NACOS_SERVER_ADDR", "47.119.40.192:8848")
NACOS_NAMESPACE = os.getenv("NACOS_NAMESPACE", "public")
NACOS_USERNAME = os.getenv("NACOS_USERNAME", "")
NACOS_PASSWORD = os.getenv("NACOS_PASSWORD", "")
SERVICE_NAME = os.getenv("SERVICE_NAME", "easyrag-service")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "5000"))
SERVICE_WEIGHT = float(os.getenv("SERVICE_WEIGHT", "1.0"))
SERVICE_CLUSTER = os.getenv("SERVICE_CLUSTER", "DEFAULT")
SERVICE_GROUP = os.getenv("SERVICE_GROUP", "DEFAULT_GROUP")
SERVICE_EPHEMERAL = os.getenv("SERVICE_EPHEMERAL", "True").lower() == "true"


def test_nacos_connection():
    """测试Nacos连接和服务注册"""
    try:
        # 创建Nacos客户端
        logger.info(f'尝试连接Nacos服务器: {NACOS_SERVER_ADDR}, namespace: {NACOS_NAMESPACE}')
        
        client_params = {
            'server_addresses': NACOS_SERVER_ADDR,
            'namespace': NACOS_NAMESPACE
        }
        
        if NACOS_USERNAME:
            client_params['username'] = NACOS_USERNAME
        if NACOS_PASSWORD:
            client_params['password'] = NACOS_PASSWORD
            
        client = NacosClient(**client_params)
        logger.info('Nacos客户端创建成功')
        
        # 获取本机IP
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        logger.info(f'获取本机IP成功: {ip}')
        
        # 尝试注册服务
        logger.info(f'尝试注册服务: {SERVICE_NAME}({ip}:{SERVICE_PORT})')
        result = client.add_naming_instance(
            service_name=SERVICE_NAME,
            ip=ip,
            port=SERVICE_PORT,
            weight=SERVICE_WEIGHT,
            cluster_name=SERVICE_CLUSTER,
            group_name=SERVICE_GROUP,
            ephemeral=SERVICE_EPHEMERAL
        )
        logger.info(f'服务注册结果: {result}')
        
        # 查看已注册服务列表，验证是否注册成功
        logger.info('查询服务实例列表...')
        instances = client.list_naming_instance(
            service_name=SERVICE_NAME,
            group_name=SERVICE_GROUP,
            clusters=[SERVICE_CLUSTER]
        )
        logger.info(f'服务实例列表: {instances}')
        
        # 发送心跳
        if SERVICE_EPHEMERAL:
            logger.info('发送心跳...')
            heartbeat_result = client.send_heartbeat(
                service_name=SERVICE_NAME,
                ip=ip,
                port=SERVICE_PORT,
                cluster_name=SERVICE_CLUSTER,
                group_name=SERVICE_GROUP
            )
            logger.info(f'心跳发送结果: {heartbeat_result}')
        
        logger.info('Nacos服务注册测试完成！')
        return True
        
    except Exception as e:
        logger.error(f'Nacos测试失败: {str(e)}', exc_info=True)
        return False


if __name__ == '__main__':
    success = test_nacos_connection()
    sys.exit(0 if success else 1)