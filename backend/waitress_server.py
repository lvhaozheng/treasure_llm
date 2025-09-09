#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waitress WSGI服务器启动脚本
适用于Windows环境的生产级WSGI服务器
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

# 设置环境变量
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('PYTHONPATH', str(project_root))

try:
    from waitress import serve
    from wsgi import application
    
    print("✅ 使用Waitress WSGI服务器启动AI古董鉴宝后端")
    print("🌐 服务地址: http://0.0.0.0:5000")
    print("📊 配置: 生产模式，多线程处理")
    print("-" * 50)
    
    # Waitress服务器配置
    serve(
        application,
        host='0.0.0.0',
        port=5000,
        threads=8,  # 线程数
        connection_limit=1000,  # 连接限制
        cleanup_interval=30,  # 清理间隔
        channel_timeout=120,  # 通道超时
        log_socket_errors=True,  # 记录套接字错误
        max_request_body_size=104857600,  # 最大请求体大小 100MB
        expose_tracebacks=False,  # 生产环境不暴露错误堆栈
        ident='AI-Antique-Backend/1.0'  # 服务器标识
    )
    
except ImportError as e:
    print(f"❌ 导入Waitress失败: {e}")
    print("请安装: pip install waitress")
    sys.exit(1)
except Exception as e:
    print(f"❌ 服务器启动失败: {e}")
    sys.exit(1)