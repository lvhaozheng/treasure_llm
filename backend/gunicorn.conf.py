# Gunicorn 配置文件
# 用于生产环境部署Flask应用

import multiprocessing
import os

# 服务器套接字
bind = "0.0.0.0:5000"
backlog = 2048

# 工作进程
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True

# 超时设置
timeout = 120  # 工作进程超时时间（秒）
keepalive = 5  # Keep-Alive连接超时时间
graceful_timeout = 30  # 优雅关闭超时时间

# 日志配置
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 进程命名
proc_name = "ai_antique_backend"

# 安全设置
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# 性能优化
max_requests_jitter = 50
preload_app = True

# 环境变量
raw_env = [
    'FLASK_ENV=production',
    'PYTHONPATH=/app'
]

# 钩子函数
def on_starting(server):
    server.log.info("AI古董鉴宝后端服务启动中...")

def on_reload(server):
    server.log.info("AI古董鉴宝后端服务重新加载...")

def when_ready(server):
    server.log.info("AI古董鉴宝后端服务已就绪")

def on_exit(server):
    server.log.info("AI古董鉴宝后端服务已停止")

def worker_int(worker):
    worker.log.info("工作进程收到中断信号")

def pre_fork(server, worker):
    server.log.info(f"工作进程 {worker.pid} 即将启动")

def post_fork(server, worker):
    server.log.info(f"工作进程 {worker.pid} 已启动")

def post_worker_init(worker):
    worker.log.info(f"工作进程 {worker.pid} 初始化完成")

def worker_abort(worker):
    worker.log.info(f"工作进程 {worker.pid} 异常终止")