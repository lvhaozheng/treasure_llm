# AI鉴宝师项目启动指南

本项目提供了多种启动方式，适应不同的开发和部署需求。

## 📋 目录

- [快速启动](#快速启动)
- [启动脚本说明](#启动脚本说明)
- [系统要求](#系统要求)
- [配置说明](#配置说明)
- [故障排除](#故障排除)

## 🚀 快速启动

### Windows 用户
```bash
# 双击运行或在命令行执行
start.bat
```

### Linux/Mac 用户
```bash
# 添加执行权限并运行
chmod +x start.sh
./start.sh
```

### 跨平台 Python 启动
```bash
# 安装依赖（可选）
pip install colorama python-dotenv requests

# 运行启动脚本
python start.py
```

## 📝 启动脚本说明

### 1. 混合启动模式（推荐开发环境）

**文件：** `start.bat` / `start.sh` / `start.py`

**特点：**
- Docker 运行依赖服务（Milvus、Redis）
- 本地运行应用服务（前端、后端）
- 支持热重载和调试
- 启动速度快

**适用场景：**
- 日常开发调试
- 代码修改和测试
- 性能调优

### 2. 完全容器化部署

**文件：** `docker-compose.full.yml`

```bash
# 启动所有服务
docker-compose -f docker-compose.full.yml up -d

# 启动开发环境（包含管理界面）
docker-compose -f docker-compose.full.yml --profile development up -d

# 启动生产环境（包含Nginx）
docker-compose -f docker-compose.full.yml --profile production up -d

# 启动监控环境
docker-compose -f docker-compose.full.yml --profile monitoring up -d
```

**适用场景：**
- 生产环境部署
- CI/CD 流水线
- 多环境一致性

### 3. 仅依赖服务

**文件：** `docker-compose.dependencies.yml`

```bash
# 仅启动依赖服务
docker-compose -f docker-compose.dependencies.yml up -d

# 包含管理界面
docker-compose -f docker-compose.dependencies.yml --profile management up -d
```

**适用场景：**
- 自定义应用启动方式
- 微服务开发
- 服务解耦测试

## 🔧 系统要求

### 基础要求
- **Python:** 3.9+
- **Node.js:** 16+
- **Docker:** 20.10+
- **Docker Compose:** 2.0+

### 硬件要求
- **内存:** 8GB+ （推荐 16GB）
- **存储:** 20GB+ 可用空间
- **GPU:** NVIDIA GPU（可选，用于AI模型加速）

### 可选依赖
```bash
# Python 增强功能
pip install colorama python-dotenv requests

# Conda 环境管理
conda create -n treasure_llm python=3.9
conda activate treasure_llm
```

## ⚙️ 配置说明

### 环境变量配置

创建 `.env` 文件（参考 `.env.example`）：

```env
# 基础配置
FLASK_ENV=development
SECRET_KEY=your-secret-key

# 数据库配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
REDIS_HOST=localhost
REDIS_PORT=6379

# AI模型配置
USE_LOCAL_MODELS=true
QWEN3_MODEL_PATH=/path/to/qwen3
SMOLVLM2_MODEL_PATH=/path/to/smolvlm2

# API配置
OPENAI_API_KEY=your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 端口配置

| 服务 | 端口 | 说明 |
|------|------|------|
| 前端 | 3000 | React 开发服务器 |
| 后端 | 5000 | Flask API 服务 |
| Milvus | 19530 | 向量数据库 |
| Redis | 6379 | 缓存服务 |
| Minio | 9000/9001 | 对象存储 |
| Attu | 3001 | Milvus 管理界面 |
| Nginx | 80/443 | 反向代理 |
| Grafana | 3002 | 监控面板 |
| Prometheus | 9090 | 监控数据 |

## 🔍 服务访问地址

启动成功后，可以通过以下地址访问各项服务：

- **主应用:** http://localhost:3000
- **API文档:** http://localhost:5000/health
- **Milvus管理:** http://localhost:3001 （开发模式）
- **监控面板:** http://localhost:3002 （监控模式）

## 🛠️ 故障排除

### 常见问题

#### 1. Docker 服务启动失败
```bash
# 检查 Docker 状态
docker --version
docker-compose --version

# 清理并重启
docker-compose down
docker system prune -f
docker-compose up -d
```

#### 2. 端口被占用
```bash
# Windows 查看端口占用
netstat -ano | findstr :3000
taskkill /f /pid <PID>

# Linux/Mac 查看端口占用
lsof -i :3000
kill -9 <PID>
```

#### 3. 前端依赖安装失败
```bash
cd frontend

# 清理缓存
npm cache clean --force
rm -rf node_modules package-lock.json

# 重新安装
npm install
```

#### 4. 后端 AI 模型加载失败
```bash
# 检查模型路径
ls -la /path/to/models

# 检查 Python 环境
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"

# 重新安装依赖
pip install -r backend/requirements.txt
```

#### 5. Milvus 连接失败
```bash
# 检查 Milvus 状态
docker logs milvus-standalone

# 重启 Milvus
docker-compose restart milvus

# 等待服务就绪
curl http://localhost:9091/healthz
```

### 日志查看

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f milvus
docker-compose logs -f redis

# 查看应用日志
tail -f logs/backend/app.log
tail -f logs/system/system.log
```

### 性能优化

#### Docker 资源配置
```yaml
# docker-compose.yml 中添加资源限制
services:
  milvus:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

#### GPU 支持
```yaml
# 启用 GPU 支持
services:
  backend:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 📞 技术支持

如果遇到问题，请按以下步骤操作：

1. 查看本文档的故障排除部分
2. 检查日志文件获取详细错误信息
3. 确认系统要求和配置是否正确
4. 提交 Issue 时请附上：
   - 操作系统信息
   - 错误日志
   - 复现步骤

## 🔄 更新和维护

### 更新代码
```bash
git pull origin main

# 重新构建（如有必要）
docker-compose build --no-cache

# 重启服务
docker-compose down
docker-compose up -d
```

### 数据备份
```bash
# 备份 Milvus 数据
docker run --rm -v milvus_data:/data -v $(pwd):/backup alpine tar czf /backup/milvus_backup.tar.gz /data

# 备份 Redis 数据
docker run --rm -v redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis_backup.tar.gz /data
```

---

**注意：** 首次启动可能需要较长时间来下载 Docker 镜像和安装依赖，请耐心等待。