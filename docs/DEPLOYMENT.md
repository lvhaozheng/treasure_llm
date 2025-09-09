# AI 鉴宝师项目部署指南（简化版）

## 概述

本文档介绍了如何使用 Docker Compose 部署简化版的 AI 鉴宝师项目。项目专注于核心的AI鉴宝功能，图片本地存储，无需数据库和复杂的监控系统。

## 系统要求

### 硬件要求
- **CPU**: 至少 4 核
- **内存**: 至少 8GB RAM
- **存储**: 至少 20GB 可用空间
- **网络**: 稳定的互联网连接

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+), macOS 10.15+, Windows 10+
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Git**: 最新版本
- **Python**: 3.9+ (如果本地开发)

### 重要依赖版本 (2025-08-30 更新)
为支持LLaVA本地多模态大模型，项目依赖已更新至以下版本：
- **PyTorch**: >=2.1.0 (从2.0.1升级，支持最新transformers)
- **Transformers**: >=4.56.0 (从4.35.0升级，完整支持LLaVA)
- **TorchVision**: >=0.15.0 (配套PyTorch升级)
- **HuggingFace Hub**: >=0.34.0 (支持最新模型格式)
- **Tokenizers**: >=0.22.0 (配套transformers升级)
- **SafeTensors**: >=0.4.3 (更好的模型文件支持)
- **OpenAI-CLIP**: 替代原clip库 (更好的兼容性)

> **注意**: 如果您从旧版本升级，建议重新构建Docker镜像或更新conda环境

## 快速部署

### 1. 克隆项目
```bash
git clone <repository-url>
cd ai-antique-appraiser
```

### 2. 配置环境变量
```bash
# 复制环境变量模板
cp env.example .env

# 编辑环境变量文件
nano .env  # Linux/macOS
# 或
notepad .env  # Windows
```

**重要配置项：**
```env
# OpenAI API 密钥（必需）
OPENAI_API_KEY=your-openai-api-key-here

# 安全密钥（必需）
SECRET_KEY=your-secret-key-here

# Milvus 配置
MILVUS_HOST=milvus
MILVUS_PORT=19530
```

### 3. 一键部署

**Linux/macOS:**
```bash
chmod +x deploy.sh
./deploy.sh
```

**Windows:**
```cmd
deploy.bat
```

### 4. 验证部署
部署完成后，访问以下地址验证服务状态：

- **前端应用**: http://localhost
- **后端 API**: http://localhost/api
- **MinIO 控制台**: http://localhost:9001 (minioadmin/minioadmin)

## 详细部署步骤

### 1. 环境准备

#### 安装 Docker
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# macOS
brew install --cask docker

# Windows
# 下载并安装 Docker Desktop
```

#### 安装 Docker Compose
```bash
# 安装最新版本
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. 项目配置

#### 创建必要目录
```bash
mkdir -p uploads data logs nginx/logs
```

#### 配置环境变量
编辑 `.env` 文件，设置以下关键配置：

```env
# Flask 配置
FLASK_ENV=production
SECRET_KEY=your-very-secure-secret-key-here

# Milvus 向量数据库
MILVUS_HOST=milvus
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=antique_vectors

# OpenAI API
OPENAI_API_KEY=sk-your-openai-api-key-here

# 文件上传
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216
ALLOWED_EXTENSIONS=jpg,jpeg,png,gif,bmp

# AI 模型配置
CLIP_MODEL_NAME=ViT-B/32
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### 3. 构建和启动服务

#### 构建镜像
```bash
docker-compose build --no-cache
```

#### 启动服务
```bash
docker-compose up -d
```

#### 检查服务状态
```bash
docker-compose ps
```

### 4. 服务初始化

#### 等待服务就绪
```bash
# 等待 Milvus 就绪
docker-compose exec milvus curl -f http://localhost:9091/healthz
```

## 服务架构

### 核心服务
1. **Frontend (React)**: 前端用户界面
2. **Backend (Flask)**: 后端 API 服务
3. **Milvus**: 向量数据库

### 基础设施服务
1. **Nginx**: 反向代理
2. **Etcd**: Milvus 元数据存储
3. **MinIO**: 对象存储服务

## 服务端口映射

| 服务 | 内部端口 | 外部端口 | 说明 |
|------|----------|----------|------|
| Frontend | 3000 | 3000 | React 应用 |
| Backend | 5000 | 5000 | Flask API |
| Nginx | 80 | 80 | 反向代理 |
| Milvus | 19530 | 19530 | 向量数据库 |
| Milvus Metrics | 9091 | 9091 | Milvus 监控 |
| MinIO | 9000 | 9000 | 对象存储 |
| MinIO Console | 9001 | 9001 | MinIO 控制台 |

## 管理命令

### 服务管理
```bash
# 启动所有服务
docker-compose up -d

# 停止所有服务
docker-compose down

# 重启所有服务
docker-compose restart

# 查看服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
```

### 数据管理
```bash
# 查看上传的图片
ls -la uploads/

# 查看分析结果
ls -la data/

# 清理数据
docker-compose exec backend curl -X POST http://localhost:5000/api/clear
```

## API 接口

### 核心功能接口

#### 1. 图像分析
```bash
POST /api/analyze
Content-Type: multipart/form-data

参数:
- image: 图片文件
- description: 描述信息（可选）
```

#### 2. 文本搜索
```bash
POST /api/search
Content-Type: application/json

{
  "query": "明清时期的陶瓷花瓶",
  "top_k": 10
}
```

#### 3. AI 对话
```bash
POST /api/chat
Content-Type: application/json

{
  "message": "请介绍一下明代青花瓷的特点"
}
```

#### 4. 上传古董
```bash
POST /api/upload
Content-Type: multipart/form-data

参数:
- image: 图片文件
- antique_id: 古董ID
- description: 描述
- name: 名称
- category: 类别
- dynasty: 朝代
- material: 材质
```

#### 5. 系统状态
```bash
GET /api/status
```

#### 6. 健康检查
```bash
GET /health
```

## 故障排除

### 常见问题

#### 1. 服务启动失败
```bash
# 查看详细错误信息
docker-compose logs

# 检查端口占用
netstat -tulpn | grep :5000
```

#### 2. Milvus 连接失败
```bash
# 检查 Milvus 状态
docker-compose exec milvus curl -f http://localhost:9091/healthz

# 查看 Milvus 日志
docker-compose logs milvus
```

#### 3. AI 服务未就绪
```bash
# 检查 OpenAI API 密钥
echo $OPENAI_API_KEY

# 查看后端日志
docker-compose logs backend
```

#### 4. 内存不足
```bash
# 检查系统资源
docker stats

# 清理 Docker 资源
docker system prune -f
```

### 日志分析
```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定时间段的日志
docker-compose logs --since="2023-01-01T00:00:00" --until="2023-01-01T23:59:59"

# 查看错误日志
docker-compose logs | grep ERROR
```

## 性能优化

### 系统优化
1. **增加内存**: 建议至少 16GB RAM
2. **使用 SSD**: 提高 I/O 性能
3. **网络优化**: 使用千兆网络

### Docker 优化
```bash
# 增加 Docker 内存限制
# 在 Docker Desktop 设置中调整内存限制

# 使用多阶段构建
# 已在 Dockerfile 中实现

# 优化镜像大小
docker system prune -f
```

### 应用优化
1. **图片压缩**: 上传前压缩图片
2. **批量处理**: 支持批量上传和分析
3. **缓存策略**: 本地文件缓存

## 安全配置

### 网络安全
1. **防火墙配置**: 只开放必要端口
2. **访问控制**: 限制管理端口访问

### 应用安全
1. **环境变量**: 敏感信息使用环境变量
2. **密钥管理**: 定期更换密钥
3. **文件验证**: 验证上传文件类型

### 数据安全
1. **本地存储**: 图片和分析结果本地存储
2. **定期备份**: 备份重要数据
3. **访问控制**: 限制文件访问权限

## 备份和恢复

### 数据备份
```bash
# 备份上传文件
tar -czf uploads_backup_$(date +%Y%m%d_%H%M%S).tar.gz uploads/

# 备份分析结果
tar -czf data_backup_$(date +%Y%m%d_%H%M%S).tar.gz data/

# 备份配置文件
tar -czf config_backup_$(date +%Y%m%d_%H%M%S).tar.gz .env docker-compose.yml
```

### 数据恢复
```bash
# 恢复上传文件
tar -xzf uploads_backup.tar.gz

# 恢复分析结果
tar -xzf data_backup.tar.gz

# 恢复配置文件
tar -xzf config_backup.tar.gz
```

## 更新和维护

### 应用更新
```bash
# 拉取最新代码
git pull

# 重新构建镜像
docker-compose build --no-cache

# 重启服务
docker-compose up -d
```

### 系统维护
```bash
# 清理 Docker 资源
docker system prune -f

# 更新 Docker 镜像
docker-compose pull

# 检查系统资源
docker stats
```

## 技术支持

如遇到部署问题，请：

1. 查看本文档的故障排除部分
2. 检查服务日志：`docker-compose logs`
3. 确认环境变量配置正确
4. 验证系统资源是否充足
5. 联系技术支持团队

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。
