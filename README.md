# AI 鉴宝师项目（简化版）

基于多模态大模型的智能古董鉴定和分析系统，专注于核心AI鉴宝功能。

## 🎯 项目特色

- **多模态AI分析**: 结合CLIP图像编码和多种视觉语言模型
- **灵活模型选择**: 默认使用OmniVision轻量级模型，可选择LLaVA高性能模型
- **轻量级优先**: OmniVision（968M参数）适合边缘设备，启动快速
- **向量数据库**: 使用Milvus进行高效的相似古董搜索
- **本地存储**: 图片和分析结果本地存储，无需复杂数据库
- **轻量部署**: 简化的Docker部署，最小化外部依赖
- **实时分析**: 支持图像上传、文本搜索、AI对话等功能
- **智能回退**: 多层回退机制，确保在各种硬件条件下稳定运行

## 🏗️ 技术架构

### 核心技术栈
- **后端**: Python + Flask
- **前端**: React 18 + React Router 6 + Redux Toolkit
- **AI核心**: LangChain + CLIP + Milvus + 多模态模型
- **多模态模型**: OmniVision（默认）+ LLaVA（可选）
- **部署**: Docker + Docker Compose

### 服务组件
- **Frontend**: React前端应用
- **Backend**: Flask API服务
- **Milvus**: 向量数据库
- **Nginx**: 反向代理
- **MinIO**: 对象存储

## 🚀 快速开始

### 环境要求
- Python 3.9+
- Node.js 16+
- Docker 20.10+
- Docker Compose 2.0+
- Milvus 2.0+

### 一键部署

1. **克隆项目**
```bash
git clone <repository-url>
cd ai-antique-appraiser
```

2. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，设置如下参数：
# - OPENAI_API_KEY: OpenAI API密钥（可选）
# - MULTIMODAL_MODEL_TYPE: 多模态模型类型（默认: omnivision）
# - OMNIVISION_MODEL_PATH: OmniVision本地模型路径（可选）
# - LLAVA_MODEL_PATH: LLaVA本地模型路径（可选）
# - SECRET_KEY: Flask应用密钥
```

3. **启动服务**
```bash
# Linux/macOS
chmod +x deploy.sh
./deploy.sh

# Windows
deploy.bat
```

4. **访问应用**
- 前端应用: http://localhost
- 后端API: http://localhost/api
- MinIO控制台: http://localhost:9001 (minioadmin/minioadmin)

## 📁 项目结构

```
ai-antique-appraiser/
├── ai_core/                 # AI核心模块
│   ├── clip_encoder/       # CLIP图像编码器
│   ├── vector_db/          # Milvus向量数据库
│   ├── langchain_agent/    # LangChain智能代理
│   ├── llava_model/        # LLaVA多模态模型
│   ├── omnivision/         # OmniVision轻量级多模态模型
│   └── ai_core.py          # AI核心控制器
├── backend/                # Flask后端
│   ├── app.py             # 主应用文件
│   ├── requirements.txt   # Python依赖
│   └── Dockerfile         # 后端容器配置
├── frontend/              # React前端
│   ├── src/              # 源代码
│   ├── package.json      # Node.js依赖
│   ├── Dockerfile        # 前端容器配置
│   └── nginx.conf        # 前端Nginx配置
├── nginx/                # Nginx配置
├── uploads/              # 上传图片存储
├── data/                 # 分析结果存储
├── .env.example          # 环境变量配置示例
├── docker-compose.yml    # Docker编排配置
├── deploy.sh            # Linux/macOS部署脚本
├── deploy.bat           # Windows部署脚本
└── DEPLOYMENT.md        # 详细部署文档
```

## 🔧 核心功能

### 1. 多模态模型选择
- **OmniVision模型**（默认）: 轻量级（968M参数），适合所有设备，启动快速
- **LLaVA模型**（可选）: 功能强大，适合高性能设备
- **动态切换**: 支持运行时切换模型类型
- **智能回退**: 模型加载失败时自动启用模拟模式

### 2. 图像分析
- 上传古董图片进行智能分析
- 识别年代、材质、真伪等信息
- 生成详细的分析报告

### 2. 图像分析
- 上传古董图片进行智能分析
- 识别年代、材质、真伪等信息
- 生成详细的分析报告

### 3. 文本搜索
- 通过文本描述搜索相似古董
- 基于向量相似度的智能匹配
- 返回相关度排序的结果

### 3. AI对话
- 与AI专家进行古董相关对话
- 获取专业鉴定建议
- 学习古董知识

### 4. 数据管理
- 本地图片存储和管理
- 分析结果本地保存
- 支持数据清理和备份

## 📚 API接口

### 核心接口
- `POST /api/analyze` - 图像分析（使用当前激活的多模态模型）
- `POST /api/search` - 文本搜索
- `POST /api/chat` - AI对话（LangChain代理）
- `POST /api/chat/multimodal` - 多模态AI对话（支持图像+文本）
- `POST /api/upload` - 上传古董
- `GET /api/status` - 系统状态
- `GET /health` - 健康检查

### 模型管理接口
- `POST /api/model/switch` - 切换多模态模型类型
- `GET /api/model/status` - 获取当前模型状态

### 模型切换示例
```bash
# 切换到LLaVA模型（高性能）
curl -X POST http://localhost:5000/api/model/switch \
  -H "Content-Type: application/json" \
  -d '{"model_type": "llava"}'

# 切换回默认的OmniVision模型（轻量级）
curl -X POST http://localhost:5000/api/model/switch \
  -H "Content-Type: application/json" \
  -d '{"model_type": "omnivision"}'

# 获取当前模型状态
curl http://localhost:5000/api/model/status
```

## 🛠️ 开发指南

### 本地开发

1. **后端开发**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

2. **前端开发**
```bash
cd frontend
npm install
npm start
```

3. **AI核心测试**
```bash
cd ai_core
python example.py
```

### 环境配置
- 设置 `OPENAI_API_KEY` 环境变量
- 确保 Milvus 服务运行在 `localhost:19530`
- 配置必要的文件权限

## 📊 性能优化

### 系统优化
- 使用SSD存储提高I/O性能
- 增加内存到16GB以上
- 优化Docker资源配置

### 应用优化
- 图片压缩和格式优化
- 批量处理支持
- 本地缓存策略

## 🔒 安全配置

### 网络安全
- 防火墙端口控制
- 访问权限管理

### 应用安全
- 环境变量密钥管理
- 文件类型验证
- 输入数据验证

## 📈 监控和维护

### 服务监控
- Docker容器状态监控
- 应用日志分析
- 系统资源监控

### 数据备份
- 定期备份上传文件
- 备份分析结果
- 配置文件备份

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交代码变更
4. 发起 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📞 技术支持

如遇到问题，请：
1. 查看 [DEPLOYMENT.md](DEPLOYMENT.md) 文档
2. 检查服务日志
3. 确认环境配置
4. 联系技术支持团队

---

**AI鉴宝师项目** - 让古董鉴定更智能、更便捷！
