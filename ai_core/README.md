# AI 核心模块（支持LLaVA本地多模态大模型）

## 概述

AI 核心模块是 AI 鉴宝师项目的核心智能组件，集成了多模态大模型技术，提供强大的古董分析和识别能力。该模块现已支持本地部署的LLaVA多模态大模型，基于 LLaVA、LangChain、CLIP 和 Milvus 构建，实现了图像识别、文本分析、向量检索和智能对话等功能。

## 技术架构

### 核心组件

1. **LLaVA 多模态大模型** (`llava_model/`)
   - 支持本地部署的LLaVA模型
   - 多模态图像和文本理解
   - 专业的古董鉴定分析能力
   - 支持API和本地模型两种方式

2. **CLIP 编码器** (`clip_encoder/`)
   - 基于 OpenAI CLIP 模型
   - 支持图像和文本的多模态编码
   - 提供特征提取和相似度计算功能

3. **向量数据库** (`vector_db/`)
   - 基于 Milvus 高性能向量数据库
   - 支持大规模向量存储和检索
   - 提供相似古董搜索功能

4. **LangChain 代理** (`langchain_agent/`)
   - 基于 LangChain 框架的智能代理
   - 集成多种专业工具
   - 提供自然语言对话和分析能力

5. **AI 核心控制器** (`ai_core.py`)
   - 整合所有组件的统一接口
   - 提供完整的古董分析流程
   - 管理各组件间的协作

## 模块结构

```
ai_core/
├── __init__.py              # 模块初始化
├── ai_core.py              # 主控制器
├── clip_encoder/           # CLIP 编码器模块
│   ├── __init__.py
│   └── clip_encoder.py
├── vector_db/              # 向量数据库模块
│   ├── __init__.py
│   └── milvus_client.py
├── langchain_agent/        # LangChain 代理模块
│   ├── __init__.py
│   └── antique_agent.py
├── llava_model/            # LLaVA 多模态大模型模块
│   ├── __init__.py
│   └── llava_client.py
├── config.py               # 配置管理
├── example.py              # 传统使用示例
├── example_llava.py        # LLaVA使用示例
├── LLaVA_部署指南.md   # LLaVA部署文档
└── README.md               # 本文档
```

## 主要功能

### 1. 古董图像分析（支持LLaVA）
- 使用LLaVA多模态大模型进行图像理解
- 图像特征提取和编码
- 古董类型识别和分类
- 年代和材质分析
- 真伪评估和保存状况分析
- 结构化鉴宝报告生成

### 2. 文本搜索和查询
- 自然语言查询处理
- 古董知识库检索
- 历史背景和文化信息查询
- 相似古董推荐

### 3. 向量数据库管理
- 古董特征向量存储
- 高效相似度搜索
- 批量数据处理
- 数据库维护和优化

### 4. 智能对话系统（多选项）
- LLaVA多模态对话（主要）
- LangChain代理对话（可选）
- 专业古董知识问答
- 分析结果解释
- 对话历史管理

## 使用方法

### 基本使用（支持LLaVA）

```python
from ai_core import AICore

# 初始化 AI 核心（支持LLaVA本地模型）
ai_core = AICore(
    openai_api_key="your-openai-api-key",  # 可选，用于LangChain代理
    milvus_host="localhost",
    milvus_port="19530",
    llava_model_path="/path/to/llava/model",  # 本地模型路径
    # llava_api_url="http://localhost:8000"  # 或使用API服务
)

# 使用LLaVA分析古董图像
result = ai_core.analyze_antique_with_llava(
    image="path/to/antique.jpg",
    user_query="请详细分析这个古董的年代、材质和价值"
)

# 与LLaVA模型对话
response = ai_core.chat_with_llava(
    image="path/to/antique.jpg",
    message="这个古董是什么材质的？"
)

# 文本搜索古董
search_result = ai_core.search_antiques_by_text(
    query_text="明清时期的陶瓷花瓶",
    top_k=10
)

# 与 LangChain 代理对话（可选）
response = ai_core.chat_with_agent("请介绍一下明代青花瓷的特点")
```

### 高级功能

```python
# 添加古董到数据库
success = ai_core.add_antique_to_database(
    antique_id=1,
    image=antique_image,
    text_description="明代青花瓷花瓶，高30厘米",
    metadata={
        "name": "明代青花瓷花瓶",
        "dynasty": "明代",
        "material": "陶瓷",
        "estimated_value": "50000-80000元"
    }
)

# 批量添加古董
batch_result = ai_core.batch_add_antiques(antiques_data)

# 获取系统状态
status = ai_core.get_system_status()

# 获取对话历史
history = ai_core.get_conversation_history()
```

## 配置要求

### 环境依赖
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.35.0+ (用于LLaVA)
- CLIP 0.2.0
- LangChain 0.0.340+
- Milvus 2.2.14+
- OpenAI API 密钥（可选，用于LangChain代理）

### 服务要求
- Milvus 向量数据库服务
- LLaVA 模型（本地或API服务）
- GPU 支持（推荐8GB+显存）

## 性能指标

### 处理能力
- 图像分析：~2-5秒/张
- 文本搜索：~1-3秒/次
- 向量检索：~100ms/次
- 对话响应：~2-8秒/次

### 存储容量
- 向量数据库：支持百万级向量存储
- 特征维度：512维（CLIP 标准）
- 索引类型：IVF_FLAT（高精度）

### 并发处理
- 支持多用户并发访问
- 异步处理能力
- 资源池化管理

## 扩展功能

### 自定义工具
可以通过继承 `AntiqueAgent` 类来添加自定义分析工具：

```python
class CustomAntiqueAgent(AntiqueAgent):
    def _custom_analysis_tool(self, query: str) -> str:
        # 实现自定义分析逻辑
        return "自定义分析结果"
```

### 模型替换
支持替换不同的 CLIP 模型和 LLM 模型：

```python
ai_core = AICore(
    clip_model_name="ViT-L/14",  # 使用更大的 CLIP 模型
    llm_model_name="gpt-4"       # 使用 GPT-4
)
```

### 数据库扩展
支持连接不同的 Milvus 实例和集合：

```python
ai_core = AICore(
    milvus_host="your-milvus-host",
    milvus_port="19530",
    collection_name="custom_antique_vectors"
)
```

## 故障排除

### 常见问题

1. **CLIP 模型加载失败**
   - 检查网络连接
   - 确保 PyTorch 版本兼容
   - 尝试使用较小的模型

2. **Milvus 连接失败**
   - 检查 Milvus 服务状态
   - 验证连接参数
   - 确认防火墙设置

3. **OpenAI API 错误**
   - 检查 API 密钥有效性
   - 确认账户余额
   - 验证 API 访问权限

### 调试模式
启用详细日志输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 开发指南

### 代码规范
- 遵循 PEP 8 代码风格
- 使用类型注解
- 添加详细的文档字符串
- 编写单元测试

### 测试
运行示例脚本：

```bash
cd ai_core
python example.py
```

### 贡献
欢迎提交 Issue 和 Pull Request 来改进模块功能。

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目 Issues
- 邮箱：[your-email@example.com]
- 文档：[项目文档链接]
