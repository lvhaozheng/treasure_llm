"""
AI 核心模块
多模态大模型古董分析系统
"""

# 使用相对导入避免路径问题
from .clip_encoder import CLIPEncoder
from .vector_db import MilvusClient
from .langchain_agent import AntiqueAgent
from .models.client import InternVL3_5Client, Qwen3Client

from .ai_core import AICore

# 导出列表
__all__ = [
    'CLIPEncoder', 'MilvusClient', 'AntiqueAgent', 
    'InternVL3_5Client', 'Qwen3Client', 'AICore'
]
