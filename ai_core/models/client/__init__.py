"""模型客户端模块

支持多种模型的客户端实现，包括 InternVL3_5、SmolVLM2 和 Qwen3
"""

from .smolvlm2_client import SmolVLM2Client
from .internvl3_5_client import InternVL3_5Client
from .qwen3_client import Qwen3Client

__all__ = ['SmolVLM2Client', 'InternVL3_5Client', 'Qwen3Client']