"""
AI 核心模块配置文件
管理各种配置参数和设置
"""

import os
from typing import Dict, Any


class AICoreConfig:
    """AI 核心配置类"""
    
    # CLIP 模型配置
    CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B/32")
    CLIP_DEVICE = os.getenv("CLIP_DEVICE", "auto")  # auto, cpu, cuda
    
    # Milvus 配置
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "antique_vectors")
    MILVUS_DIM = int(os.getenv("MILVUS_DIM", "512"))
    
    # LangChain 配置
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    

    
    # 搜索配置
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
    DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", "0.5"))
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 缓存配置
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 秒
    
    # 性能配置
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    
    # 安全配置
    API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    @classmethod
    def get_clip_config(cls) -> Dict[str, Any]:
        """获取 CLIP 配置"""
        return {
            "model_name": cls.CLIP_MODEL_NAME,
            "device": cls.CLIP_DEVICE
        }
    
    @classmethod
    def get_milvus_config(cls) -> Dict[str, Any]:
        """获取 Milvus 配置"""
        return {
            "host": cls.MILVUS_HOST,
            "port": cls.MILVUS_PORT,
            "collection_name": cls.MILVUS_COLLECTION_NAME,
            "dim": cls.MILVUS_DIM
        }
    

    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """获取 LLM 配置"""
        return {
            "model_name": cls.LLM_MODEL_NAME,
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.LLM_MAX_TOKENS
        }
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """获取搜索配置"""
        return {
            "default_top_k": cls.DEFAULT_TOP_K,
            "default_score_threshold": cls.DEFAULT_SCORE_THRESHOLD
        }
    
    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """获取性能配置"""
        return {
            "batch_size": cls.BATCH_SIZE,
            "max_concurrent_requests": cls.MAX_CONCURRENT_REQUESTS,
            "enable_cache": cls.ENABLE_CACHE,
            "cache_ttl": cls.CACHE_TTL
        }
    
    @classmethod
    def get_security_config(cls) -> Dict[str, Any]:
        """获取安全配置"""
        return {
            "api_key_header": cls.API_KEY_HEADER,
            "rate_limit_per_minute": cls.RATE_LIMIT_PER_MINUTE
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """验证配置有效性"""
        try:
            # 验证数值配置
            assert cls.MILVUS_DIM > 0, "MILVUS_DIM 必须大于 0"
            assert 0 <= cls.LLM_TEMPERATURE <= 2, "LLM_TEMPERATURE 必须在 0-2 之间"
            assert cls.LLM_MAX_TOKENS > 0, "LLM_MAX_TOKENS 必须大于 0"

            assert cls.DEFAULT_TOP_K > 0, "DEFAULT_TOP_K 必须大于 0"
            assert 0 <= cls.DEFAULT_SCORE_THRESHOLD <= 1, "DEFAULT_SCORE_THRESHOLD 必须在 0-1 之间"
            
            # 验证字符串配置
            assert cls.CLIP_MODEL_NAME, "CLIP_MODEL_NAME 不能为空"
            assert cls.MILVUS_HOST, "MILVUS_HOST 不能为空"
            assert cls.LLM_MODEL_NAME, "LLM_MODEL_NAME 不能为空"
            
            return True
            
        except AssertionError as e:
            print(f"配置验证失败: {e}")
            return False
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("AI 核心模块配置:")
        print(f"  CLIP 模型: {cls.CLIP_MODEL_NAME}")
        print(f"  CLIP 设备: {cls.CLIP_DEVICE}")
        print(f"  Milvus 地址: {cls.MILVUS_HOST}:{cls.MILVUS_PORT}")
        print(f"  Milvus 集合: {cls.MILVUS_COLLECTION_NAME}")
        print(f"  向量维度: {cls.MILVUS_DIM}")
        print(f"  LLM 模型: {cls.LLM_MODEL_NAME}")
        print(f"  LLM 温度: {cls.LLM_TEMPERATURE}")
        print(f"  LLM 最大令牌: {cls.LLM_MAX_TOKENS}")

        print(f"  默认搜索结果数: {cls.DEFAULT_TOP_K}")
        print(f"  默认相似度阈值: {cls.DEFAULT_SCORE_THRESHOLD}")
        print(f"  日志级别: {cls.LOG_LEVEL}")
        print(f"  启用缓存: {cls.ENABLE_CACHE}")
        print(f"  批处理大小: {cls.BATCH_SIZE}")
        print(f"  最大并发请求: {cls.MAX_CONCURRENT_REQUESTS}")


# 预定义配置模板
class ConfigTemplates:
    """配置模板"""
    
    @staticmethod
    def development() -> Dict[str, str]:
        """开发环境配置"""
        return {
            "CLIP_MODEL_NAME": "ViT-B/32",
            "CLIP_DEVICE": "cpu",
            "MILVUS_HOST": "localhost",
            "MILVUS_PORT": "19530",
            "LLM_MODEL_NAME": "gpt-3.5-turbo",
            "LLM_TEMPERATURE": "0.7",
            "LOG_LEVEL": "DEBUG",
            "ENABLE_CACHE": "false"
        }
    
    @staticmethod
    def production() -> Dict[str, str]:
        """生产环境配置"""
        return {
            "CLIP_MODEL_NAME": "ViT-L/14",
            "CLIP_DEVICE": "cuda",
            "MILVUS_HOST": "milvus-cluster",
            "MILVUS_PORT": "19530",
            "LLM_MODEL_NAME": "gpt-4",
            "LLM_TEMPERATURE": "0.5",
            "LOG_LEVEL": "INFO",
            "ENABLE_CACHE": "true",
            "BATCH_SIZE": "64",
            "MAX_CONCURRENT_REQUESTS": "50"
        }
    
    @staticmethod
    def testing() -> Dict[str, str]:
        """测试环境配置"""
        return {
            "CLIP_MODEL_NAME": "ViT-B/32",
            "CLIP_DEVICE": "cpu",
            "MILVUS_HOST": "localhost",
            "MILVUS_PORT": "19530",
            "LLM_MODEL_NAME": "gpt-3.5-turbo",
            "LLM_TEMPERATURE": "0.1",
            "LOG_LEVEL": "WARNING",
            "ENABLE_CACHE": "false",
            "BATCH_SIZE": "8"
        }


def load_config_from_env():
    """从环境变量加载配置"""
    # 配置已经通过类属性从环境变量加载
    pass


def set_config_from_dict(config_dict: Dict[str, str]):
    """从字典设置配置"""
    for key, value in config_dict.items():
        if hasattr(AICoreConfig, key):
            # 根据属性类型转换值
            attr_value = getattr(AICoreConfig, key)
            if isinstance(attr_value, bool):
                setattr(AICoreConfig, key, value.lower() == "true")
            elif isinstance(attr_value, int):
                setattr(AICoreConfig, key, int(value))
            elif isinstance(attr_value, float):
                setattr(AICoreConfig, key, float(value))
            else:
                setattr(AICoreConfig, key, value)


# 初始化时加载配置
load_config_from_env()
