"""AI 核心主控制器
整合 CLIP 编码器、向量数据库和 LangChain 代理
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
from PIL import Image
from functools import wraps

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logger_config import get_ai_logger

# 导入报告格式化器
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'services'))
try:
    from report_formatter import format_appraisal_report
    REPORT_FORMATTER_AVAILABLE = True
    logger = get_ai_logger('ai_core')
    logger.info("报告格式化器导入成功")
except ImportError as e:
    REPORT_FORMATTER_AVAILABLE = False
    logger = get_ai_logger('ai_core')
    logger.warning(f"报告格式化器导入失败: {e}")

from clip_encoder import CLIPEncoder
from vector_db import MilvusClient
from langchain_agent import AntiqueAgent
from models.client import InternVL3_5Client, Qwen3Client
from optimized_query_processor import OptimizedQueryProcessor
# 配置AI核心模块日志
logger = get_ai_logger('ai_core')


def log_method_call(method_name: str = None):
    """日志装饰器，记录方法调用的详细信息"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 获取方法名
            func_name = method_name or func.__name__
            
            # 记录调用开始
            start_time = time.time()
            call_id = f"{func_name}_{int(start_time * 1000) % 100000}"
            
            # 记录输入参数（安全处理，避免记录敏感信息）
            safe_args = []
            for i, arg in enumerate(args):
                if isinstance(arg, (str, int, float, bool)):
                    safe_args.append(str(arg)[:100])  # 限制长度
                elif isinstance(arg, (list, dict)):
                    safe_args.append(f"{type(arg).__name__}(len={len(arg)})")
                else:
                    safe_args.append(f"{type(arg).__name__}")
            
            safe_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_kwargs[k] = str(v)[:100]  # 限制长度
                elif isinstance(v, (list, dict)):
                    safe_kwargs[k] = f"{type(v).__name__}(len={len(v)})"
                else:
                    safe_kwargs[k] = f"{type(v).__name__}"
            
            logger.info(f"🚀 [{call_id}] 开始调用 {func_name}")
            logger.info(f"📥 [{call_id}] 输入参数: args={safe_args}, kwargs={safe_kwargs}")
            
            try:
                # 执行方法
                result = func(self, *args, **kwargs)
                
                # 记录执行时间
                end_time = time.time()
                execution_time = end_time - start_time
                
                # 记录返回结果（安全处理）
                if isinstance(result, dict):
                    result_info = f"dict(keys={list(result.keys())[:5]})"
                elif isinstance(result, (list, tuple)):
                    result_info = f"{type(result).__name__}(len={len(result)})"
                elif hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                    result_info = "generator/iterator"
                else:
                    result_info = f"{type(result).__name__}"
                
                logger.info(f"✅ [{call_id}] 调用成功完成 {func_name}")
                logger.info(f"📤 [{call_id}] 返回结果: {result_info}")
                logger.info(f"⏱️ [{call_id}] 执行时间: {execution_time:.3f}秒")
                
                return result
                
            except Exception as e:
                # 记录异常
                end_time = time.time()
                execution_time = end_time - start_time
                
                logger.error(f"❌ [{call_id}] 调用失败 {func_name}")
                logger.error(f"💥 [{call_id}] 异常信息: {str(e)}")
                logger.error(f"⏱️ [{call_id}] 执行时间: {execution_time:.3f}秒")
                logger.error(f"📍 [{call_id}] 异常堆栈: {traceback.format_exc()}")
                
                raise
        
        return wrapper
    return decorator


class AICore:
    """AI 核心主控制器"""
    
    def __init__(self, 
                 openai_api_key: str = "",
                 milvus_host: str = "localhost",
                 milvus_port: str = "19530",
                 clip_model_name: str = "ViT-B/32",
                 llm_model_name: str = "gpt-3.5-turbo",
                 internvl3_5_model_path: str = "",
                 max_tokens: int = 512,
                 temperature: float = 0.7):
        """
        初始化 AI 核心
        
        Args:
            openai_api_key: OpenAI API 密钥（可选，用于LangChain代理）
            milvus_host: Milvus 服务器地址
            milvus_port: Milvus 服务器端口
            clip_model_name: CLIP 模型名称
            llm_model_name: LLM 模型名称
            internvl3_5_model_path: InternVL3_5-1B 本地模型路径
            max_tokens: 最大生成token数
            temperature: 生成温度
        """
        self.openai_api_key = openai_api_key
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.clip_model_name = clip_model_name
        self.llm_model_name = llm_model_name
        self.internvl3_5_model_path = internvl3_5_model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # 初始化组件
        self.clip_encoder = None
        self.vector_db = None
        self.agent = None
        self.internvl3_5_client = None
        self.optimized_query_processor = None
        
        # 初始化缓存系统
        self._init_cache()
        
        try:
            self._init_components()
            logger.info("AI 核心初始化成功")
        except Exception as e:
            logger.error(f"AI 核心初始化失败: {e}")
            raise
    
    def _init_components(self):
        """初始化各个组件"""
        # 初始化 CLIP 编码器
        self.clip_encoder = CLIPEncoder(
            model_name=self.clip_model_name
        )
        
        # 初始化向量数据库
        self.vector_db = MilvusClient(
            host=self.milvus_host,
            port=self.milvus_port,
            collection_name="antique_vectors",
            dim=512  # CLIP 特征维度
        )
        
        # 初始化 InternVL3_5 客户端
        self._init_internvl3_5_client()
        
        # 初始化 LangChain 代理（支持本地模型）
        self.agent = AntiqueAgent(
            openai_api_key=self.openai_api_key,
            model_name=self.llm_model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            use_local_models=True,  # 启用本地模型支持
            clip_encoder=self.clip_encoder,  # 传入CLIP编码器
            vector_db=self.vector_db,  # 传入向量数据库
            internvl3_5_client=self.internvl3_5_client  # 传入已创建的InternVL3.5客户端
        )
        
        # 初始化优化查询处理器（传入已创建的组件，避免重复初始化）
        self.optimized_query_processor = OptimizedQueryProcessor(
            similarity_threshold=0.6,  # 相似度阈值
            top_k=5,
            clip_encoder=self.clip_encoder,  # 复用已创建的CLIP编码器
            milvus_client=self.vector_db,  # 复用已创建的Milvus客户端
            internvl3_5_client=self.internvl3_5_client  # 复用已创建的InternVL3.5客户端
        )
        
        if not self.openai_api_key:
            logger.info("未提供OpenAI API密钥，将使用本地模型（Qwen3 + SmolVLM2）")
        else:
            logger.info("已配置OpenAI API密钥，支持LangChain代理和本地模型混合使用")
    
    def _init_internvl3_5_client(self):
        """初始化 InternVL3_5 客户端"""
        try:
            # 创建 InternVL3_5 客户端
            self.internvl3_5_client = InternVL3_5Client(
                model_path=self.internvl3_5_model_path if self.internvl3_5_model_path else None,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # 记录模型信息
            model_info = self.internvl3_5_client.get_model_info()
            logger.info(f"🎉 InternVL3_5 模型加载信息:")
            for key, value in model_info.items():
                logger.info(f"  {key}: {value}")
                
            logger.info("✅ InternVL3_5 客户端初始化成功")
            
        except Exception as e:
            error_msg = f"InternVL3_5 客户端初始化失败: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    

    
    @log_method_call("analyze_antique_image_stream")
    def analyze_antique_image_stream(self, image: Union[str, Image.Image, np.ndarray], 
                                    description: str = ""):
        """
        流式分析古董图像，优先使用优化查询处理器进行智能判断
        
        Args:
            image: 图像输入（文件路径、PIL图像或numpy数组）
            description: 可选的文本描述
            
        Yields:
            分析结果文本片段
        """
        try:
            # 构建分析提示
            if description:
                prompt = f"请根据以下要求分析这张古董文物图片：{description}。请提供详细的专业鉴定分析，包括文物类型、年代、材质、工艺特点、真伪评估、保存状况和历史价值。请直接提供分析结果，不要重复用户的要求。"
            else:
                prompt = "请详细分析这张古董文物图片，包括文物类型、年代、材质、工艺特点、真伪评估、保存状况和历史价值。"
            
            # 首先尝试使用优化查询处理器的流式方法
            if hasattr(self, 'optimized_query_processor') and self.optimized_query_processor:
                logger.info("🔍 使用优化查询处理器进行流式智能分析")
                try:
                    # 使用优化查询处理器的流式方法
                    for text_chunk in self.optimized_query_processor.process_image_text_query_stream(
                        image=image,
                        text_query=prompt
                    ):
                        yield text_chunk
                    return
                except Exception as e:
                    logger.warning(f"优化查询处理器流式分析失败，回退到大模型: {e}")
            
            # 回退到InternVL3_5进行流式分析
            if self.internvl3_5_client and hasattr(self.internvl3_5_client, 'chat_about_antique_stream'):
                logger.info("使用InternVL3_5进行流式古董图像分析")
                # 使用真正的流式方法
                for text_chunk in self.internvl3_5_client.chat_about_antique_stream(prompt, image):
                    yield text_chunk
            elif self.internvl3_5_client and hasattr(self.internvl3_5_client, 'chat_about_antique'):
                logger.info("使用InternVL3_5进行模拟流式古董图像分析")
                # 备用方案：使用chat_about_antique方法获取完整回复，然后进行流式输出
                response = self.internvl3_5_client.chat_about_antique(prompt, image)
                # 按句子分割进行流式输出
                sentences = response.split('。')
                for sentence in sentences:
                    if sentence.strip():
                        yield sentence.strip() + '。'
            else:
                yield "InternVL3_5客户端不可用，无法进行流式图像分析。"
                
        except Exception as e:
            logger.error(f"流式古董图像分析失败: {e}")
            yield f"分析过程中出现错误: {str(e)}"
    
    @log_method_call("analyze_antique_image")
    def analyze_antique_image(self, image: Union[str, Image.Image, np.ndarray], 
                            description: str = "") -> Dict[str, Any]:
        """
        分析古董图像
        
        Args:
            image: 图像输入
            description: 文本描述
            
        Returns:
            分析结果
        """
        try:
            if not self.internvl3_5_client:
                raise RuntimeError("InternVL3_5客户端未初始化")
            
            # 1. 使用 InternVL3_5 生成鉴定报告
            internvl3_5_report = self.internvl3_5_client.generate_appraisal_report(
                image=image,
                user_query=description
            )
            
            # 2. 使用混合代理进行分析（优先使用本地模型）
            agent_analysis = None
            if self.agent:
                try:
                    agent_result = self.agent.analyze_antique(
                        image=image,
                        text_description=description
                    )
                    agent_analysis = agent_result
                except Exception as e:
                    logger.warning(f"代理分析失败，仅使用InternVL3_5: {e}")
            
            # 3. 使用 CLIP 编码图像用于相似性搜索
            image_features = self.clip_encoder.encode_image(image)
            
            # 4. 搜索相似古董
            similar_antiques = self.vector_db.search_similar_images(
                query_vector=image_features.tolist(),
                top_k=5,
                score_threshold=0.6
            )
            
            # 5. 整合结果并格式化为标准JSON结构
            if REPORT_FORMATTER_AVAILABLE and internvl3_5_report.get("analysis_text"):
                # 使用新的格式化器生成结构化报告
                structured_report = format_appraisal_report(
                    ai_analysis=internvl3_5_report.get("analysis_text", ""),
                    user_query=description,
                    image_path=str(image) if isinstance(image, str) else None,
                    model_info=internvl3_5_report.get("model_info", {})
                )
                
                # 添加额外的分析数据
                structured_report["additional_analysis"] = {
                    "agent_analysis": agent_analysis,
                    "image_features": image_features.tolist(),
                    "similar_antiques": similar_antiques,
                    "internvl3_5_raw": internvl3_5_report
                }
                
                result = structured_report
            else:
                # 回退到原始格式
                result = {
                    "internvl3_5_analysis": internvl3_5_report,
                    "agent_analysis": agent_analysis,
                    "image_features": image_features.tolist(),
                    "similar_antiques": similar_antiques,
                    "user_query": description,
                "analysis_timestamp": self._get_timestamp(),
                "model_info": {
                    "internvl3_5_model": self.internvl3_5_client.get_model_info(),
                    "clip_model": self.clip_encoder.get_model_info(),
                    "agent_info": self.agent.get_agent_info() if self.agent else None
                }
            }
            
            logger.info("基于InternVL3_5的古董图像分析完成")
            return result
            
        except Exception as e:
            logger.error(f"基于InternVL3_5的古董图像分析失败: {e}")
            # 回退到LangChain方法
            return self._analyze_with_langchain(image, description)
    
    @log_method_call("analyze_antique_image_optimized")
    def analyze_antique_image_optimized(self, image: Union[str, Image.Image, np.ndarray], 
                                      description: str = "") -> Dict[str, Any]:
        """
        使用优化查询处理器分析古董图像
        
        Args:
            image: 图像输入
            description: 文本描述
            
        Returns:
            优化的分析结果
        """
        try:
            if not self.optimized_query_processor:
                logger.warning("优化查询处理器未初始化，回退到标准分析")
                return self.analyze_antique_image(image, description)
            
            # 使用优化查询处理器进行分析
            result = self.optimized_query_processor.process_image_text_query(
                image=image,
                text_query=description
            )
            
            logger.info("优化古董图像分析完成")
            return result
            
        except Exception as e:
            logger.error(f"优化古董图像分析失败: {e}")
            # 回退到标准分析方法
            return self.analyze_antique_image(image, description)
    
    def _analyze_with_langchain(self, image: Union[str, Image.Image, np.ndarray], 
                               description: str = "") -> Dict[str, Any]:
        """
        使用LangChain代理分析古董（回退方法）
        """
        try:
            if not self.agent:
                logger.warning("未LangChain代理未初始化，仅返回基本信息")
                # 仅使用CLIP进行特征提取
                image_features = self.clip_encoder.encode_image(image)
                similar_antiques = self.vector_db.search_similar_images(
                    query_vector=image_features.tolist(),
                    top_k=5,
                    score_threshold=0.6
                )
                
                return {
                    "image_features": image_features.tolist(),
                    "similar_antiques": similar_antiques,
                    "description": description,
                    "analysis_timestamp": self._get_timestamp(),
                    "note": "仅进行了图像特征提取和相似性搜索"
                }
            
            # 1. 使用 CLIP 编码图像
            image_features = self.clip_encoder.encode_image(image)
            
            # 2. 使用 LangChain 代理分析
            agent_result = self.agent.analyze_antique(
                image_features=image_features,
                text_description=description
            )
            
            # 3. 搜索相似古董
            similar_antiques = self.vector_db.search_similar_images(
                query_vector=image_features.tolist(),
                top_k=5,
                score_threshold=0.6
            )
            
            # 4. 整合结果
            result = {
                "image_features": image_features.tolist(),
                "agent_analysis": agent_result,
                "similar_antiques": similar_antiques,
                "description": description,
                "analysis_timestamp": self._get_timestamp()
            }
            
            logger.info("古董图像分析完成")
            return result
            
        except Exception as e:
            logger.error(f"古董图像分析失败: {e}")
            raise
    
    @log_method_call("search_antiques_by_text")
    def search_antiques_by_text(self, query_text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        通过文本搜索古董
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果
        """
        try:
            # 1. 使用 CLIP 编码查询文本
            text_features = self.clip_encoder.encode_text(query_text)
            
            # 2. 在向量数据库中搜索
            similar_antiques = self.vector_db.search_similar_texts(
                query_vector=text_features.tolist(),
                top_k=top_k,
                score_threshold=0.5
            )
            
            # 3. 使用代理进行文本分析
            agent_analysis = self.agent.chat(f"请分析这个查询：{query_text}")
            
            result = {
                "query_text": query_text,
                "text_features": text_features.tolist(),
                "similar_antiques": similar_antiques,
                "agent_analysis": agent_analysis,
                "search_timestamp": self._get_timestamp()
            }
            
            logger.info("文本搜索古董完成")
            return result
            
        except Exception as e:
            logger.error(f"文本搜索古董失败: {e}")
            raise
    
    @log_method_call("analyze_antique_text_stream")
    def analyze_antique_text_stream(self, description: str):
        """
        流式分析古董文本描述，使用Qwen3模型
        
        Args:
            description: 古董文本描述
            
        Yields:
            分析结果文本片段
        """
        try:
            # 构建分析提示
            prompt = f"""你是一位专业的古董鉴定专家，请根据以下要求进行古董分析：{description}
            
请提供详细的专业鉴定分析，包括：
1. 文物类型和名称
2. 年代判断
3. 材质分析
4. 工艺特点
5. 真伪评估
6. 保存状况
7. 历史价值

请直接提供分析结果，不要重复用户的要求。请确保回答完全使用中文，不要使用英文。"""
            
            # 优先使用LangChain代理（包含Qwen3）
            if self.agent and hasattr(self.agent, 'qwen3_client') and self.agent.qwen3_client:
                logger.info("使用Qwen3进行流式古董文本分析")
                
                messages = [{"role": "user", "content": prompt}]
                
                # 检查是否支持流式输出
                if hasattr(self.agent.qwen3_client, 'chat_stream'):
                    for text_chunk in self.agent.qwen3_client.chat_stream(messages, max_length=512):
                        yield text_chunk
                else:
                    # 备用方案：使用普通chat方法然后模拟流式输出
                    response = self.agent.qwen3_client.chat(messages, max_length=512)
                    sentences = response.split('。')
                    for sentence in sentences:
                        if sentence.strip():
                            yield sentence.strip() + '。'
            elif self.agent and hasattr(self.agent, '_chat_with_local_models'):
                logger.info("使用代理本地模型进行流式古董文本分析")
                # 使用代理的本地模型方法获取完整回复，然后进行流式输出
                response = self.agent._chat_with_local_models(prompt)
                # 按句子分割进行流式输出
                sentences = response.split('。')
                for sentence in sentences:
                    if sentence.strip():
                        yield sentence.strip() + '。'
            elif self.internvl3_5_client and hasattr(self.internvl3_5_client, 'text_chat'):
                logger.info("使用InternVL3_5进行流式古董文本分析")
                # 备用方案：使用InternVL3_5
                response = self.internvl3_5_client.text_chat(prompt)
                # 按句子分割进行流式输出
                sentences = response.split('。')
                for sentence in sentences:
                    if sentence.strip():
                        yield sentence.strip() + '。'
            else:
                yield "文本分析模型不可用，无法进行流式文本分析。"
                
        except Exception as e:
            logger.error(f"流式古董文本分析失败: {e}")
            yield f"分析过程中出现错误: {str(e)}"
    
    @log_method_call("analyze_antique_text")
    def analyze_antique_text(self, description: str) -> Dict[str, Any]:
        """
        分析古董文本描述
        
        Args:
            description: 古董文本描述
            
        Returns:
            分析结果
        """
        try:
            # 检查缓存
            cache_key = self._get_cache_key(description)
            if cache_key in self.text_analysis_cache:
                logger.info(f"从缓存中获取文本分析结果: {cache_key[:8]}...")
                cached_result = self.text_analysis_cache[cache_key].copy()
                cached_result["from_cache"] = True
                cached_result["analysis_timestamp"] = self._get_timestamp()
                return cached_result
            
            # 1. 直接使用代理进行文本分析（不进行向量检索）
            agent_analysis = None
            if self.agent:
                try:
                    # 使用代理的文本分析功能
                    agent_result = self.agent.analyze_antique(
                        image=None,
                        text_description=description
                    )
                    agent_analysis = agent_result
                except Exception as e:
                    logger.warning(f"代理文本分析失败: {e}")
                    # 如果代理失败，返回基本信息
                    agent_analysis = {
                        "analysis": f"无法分析文本描述: {description}",
                        "error": str(e)
                    }
            else:
                # 如果没有代理，返回基本信息
                agent_analysis = {
                    "analysis": f"收到文本描述: {description}，但AI代理未初始化",
                    "note": "需要初始化AI代理才能进行详细分析"
                }
            
            # 2. 整合结果（移除向量搜索）
            result = {
                "text_analysis": agent_analysis,
                "user_description": description,
                "analysis_timestamp": self._get_timestamp(),
                "from_cache": False,
                "optimization": "跳过向量检索，直接使用大模型分析",
                "model_info": {
                    "agent_info": self.agent.get_agent_info() if self.agent else None
                }
            }
            
            # 缓存结果
            self._manage_cache_size()
            self.text_analysis_cache[cache_key] = result.copy()
            logger.info(f"文本分析结果已缓存: {cache_key[:8]}...")
            
            logger.info("古董文本分析完成")
            return result
            
        except Exception as e:
            logger.error(f"古董文本分析失败: {e}")
            raise
    
    @log_method_call("add_antique_to_database")
    def add_antique_to_database(self, antique_id: int, 
                               image: Union[str, Image.Image, np.ndarray],
                               text_description: str,
                               metadata: Dict[str, Any]) -> bool:
        """
        添加古董到向量数据库
        
        Args:
            antique_id: 古董ID
            image: 图像
            text_description: 文本描述
            metadata: 元数据
            
        Returns:
            是否添加成功
        """
        try:
            # 1. 编码图像和文本
            image_features = self.clip_encoder.encode_image(image)
            text_features = self.clip_encoder.encode_text(text_description)
            
            # 2. 分别添加图像向量和文本向量到向量数据库
            # 插入图像向量
            self.vector_db.insert_vectors(
                antique_ids=[antique_id],
                vectors=[image_features.tolist()],
                vector_types=["image"],
                metadata=[{**metadata, "description": text_description, "vector_type": "image"}]
            )
            
            # 插入文本向量
            self.vector_db.insert_vectors(
                antique_ids=[antique_id],
                vectors=[text_features.tolist()],
                vector_types=["text"],
                metadata=[{**metadata, "description": text_description, "vector_type": "text"}]
            )
            
            logger.info(f"成功添加古董 {antique_id} 到向量数据库")
            return True
            
        except Exception as e:
            logger.error(f"添加古董到数据库失败: {e}")
            return False
    
    @log_method_call("batch_add_antiques")
    def batch_add_antiques(self, antiques_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量添加古董
        
        Args:
            antiques_data: 古董数据列表，每个元素包含 antique_id, image, text_description, metadata
            
        Returns:
            批量添加结果
        """
        try:
            results = {
                "success_count": 0,
                "failed_count": 0,
                "failed_items": []
            }
            
            for antique_data in antiques_data:
                try:
                    success = self.add_antique_to_database(
                        antique_id=antique_data["antique_id"],
                        image=antique_data["image"],
                        text_description=antique_data["text_description"],
                        metadata=antique_data["metadata"]
                    )
                    
                    if success:
                        results["success_count"] += 1
                    else:
                        results["failed_count"] += 1
                        results["failed_items"].append(antique_data["antique_id"])
                        
                except Exception as e:
                    results["failed_count"] += 1
                    results["failed_items"].append(antique_data["antique_id"])
                    logger.error(f"添加古董 {antique_data['antique_id']} 失败: {e}")
            
            logger.info(f"批量添加完成：成功 {results['success_count']} 个，失败 {results['failed_count']} 个")
            return results
            
        except Exception as e:
            logger.error(f"批量添加古董失败: {e}")
            raise
    
    @log_method_call("remove_antique_from_database")
    def remove_antique_from_database(self, antique_id: int) -> bool:
        """
        从向量数据库中删除古董
        
        Args:
            antique_id: 古董ID
            
        Returns:
            是否删除成功
        """
        try:
            success = self.vector_db.delete_by_antique_id(antique_id)
            if success:
                logger.info(f"成功从向量数据库删除古董 {antique_id}")
            else:
                logger.warning(f"从向量数据库删除古董 {antique_id} 失败")
            
            return success
            
        except Exception as e:
            logger.error(f"删除古董失败: {e}")
            return False
    
    @log_method_call("chat_with_internvl3_5")
    def chat_with_internvl3_5(self, image: Union[str, Image.Image, np.ndarray], 
                          message: str) -> str:
        """
        使用InternVL3_5模型与用户对话
        
        Args:
            image: 古董图像
            message: 用户消息
            
        Returns:
            InternVL3_5模型回复
        """
        try:
            if not self.internvl3_5_client:
                return "抱歉，InternVL3_5模型未初始化，请检查配置。"
            
            response = self.internvl3_5_client.chat_about_antique(image, message)
            return response
            
        except Exception as e:
            logger.error(f"与InternVL3_5模型对话失败: {e}")
            return f"抱歉，处理您的消息时出现错误: {str(e)}"
    
    def chat_with_smolvlm2(self, image: Union[str, Image.Image, np.ndarray], 
                          message: str) -> str:
        """
        使用SmolVLM2模型与用户对话（兼容性方法，实际使用InternVL3_5）
        
        Args:
            image: 古董图像
            message: 用户消息
            
        Returns:
            InternVL3_5模型回复
        """
        # 为了保持向后兼容性，重定向到新的方法
        return self.chat_with_internvl3_5(image, message)
    
    @log_method_call("chat_with_agent")
    def chat_with_agent(self, message: str, image: Union[str, Image.Image, np.ndarray] = None) -> str:
        """
        与 AI 代理对话（支持本地模型混合使用）
        
        Args:
            message: 用户消息
            image: 可选的图像输入
            
        Returns:
            代理回复
        """
        try:
            if not self.agent:
                return "抱歉，AI代理未初始化。"
            
            response = self.agent.chat(message, image)
            return response
            
        except Exception as e:
            logger.error(f"与代理对话失败: {e}")
            return f"抱歉，处理您的消息时出现错误: {str(e)}"
    
    @log_method_call("get_system_status")
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态信息
        """
        try:
            status = {
                "clip_encoder": self.clip_encoder.get_model_info() if self.clip_encoder else None,
                "vector_db": self.vector_db.get_collection_stats() if self.vector_db else None,
                "agent": self.agent.get_agent_info() if self.agent else None,
                "internvl3_5_client": self.internvl3_5_client.get_model_info() if self.internvl3_5_client else None,
                "system_timestamp": self._get_timestamp()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            raise
    
    @log_method_call("get_conversation_history")
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Returns:
            对话历史列表
        """
        try:
            return self.agent.get_conversation_history()
            
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []
    
    @log_method_call("clear_conversation_history")
    def clear_conversation_history(self):
        """清除对话历史"""
        try:
            self.agent.clear_memory()
            logger.info("对话历史已清除")
            
        except Exception as e:
            logger.error(f"清除对话历史失败: {e}")
    
    def _init_cache(self):
        """初始化缓存系统"""
        import hashlib
        self.text_analysis_cache = {}  # 文本分析结果缓存
        self.cache_max_size = 100  # 最大缓存条目数
        logger.info("缓存系统初始化完成")
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _manage_cache_size(self):
        """管理缓存大小，防止内存溢出"""
        if len(self.text_analysis_cache) > self.cache_max_size:
            # 删除最旧的缓存条目
            oldest_key = next(iter(self.text_analysis_cache))
            del self.text_analysis_cache[oldest_key]
            logger.debug(f"删除最旧的缓存条目: {oldest_key}")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    @log_method_call("close")
    def close(self):
        """关闭所有连接"""
        try:
            if self.vector_db:
                self.vector_db.close()
            if self.internvl3_5_client:
                self.internvl3_5_client.cleanup()
            logger.info("AI 核心已关闭")
            
        except Exception as e:
            logger.error(f"关闭 AI 核心失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
