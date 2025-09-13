#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InternVL3_5-1B 多模态模型客户端
基于 transformers 库实现的 InternVL3_5-1B 模型推理客户端
支持图像分析、文本生成和古董鉴定功能
"""

import os
import sys
import torch
import json
import logging
import tempfile
from PIL import Image
from typing import Union, Dict, Any, Optional, List
import numpy as np
from datetime import datetime

# 添加项目根目录到系统路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(project_root)

# 尝试导入logger配置
try:
    from utils.logger_config import get_ai_logger
except ImportError:
    # 如果导入失败，创建简单的logger
    import logging
    def get_ai_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

# 配置日志
logger = get_ai_logger('internvl3_5_client')

# 检查 transformers 库是否可用
try:
    from transformers import (
        AutoProcessor, AutoModel,
        AutoConfig, AutoTokenizer,
        TextIteratorStreamer
    )
    from threading import Thread
    TRANSFORMERS_AVAILABLE = True
    logger.info("transformers 库可用")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.error(f"transformers 库不可用: {e}")

# 检查 GPU 支持
CUDA_AVAILABLE = torch.cuda.is_available()


class InternVL3_5Client:
    """InternVL3_5-1B 多模态模型客户端
    
    使用 transformers 库的 AutoModel 加载和推理 InternVL3_5-1B 模型
    支持图像分析、对话和古董鉴定功能
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 max_tokens: int = 512,
                 temperature: float = 0.7):
        """
        初始化 InternVL3_5 客户端
        
        Args:
            model_path: 本地模型路径（可选）
            device: 设备类型（cuda/cpu/auto）
            max_tokens: 最大生成token数
            temperature: 生成温度
            
        Raises:
            RuntimeError: 当 transformers 库不可用或模型加载失败时
        """
        # 检查 transformers 库
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers 库不可用，InternVL3_5 客户端需要 transformers 库才能运行。\n"
                "请安装：pip install transformers>=4.46.0"
            )
        
        # 确定模型路径
        self.model_path = model_path or self._get_default_model_path()
        
        # 设备配置
        if device == "auto":
            device_str = "cuda" if CUDA_AVAILABLE else "cpu"
            self.device = torch.device(device_str)
        else:
            device_str = device or ("cuda" if CUDA_AVAILABLE else "cpu")
            self.device = torch.device(device_str)
        
        # 生成参数
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # 模型组件
        self.model = None
        self.processor = None
        
        # 初始化状态
        self.is_initialized = False
        
        logger.info(f"InternVL3_5客户端初始化: 设备={self.device}, 模型路径={self.model_path}")
        
        # 初始化模型
        self._init_model()
    
    def _get_default_model_path(self) -> str:
        """获取默认模型路径"""
        # 检查本地模型路径
        local_paths = [
            os.path.join(os.path.dirname(__file__), "..", "InternVL3_5-1B"),
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "InternVL3_5-1B"),
            "./models/InternVL3_5-1B",
            "./InternVL3_5-1B"
        ]
        
        for path in local_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) and os.path.isdir(abs_path):
                config_file = os.path.join(abs_path, "config.json")
                if os.path.exists(config_file):
                    logger.info(f"找到本地模型: {abs_path}")
                    return abs_path
        
        # 如果没有找到本地模型，返回HuggingFace模型ID
        logger.warning("未找到本地模型，将使用HuggingFace模型")
        return "OpenGVLab/InternVL3_5-1B"
    
    def _init_model(self):
        """初始化 InternVL3_5 模型"""
        try:
            logger.info(f"正在加载 InternVL3_5 模型: {self.model_path}")
            
            # 加载tokenizer和image processor
            logger.info("正在加载 tokenizer 和 image processor...")
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # 直接加载tokenizer，避免processor的兼容性问题
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    use_fast=False
                )
                
                # 创建一个简单的processor包装器
                class SimpleProcessor:
                    def __init__(self, tokenizer):
                        self.tokenizer = tokenizer
                    
                    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True):
                        # 简单的聊天模板实现
                        if len(conversation) == 1 and conversation[0]["role"] == "user":
                            content = conversation[0]["content"]
                            if isinstance(content, list):
                                # 多模态内容
                                text_parts = [item["text"] for item in content if item["type"] == "text"]
                                text = " ".join(text_parts)
                            else:
                                text = content
                            
                            if add_generation_prompt:
                                prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
                            else:
                                prompt = f"<|im_start|>user\n{text}<|im_end|>\n"
                            return prompt
                        return ""
                    
                    def __call__(self, text=None, images=None, return_tensors="pt"):
                        if text is not None:
                            return self.tokenizer(text, return_tensors=return_tensors)
                        return {}
                    
                    def batch_decode(self, *args, **kwargs):
                        return self.tokenizer.batch_decode(*args, **kwargs)
                
                self.processor = SimpleProcessor(self.tokenizer)
            
            # 加载模型
            logger.info("正在加载 AutoModel...")
            
            # 根据设备类型配置模型加载参数
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            if self.device.type == "cuda" and CUDA_AVAILABLE:
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["device_map"] = "auto"
            else:
                # CPU模式下使用float32，避免bfloat16兼容性问题
                model_kwargs["torch_dtype"] = torch.float32
                logger.info("使用CPU模式，设置torch_dtype为float32")
            
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
                
                # 如果是CPU模式，手动移动模型到CPU
                if self.device.type == "cpu":
                    self.model = self.model.to(self.device)
                    logger.info("模型已移动到CPU设备")
                
            except torch.cuda.OutOfMemoryError as cuda_error:
                if self.device.type == "cuda":
                    logger.warning(f"CUDA内存不足，自动回退到CPU: {cuda_error}")
                    # 清理CUDA缓存
                    torch.cuda.empty_cache()
                    
                    # 切换到CPU并重新配置参数
                    self.device = torch.device("cpu")
                    model_kwargs["torch_dtype"] = torch.float32
                    if "device_map" in model_kwargs:
                        del model_kwargs["device_map"]
                    
                    logger.info("重新使用CPU模式加载模型...")
                    self.model = AutoModel.from_pretrained(
                        self.model_path,
                        **model_kwargs
                    )
                    self.model = self.model.to(self.device)
                    logger.info("模型已成功回退到CPU设备")
                else:
                    raise cuda_error
            
            # 设置评估模式
            self.model.eval()
            
            self.is_initialized = True
            logger.info("✅ InternVL3_5 模型初始化完成")
            
        except Exception as e:
            logger.error(f"❌ InternVL3_5 模型初始化失败: {e}")
            raise RuntimeError(f"模型初始化失败: {e}")
    
    def chat_about_antique_stream(self, query: str, image_path: Optional[str] = None):
        """
        古董鉴定对话方法 - 流式输出版本
        
        Args:
            query: 用户查询
            image_path: 图片路径（可选，如果提供则使用多模态功能）
            
        Yields:
            str: 流式输出的文本片段
        """
        try:
            if not self.is_initialized:
                yield "模型未初始化，请先调用initialize()方法"
                return
            
            # 构建古董鉴定专用提示词
            antique_prompt = f"""
你是一位专业的古董鉴定专家，拥有丰富的文物鉴定经验。请根据用户的问题，提供专业、准确的古董鉴定建议。

用户问题：{query}

请从以下几个方面进行分析：
1. 器物特征分析
2. 年代判断依据
3. 工艺特点
4. 价值评估
5. 收藏建议

请用专业但易懂的语言回答：
"""
            
            # 处理图像输入
            pixel_values = None
            if image_path and os.path.exists(image_path):
                try:
                    # 加载并处理图像
                    from PIL import Image
                    import torchvision.transforms as transforms
                    
                    image = Image.open(image_path).convert('RGB')
                    
                    # 图像预处理
                    transform = transforms.Compose([
                        transforms.Resize((448, 448)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    
                    # 根据模型的实际数据类型设置输入数据类型
                    model_dtype = next(self.model.parameters()).dtype
                    pixel_values = transform(image).unsqueeze(0).to(self.device, dtype=model_dtype)
                    logger.info(f"✅ 成功加载图像: {image_path}，设备: {self.device}")
                    
                except Exception as img_e:
                    logger.warning(f"图像加载失败: {img_e}，将使用纯文本模式")
                    pixel_values = None
            
            # 创建流式输出器
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # 使用官方的generation_config
            generation_config = dict(
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                streamer=streamer
            )
            
            # 在线程中运行生成
            def generate():
                self.model.chat(
                    self.processor.tokenizer, 
                    pixel_values,  # 如果有图像则传入pixel_values，否则为None
                    antique_prompt, 
                    generation_config
                )
            
            thread = Thread(target=generate)
            thread.start()
            
            # 流式输出
            for new_text in streamer:
                yield new_text
            
            thread.join()
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 古董鉴定流式对话失败: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
            yield f"古董鉴定对话过程中出现错误: {type(e).__name__}: {str(e)}"
    
    def chat_about_antique(self, query: str, image_path: Optional[str] = None) -> str:
        """
        古董鉴定对话方法 - 支持图文混合输入
        
        Args:
            query: 用户查询
            image_path: 图片路径（可选，如果提供则使用多模态功能）
            
        Returns:
            str: 模型回复
        """
        try:
            if not self.is_initialized:
                return "模型未初始化，请先调用initialize()方法"
            
            # 构建古董鉴定专用提示词
            antique_prompt = f"""
你是一位专业的古董鉴定专家，拥有丰富的文物鉴定经验。请根据用户的问题，提供专业、准确的古董鉴定建议。

用户问题：{query}

请从以下几个方面进行分析：
1. 器物特征分析
2. 年代判断依据
3. 工艺特点
4. 价值评估
5. 收藏建议

请用专业但易懂的语言回答：
"""
            
            # 使用官方的generation_config
            generation_config = dict(
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False
            )
            
            # 处理图像输入
            pixel_values = None
            if image_path and os.path.exists(image_path):
                try:
                    # 加载并处理图像
                    from PIL import Image
                    import torchvision.transforms as transforms
                    
                    image = Image.open(image_path).convert('RGB')
                    
                    # 图像预处理
                    transform = transforms.Compose([
                        transforms.Resize((448, 448)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    
                    # 根据设备类型设置数据类型
                    if self.device == "cuda" and CUDA_AVAILABLE:
                        pixel_values = transform(image).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
                    else:
                        pixel_values = transform(image).unsqueeze(0).to(self.device, dtype=torch.float32)
                    logger.info(f"✅ 成功加载图像: {image_path}，设备: {self.device}")
                    
                except Exception as img_e:
                    logger.warning(f"图像加载失败: {img_e}，将使用纯文本模式")
                    pixel_values = None
            
            # 使用官方的chat方法：model.chat(tokenizer, pixel_values, question, generation_config)
            response = self.model.chat(
                self.processor.tokenizer, 
                pixel_values,  # 如果有图像则传入pixel_values，否则为None
                antique_prompt, 
                generation_config
            )
            
            return response
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 古董鉴定对话失败: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
            return f"古董鉴定对话过程中出现错误: {type(e).__name__}: {str(e)}"
    
    def text_chat(self, message: str) -> str:
        """纯文本对话
        
        Args:
            message: 用户消息
            
        Returns:
            str: 模型回复
        """
        try:
            if not self.is_initialized:
                return "模型未初始化，请检查模型配置"
            
            # 构建对话格式
            conversation = [
                {
                    "role": "user",
                    "content": message
                }
            ]
            
            # 应用聊天模板
            prompt_text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # 处理输入
            inputs = self.processor(
                text=prompt_text,
                return_tensors="pt"
            ).to(self.device)
            
            # 生成回复
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # 解码回复
            input_length = inputs["input_ids"].shape[1]
            generated_text = self.processor.tokenizer.decode(
                generated_ids[0][input_length:], 
                skip_special_tokens=True
            ).strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"文本对话失败: {e}")
            return f"抱歉，处理您的消息时出现错误: {str(e)}"
    
    def chat(self, query: str, image_path: Optional[str] = None) -> str:
        """
        通用对话方法 - 支持图文混合输入
        
        Args:
            query: 用户查询
            image_path: 图片路径（可选，如果提供则使用多模态功能）
            
        Returns:
            str: 模型回复
        """
        try:
            if not self.is_initialized:
                return "模型未初始化，请先调用initialize()方法"
            
            # 使用官方的generation_config
            generation_config = dict(
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False
            )
            
            # 处理图像输入
            pixel_values = None
            if image_path and os.path.exists(image_path):
                try:
                    # 加载并处理图像
                    from PIL import Image
                    import torchvision.transforms as transforms
                    
                    image = Image.open(image_path).convert('RGB')
                    
                    # 图像预处理
                    transform = transforms.Compose([
                        transforms.Resize((448, 448)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    
                    # 根据设备类型设置数据类型
                    if self.device == "cuda" and CUDA_AVAILABLE:
                        pixel_values = transform(image).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
                    else:
                        pixel_values = transform(image).unsqueeze(0).to(self.device, dtype=torch.float32)
                    logger.info(f"✅ 成功加载图像: {image_path}，设备: {self.device}")
                    
                except Exception as img_e:
                    logger.warning(f"图像加载失败: {img_e}，将使用纯文本模式")
                    pixel_values = None
            
            # 使用官方的chat方法：model.chat(tokenizer, pixel_values, question, generation_config)
            response = self.model.chat(
                self.processor.tokenizer, 
                pixel_values,  # 如果有图像则传入pixel_values，否则为None
                query, 
                generation_config
            )
            
            return response
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 对话失败: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
            return f"对话过程中出现错误: {type(e).__name__}: {str(e)}"
    

    
    def generate_appraisal_report(self,
                                image: Union[str, Image.Image, np.ndarray],
                                user_query: str = "") -> Dict[str, Any]:
        """
        生成古董鉴定报告
        
        Args:
            image: 古董图像（可以是文件路径、PIL.Image对象或numpy数组）
            user_query: 用户特别关注的问题
            
        Returns:
            鉴定报告字典
        """
        try:
            # 处理不同类型的图像输入
            image_path = None
            
            if isinstance(image, str):
                # 如果是字符串，假设是文件路径
                if os.path.exists(image):
                    image_path = image
                else:
                    logger.warning(f"图像路径不存在: {image}")
            elif isinstance(image, Image.Image):
                # 如果是PIL Image对象，保存为临时文件
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"temp_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                image.save(temp_path)
                image_path = temp_path
                logger.info(f"PIL Image已保存为临时文件: {temp_path}")
            elif isinstance(image, np.ndarray):
                # 如果是numpy数组，转换为PIL Image再保存
                pil_image = Image.fromarray(image.astype('uint8'))
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"temp_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                pil_image.save(temp_path)
                image_path = temp_path
                logger.info(f"numpy数组已转换并保存为临时文件: {temp_path}")
            else:
                logger.warning(f"不支持的图像类型: {type(image)}")
            
            # 构建专业的古董鉴定提示词
            base_prompt = """作为一位资深的古董鉴定专家，请按照以下模板对这件文物进行详细的专业鉴赏分析：

**一、藏品基础信息（关键信息分类）**
**藏品核心属性**
- 藏品名称：【精准标注品类、年代（若初步判断）、核心特征】
- 材质类型：【明确材质及细分品类，如陶瓷（青花瓷/青瓷）、金属（黄铜/白银）、玉石（和田玉/翡翠）等】
- 规格参数：【尽可能估算尺寸规格】
- 外观特征：【简要描述整体状态和保存情况】

**二、藏品真伪判定**
**宏观特征比对分析**
- 器型/形制比对：【对比同期标准器特征，分析器型是否符合历史特征】
- 纹饰/工艺判断：【分析纹饰细节与工艺特点，判断是否符合时代特征】
- 胎质/釉色/材质老化观察：【针对不同材质分析老化特征和制作工艺】
- 款识/印记考证（如有）：【分析款识字体、章法等特征】

**三、鉴赏声明（不承诺与专业建议）**
- 结论局限性声明：【明确本报告结论的边界和局限性】
- 无承诺声明：【声明不承担市场价值等责任】
- 专业复核建议：【建议寻求更权威的专业鉴定】
- 来源合法性声明：【明确委托方责任】

请严格按照上述模板格式提供详细、专业的鉴赏报告，用中文回答。"""

            if user_query:
                prompt = f"{base_prompt}\n\n请特别关注以下要求：{user_query}。请直接提供分析结果，不要重复用户的要求。"
            else:
                prompt = base_prompt
            
            # 获取分析结果
            analysis_text = self.chat_about_antique(prompt, image_path)
            
            # 构建报告
            report = {
                "model_name": "InternVL3_5-1B",
                "analysis_text": analysis_text,
                "user_query": user_query,
                "timestamp": self._get_timestamp(),
                "model_info": self.get_model_info()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成鉴定报告失败: {e}")
            return {
                "error": f"报告生成失败: {str(e)}",
                "model_name": "InternVL3_5-1B",
                "timestamp": self._get_timestamp()
            }
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        info = {
            "model_name": "InternVL3_5-1B",
            "model_path": self.model_path,
            "device": self.device,
            "is_initialized": self.is_initialized,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "cuda_available": CUDA_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE
        }
        
        if self.is_initialized and self.model:
            try:
                # 计算参数数量
                total_params = sum(p.numel() for p in self.model.parameters())
                info["total_parameters"] = total_params
                info["model_dtype"] = str(next(self.model.parameters()).dtype)
                
                # GPU内存使用情况
                if CUDA_AVAILABLE and self.device == "cuda":
                    info["gpu_memory_allocated"] = torch.cuda.memory_allocated(0) / 1024**3
                    info["gpu_memory_reserved"] = torch.cuda.memory_reserved(0) / 1024**3
            except Exception as e:
                logger.warning(f"获取详细模型信息失败: {e}")
        
        return info
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
                self.processor = None
            
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
            
            import gc
            gc.collect()
            
            self.is_initialized = False
            logger.info("InternVL3_5客户端资源已清理")
        except Exception as e:
            logger.warning(f"清理资源时出现错误: {e}")
    
    def __del__(self):
        """析构函数"""
        self.cleanup()


# 兼容性别名
InternVLClient = InternVL3_5Client