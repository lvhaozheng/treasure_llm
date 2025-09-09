#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 客户端实现
支持 Qwen3 模型的加载、推理和对话功能
"""

import os
import sys
import torch
import json
import logging
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional, Union

# 版本检查
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
    from transformers import BitsAndBytesConfig, TextIteratorStreamer
    from threading import Thread
except ImportError as e:
    print(f"❌ 导入transformers失败: {e}")
    print("请安装: pip install transformers>=4.47.0")
    sys.exit(1)

# CUDA可用性检查
CUDA_AVAILABLE = torch.cuda.is_available()

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Qwen3Client:
    """Qwen3 模型客户端"""
    
    def __init__(self, device="auto", quantization="none", model_path=None):
        """
        初始化 Qwen3 客户端
        
        Args:
            device: 设备选择 ("auto", "cuda", "cpu")
            quantization: 量化模式 ("none", "4bit", "8bit")
            model_path: 模型路径 (本地路径或HuggingFace模型名)
        """
        self.device = self._determine_device(device)
        self.quantization = quantization
        self.model_path = self._determine_model_path(model_path)
        
        # 模型和分词器
        self.model = None
        self.tokenizer = None
        
        logger.info(f"初始化 Qwen3 客户端: 设备={self.device}, 量化={quantization}")
        
        # 配置CUDA内存优化
        if self.device == "cuda":
            self._configure_cuda_memory()
        
        # 初始化模型
        self._init_model()
    
    def _determine_device(self, device):
        """确定使用的设备"""
        if device == "auto":
            return "cuda" if CUDA_AVAILABLE else "cpu"
        elif device == "cuda" and not CUDA_AVAILABLE:
            logger.warning("CUDA不可用，回退到CPU模式")
            return "cpu"
        return device
    
    def _determine_model_path(self, model_path):
        """确定模型路径"""
        if model_path:
            return model_path
        
        # 检查本地模型路径
        base_dir = os.path.dirname(__file__)  # models/client 目录
        models_dir = os.path.dirname(base_dir)  # models 目录
        
        possible_paths = [
            os.path.join(models_dir, "qwen3-0.6B"),  # 本地 qwen3-0.6B 模型
            os.path.join(models_dir, "qwen3_model"),  # 备用路径
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                config_file = os.path.join(path, 'config.json')
                if os.path.exists(config_file):
                    logger.info(f"找到本地 Qwen3 模型: {path}")
                    return path
        
        # 使用默认的HuggingFace模型
        logger.info("未找到本地模型，将从 HuggingFace 下载")
        return "Qwen/Qwen2.5-7B-Instruct"  # 默认使用Qwen2.5
    
    def _configure_cuda_memory(self):
        """配置 CUDA 显存优化参数"""
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        if CUDA_AVAILABLE and self.device == "cuda":
            try:
                torch.cuda.empty_cache()
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"CUDA显存优化已启用: {gpu_memory_gb:.1f}GB GPU")
                
                if gpu_memory_gb < 8:
                    torch.cuda.set_per_process_memory_fraction(0.8)
                    logger.info("启用显存优化模式")
                    
            except Exception as e:
                logger.warning(f"CUDA显存优化配置失败: {e}")
    
    def _init_model(self):
        """初始化 Qwen3 模型"""
        try:
            logger.info(f"正在加载 Qwen3 模型: {self.model_path}")
            
            # 清理内存
            self._cleanup_memory()
            
            # 获取加载配置
            load_kwargs = self._get_optimal_load_config()
            
            # 加载分词器
            logger.info("正在加载 Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            logger.info("正在加载 Model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **load_kwargs
            )
            
            # CPU模式需要手动移动到设备
            if self.device == "cpu" and "device_map" not in load_kwargs:
                self.model = self.model.to(self.device)
            
            # 设置模型为评估模式
            self.model.eval()
            
            # 显示成功信息
            actual_dtype = next(self.model.parameters()).dtype if self.model else "unknown"
            logger.info(f"✅ 成功加载 Qwen3 模型")
            logger.info(f"📝 模型类型: Qwen3")
            logger.info(f"📝 设备: {self.device}")
            logger.info(f"📝 数据类型: {actual_dtype}")
            logger.info(f"📝 词汇表大小: {self.tokenizer.vocab_size}")
            
            # 显示内存使用情况
            if CUDA_AVAILABLE and self.device == "cuda":
                allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"📝 GPU内存使用: {allocated_memory:.1f}GB")
                
        except Exception as e:
            logger.error(f"Qwen3 模型加载失败: {e}")
            self._cleanup_memory()
            raise RuntimeError(f"Qwen3 模型初始化失败: {e}")
    
    def _get_optimal_load_config(self):
        """获取最优的模型加载配置"""
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        
        # CPU模式配置
        if not CUDA_AVAILABLE or self.device == "cpu":
            logger.info("使用CPU模式加载模型")
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch.float32
            return load_kwargs
        
        # GPU模式配置
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU显存: {gpu_memory_gb:.1f}GB")
            
            # 根据量化模式选择配置
            if self.quantization == "4bit":
                if gpu_memory_gb < 4:
                    logger.warning("显存不足，强制使用CPU模式")
                    load_kwargs["device_map"] = "cpu"
                    load_kwargs["torch_dtype"] = torch.float32
                else:
                    logger.info("使用4bit量化模式")
                    load_kwargs["quantization_config"] = self._get_4bit_config()
                    load_kwargs["device_map"] = "auto"
                    
            elif self.quantization == "8bit":
                if gpu_memory_gb < 6:
                    logger.warning("显存不足，降级到4bit量化")
                    load_kwargs["quantization_config"] = self._get_4bit_config()
                else:
                    logger.info("使用8bit量化模式")
                    load_kwargs["quantization_config"] = self._get_8bit_config()
                load_kwargs["device_map"] = "auto"
                
            else:  # 无量化模式
                if gpu_memory_gb < 8:
                    logger.warning("显存不足，自动启用4bit量化")
                    load_kwargs["quantization_config"] = self._get_4bit_config()
                    load_kwargs["device_map"] = "auto"
                else:
                    logger.info("显存充足，使用bfloat16精度")
                    load_kwargs["device_map"] = "auto"
                    
        except Exception as e:
            logger.warning(f"GPU配置检测失败: {e}")
            load_kwargs["device_map"] = "auto"
        
        return load_kwargs
    
    def _get_4bit_config(self):
        """获取4bit量化配置"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    def _get_8bit_config(self):
        """获取8bit量化配置"""
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    
    def _cleanup_memory(self):
        """清理内存和显存缓存"""
        try:
            import gc
            gc.collect()
            
            if CUDA_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            logger.warning(f"内存清理失败: {e}")
    
    def chat_stream(self, messages, max_length=512, temperature=0.7, top_p=0.9):
        """使用标准messages格式进行流式对话"""
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("模型未初始化")
            
            logger.info(f"处理流式对话，消息数量: {len(messages)}")
            
            # 应用聊天模板，禁用thinking模式
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # 禁用thinking模式，直接返回最终答案
            )
            
            # 编码输入
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # 创建流式输出器
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # 生成配置
            generation_config = {
                "max_new_tokens": max_length,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer
            }
            
            # 在线程中运行生成
            def generate():
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        **generation_config
                    )
            
            thread = Thread(target=generate)
            thread.start()
            
            # 流式输出
            for new_text in streamer:
                yield new_text
            
            thread.join()
            
        except Exception as e:
            logger.error(f"流式对话失败: {e}")
            yield f"对话过程中出现错误: {type(e).__name__}: {str(e)}"
    
    def chat(self, messages, max_length=512, temperature=0.7, top_p=0.9):
        """使用标准messages格式进行对话"""
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("模型未初始化")
            
            logger.info(f"处理对话，消息数量: {len(messages)}")
            
            # 应用聊天模板，禁用thinking模式
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # 禁用thinking模式，直接返回最终答案
            )
            
            # 编码输入
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # 生成配置
            generation_config = {
                "max_new_tokens": max_length,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 解码回复
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            logger.info(f"对话完成，生成文本长度: {len(generated_text)}")
            return generated_text
            
        except Exception as e:
            logger.error(f"对话失败: {e}")
            raise
    
    def text_chat(self, question, max_length=512):
        """简单文本对话"""
        messages = [
            {"role": "user", "content": question}
        ]
        return self.chat(messages, max_length)
    
    def get_model_info(self):
        """获取模型信息"""
        try:
            info = {
                "model_name": "Qwen3",
                "model_path": self.model_path,
                "device": self.device,
                "quantization": self.quantization,
                "transformers_version": self._get_transformers_version(),
                "cuda_available": CUDA_AVAILABLE,
                "model_loaded": self.model is not None,
                "tokenizer_loaded": self.tokenizer is not None
            }
            
            if self.model:
                info["model_dtype"] = str(next(self.model.parameters()).dtype)
                
            if self.tokenizer:
                info["vocab_size"] = self.tokenizer.vocab_size
                info["max_length"] = getattr(self.tokenizer, 'model_max_length', 'unknown')
            
            if CUDA_AVAILABLE and self.device == "cuda":
                info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.1f}GB"
            
            return info
            
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {"error": str(e)}
    
    def _get_transformers_version(self):
        """获取transformers版本"""
        try:
            import transformers
            return transformers.__version__
        except:
            return "unknown"
    
    def cleanup(self):
        """清理资源"""
        try:
            logger.info("正在清理Qwen3模型资源...")
            
            if hasattr(self, 'model') and self.model:
                del self.model
                self.model = None
            
            if hasattr(self, 'tokenizer') and self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            self._cleanup_memory()
            logger.info("Qwen3模型资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")


def create_qwen3_client(device="auto", quantization="none", model_path=None):
    """创建Qwen3客户端的便捷函数"""
    try:
        client = Qwen3Client(device=device, quantization=quantization, model_path=model_path)
        logger.info("Qwen3客户端创建成功")
        return client
    except Exception as e:
        logger.error(f"Qwen3客户端创建失败: {e}")
        raise