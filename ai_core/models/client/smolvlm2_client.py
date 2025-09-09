"""
SmolVLM2 多模态模型客户端
基于 transformers 库实现的 SmolVLM2 模型推理客户端
支持图像分析、文本生成和古董鉴定功能
"""

import os
import sys
import torch
import json
import logging
from PIL import Image
from typing import Union, Dict, Any, Optional, List
import numpy as np
from datetime import datetime

# 版本比较相关导入
try:
    from packaging import version
except ImportError:
    # 如果packaging不可用，使用distutils作为备选
    try:
        from distutils.version import LooseVersion as version
    except ImportError:
        version = None

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
logger = get_ai_logger('smolvlm2_client')

# 检查 transformers 库是否可用
try:
    from transformers import (
        AutoProcessor, AutoModelForVision2Seq,
        AutoConfig, AutoTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
    logger.info("transformers 库可用")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.error(f"transformers 库不可用: {e}")

# 检查 GPU 支持
CUDA_AVAILABLE = torch.cuda.is_available()


class SmolVLM2Client:
    """SmolVLM2 多模态模型客户端
    
    使用 transformers 库的 AutoModelForVision2Seq 加载和推理 SmolVLM2 模型
    支持图像分析、对话和古董鉴定功能
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 quantization_mode: Optional[str] = None):
        """
        初始化 SmolVLM2 客户端
        
        Args:
            model_path: 本地模型路径（可选）
            device: 设备类型（cuda/cpu/auto）
            max_tokens: 最大生成token数
            temperature: 生成温度
            quantization_mode: 量化模式（auto/4bit/8bit/none）
            
        Raises:
            RuntimeError: 当 transformers 库不可用或模型加载失败时
        """
        # 检查 transformers 库
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers 库不可用，SmolVLM2 客户端需要 transformers 库才能运行。\n"
                "请安装：pip install transformers>=4.46.0"
            )
        
        # 确定模型路径
        self.model_path = model_path or self._get_default_model_path()
        logger.info(f"使用模型路径: {self.model_path}")
        
        # GPU设备智能检测和配置
        if device == "auto" or device is None:
            if CUDA_AVAILABLE:
                self.device = "cuda"
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
                logger.info(f"检测到GPU: {gpu_name}, 显存: {gpu_memory:.1f}GB")
                logger.info(f"将使用GPU加速 SmolVLM2 模型推理")
            else:
                self.device = "cpu"
                logger.info("未检测到可用GPU，将使用CPU模式")
        else:
            self.device = device
            if device == "cuda" and not CUDA_AVAILABLE:
                logger.warning("指定使用GPU但未检测到CUDA，回退到CPU模式")
                self.device = "cpu"
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.quantization_mode = quantization_mode  # 新增量化模式参数
        
        # 验证量化模式参数
        valid_quant_modes = [None, "auto", "4bit", "8bit", "none"]
        if self.quantization_mode not in valid_quant_modes:
            logger.warning(f"无效的量化模式: {self.quantization_mode}，使用全量精度加载")
            self.quantization_mode = "none"
        
        # 如果未指定量化模式，使用none作为默认值（全量加载）
        if self.quantization_mode is None or self.quantization_mode == "auto":
            self.quantization_mode = "none"
            logger.info("未指定量化模式，使用全量精度加载")
        
        # 只有在需要量化时才检查bitsandbytes兼容性
        if self.quantization_mode in ["4bit", "8bit"]:
            # 检查bitsandbytes版本兼容性
            if not self._check_bitsandbytes_compatibility():
                logger.warning("检测到bitsandbytes兼容性问题，自动切换到CPU模式并禁用量化")
                self.device = "cpu"
                self.quantization_mode = "none"
        else:
            logger.info("使用全量精度加载，跳过bitsandbytes检查")
        
        if self.quantization_mode:
            logger.info(f"指定量化模式: {self.quantization_mode}")
        
        self.model = None
        self.processor = None
        
        # 初始化模型
        self._init_model()
        
        # 配置 CUDA 显存优化
        self._configure_cuda_memory()
    
    def _get_default_model_path(self) -> str:
        """获取默认模型路径"""
        base_dir = os.path.dirname(__file__)  # models/client 目录
        models_dir = os.path.dirname(base_dir)  # models 目录
        
        # 优先检查 SmolVLM2 256M model (轻量级模型)
        small_model_path = os.path.join(models_dir, "smolvlm2_256M")
        if os.path.exists(small_model_path):
            # 检查关键配置文件
            required_files = ['config.json', 'tokenizer_config.json', 'processor_config.json']
            missing_files = []
            for file in required_files:
                file_path = os.path.join(small_model_path, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if not missing_files:
                logger.info(f"找到 SmolVLM2 Small Model (256M): {small_model_path}")
                logger.info("使用轻量级 SmolVLM2-256M 模型，内存占用更少")
                return small_model_path
            else:
                logger.warning(f"Small model 路径缺少文件: {missing_files}")
        
        # 检查标准 SmolVLM2 模型目录 (2.2B参数)
        default_model_path = os.path.join(models_dir, "SmolVLM2_model")
        if os.path.exists(default_model_path):
            # 检查关键配置文件
            required_files = ['config.json', 'tokenizer_config.json', 'processor_config.json']
            missing_files = []
            for file in required_files:
                file_path = os.path.join(default_model_path, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if not missing_files:
                logger.info(f"找到标准 SmolVLM2 模型 (2.2B): {default_model_path}")
                return default_model_path
            else:
                logger.warning(f"标准模型路径缺少文件: {missing_files}")
        
        # 优先使用 ModelScope 模型（国内访问更快）
        logger.info("未找到本地模型，将优先从 ModelScope 下载")
        return "HuggingFaceTB/SmolVLM2-2.2B-Base"
    
    def _configure_cuda_memory(self):
        """配置 CUDA 显存优化参数"""
        # 设置PyTorch内存优化环境变量
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        if CUDA_AVAILABLE and self.device == "cuda":
            try:
                # 清理显存缓存
                torch.cuda.empty_cache()
                
                # 获取GPU信息
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                gpu_free_gb = gpu_free_memory / 1024**3
                
                logger.info(f"CUDA显存优化已启用: {gpu_memory_gb:.1f}GB GPU, 可用: {gpu_free_gb:.1f}GB")
                
                # 根据显存大小配置优化策略
                if gpu_memory_gb < 6:
                    logger.warning(f"显存较小({gpu_memory_gb:.1f}GB)，将启用激进内存优化")
                    # 启用显存分片以避免内存碎片
                    torch.cuda.set_per_process_memory_fraction(0.8)
                elif gpu_memory_gb < 8:
                    logger.info(f"中等显存({gpu_memory_gb:.1f}GB)，启用标准内存优化")
                    torch.cuda.set_per_process_memory_fraction(0.9)
                else:
                    logger.info(f"充足显存({gpu_memory_gb:.1f}GB)，使用默认内存设置")
                
            except Exception as e:
                logger.warning(f"CUDA显存优化配置失败: {e}")
        
        # 配置通用内存优化
        try:
            import gc
            gc.collect()
            
            # 设置PyTorch线程数（避免CPU过载）
            optimal_threads = min(4, max(1, torch.get_num_threads() // 2))
            torch.set_num_threads(optimal_threads)
            logger.info(f"设置PyTorch线程数: {optimal_threads}")
            
            # 启用内存映射以减少内存占用
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
        except Exception as e:
            logger.warning(f"通用内存优化配置失败: {e}")
    
    def _init_model(self):
        """初始化 SmolVLM2 模型（按照ModelScope标准实现）"""
        try:
            logger.info(f"正在使用 transformers AutoModelForVision2Seq 加载 SmolVLM2 模型: {self.model_path}")
            
            # 检查模型配置
            if os.path.exists(self.model_path):
                config_path = os.path.join(self.model_path, 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    model_type = config.get('model_type', 'unknown')
                    architectures = config.get('architectures', [])
                    logger.info(f"模型类型: {model_type}, 架构: {architectures}")
            
            # 清理内存
            self._cleanup_memory()
            
            # 获取最优加载配置
            load_kwargs = self._get_optimal_load_config()
            
            # 加载处理器（轻量级，优先加载）
            logger.info("正在加载 AutoProcessor...")
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*special tokens.*")
                warnings.filterwarnings("ignore", message=".*image processor.*")
                warnings.filterwarnings("ignore", message=".*torch_dtype.*")
                
                # 按照ModelScope标准加载方式
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    cache_dir=None,  # 使用默认缓存目录
                )
            
            # 再次清理内存
            self._cleanup_memory()
            
            # 加载模型（内存密集型操作）
            logger.info("正在加载 AutoModelForVision2Seq...")
            logger.info(f"加载参数: {load_kwargs}")
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                # 按照ModelScope标准加载方式
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    cache_dir=None,  # 使用默认缓存目录
                    **load_kwargs
                )
            
            # CPU模式需要手动移动到设备
            if self.device == "cpu" and "device_map" not in load_kwargs:
                logger.info("将模型移动到CPU...")
                self.model = self.model.to(self.device)
            
            # 设置模型为评估模式
            self.model.eval()
            
            # 最终内存清理
            self._cleanup_memory()
            
            # 显示成功信息
            actual_dtype = next(self.model.parameters()).dtype if self.model else "unknown"
            logger.info(f"✅ 成功加载 SmolVLM2 模型")
            logger.info(f"📝 模型类型: SmolVLM2-2.2B")
            logger.info(f"📝 设备: {self.device}")
            logger.info(f"📝 数据类型: {actual_dtype}")
            logger.info(f"📝 上下文长度: {getattr(self.processor.tokenizer, 'model_max_length', 'unknown')}")
            
            # 显示内存使用情况
            if CUDA_AVAILABLE and self.device == "cuda":
                allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
                cached_memory = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"📝 GPU内存使用: 已分配{allocated_memory:.1f}GB, 缓存{cached_memory:.1f}GB")
            
        except Exception as e:
            error_msg = f"使用 transformers 加载 SmolVLM2 模型失败: {e}"
            logger.error(error_msg)
            logger.error(f"错误类型: {type(e)}")
            
            # 提供详细的错误信息和解决方案
            if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                error_msg += "\n\n💡 内存不足解决方案:"
                error_msg += "\n1. 使用 CPU 模式：--device cpu"
                error_msg += "\n2. 关闭其他占用内存的应用"
                error_msg += "\n3. 增加系统虚拟内存大小"
                error_msg += "\n4. 尝试重启系统释放内存"
            elif "load_in_8bit" in str(e).lower() or "bitsandbytes" in str(e).lower() or "SCB" in str(e):
                error_msg += "\n\n💡 量化库问题解决方案:"
                if "SCB" in str(e):
                    error_msg += "\n⚠️  检测到SCB属性错误，这是bitsandbytes版本兼容性问题"
                    error_msg += "\n1. 卸载当前版本: pip uninstall bitsandbytes"
                    error_msg += "\n2. 安装兼容版本: pip install bitsandbytes>=0.43.0"
                    error_msg += "\n3. 重启Python环境后重试"
                    error_msg += "\n4. 或使用 CPU 模式：--device cpu"
                    error_msg += "\n5. 或禁用量化：--quantization none"
                else:
                    error_msg += "\n1. 安装 bitsandbytes: pip install bitsandbytes>=0.43.0"
                    error_msg += "\n2. 或使用 CPU 模式：--device cpu"
            elif "flash_attention" in str(e).lower():
                error_msg += "\n\n💡 FlashAttention 问题解决方案:"
                error_msg += "\n1. 安装 flash-attn: pip install flash-attn"
                error_msg += "\n2. 或将在代码中禁用 flash_attention"
            elif "transformers" in str(e).lower():
                error_msg += "\n\n💡 Transformers 问题解决方案:"
                error_msg += "\n1. 更新 transformers: pip install transformers>=4.47.0 --upgrade"
                error_msg += "\n2. 检查模型文件完整性"
                error_msg += "\n3. 检查网络连接（如需下载模型）"
            
            # 清理内存后再抛出异常
            self._cleanup_memory()
            raise RuntimeError(error_msg)
            
    def _get_optimal_load_config(self) -> Dict[str, Any]:
        """根据系统资源获取最优加载配置"""
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if self.device == "cuda" and CUDA_AVAILABLE:
            try:
                # 获取GPU显存信息
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                gpu_free_gb = gpu_free_memory / 1024**3
                
                logger.info(f"显存状态: 总量{gpu_memory_gb:.1f}GB, 可用{gpu_free_gb:.1f}GB")
                
                # 只有在需要量化时才检查量化库支持
                bitsandbytes_available = False
                if self.quantization_mode in ["4bit", "8bit"]:
                    bitsandbytes_available = self._check_bitsandbytes()
                    # 如果检测到SCB问题，禁用量化
                    if not self._check_bitsandbytes_compatibility():
                        logger.warning("检测到bitsandbytes兼容性问题，禁用量化")
                        bitsandbytes_available = False
                
                # 处理强制量化模式
                if self.quantization_mode == "4bit":
                    if bitsandbytes_available:
                        logger.info("强制使用4bit量化模式")
                        quantization_config = self._get_4bit_config()
                        if quantization_config:
                            load_kwargs.update({
                                "dtype": torch.float16,
                                "device_map": "auto",
                                "quantization_config": quantization_config,
                                "max_memory": {0: f"{int(gpu_free_gb * 0.7)}GB", "cpu": "6GB"}
                            })
                            return load_kwargs
                    logger.error("4bit量化不可用，请安装: pip install bitsandbytes")
                    raise RuntimeError("4bit量化不可用")
                elif self.quantization_mode == "8bit":
                    if bitsandbytes_available:
                        logger.info("强制使用8bit量化模式")
                        quantization_config = self._get_8bit_config()
                        if quantization_config:
                            load_kwargs.update({
                                "dtype": torch.float16,
                                "device_map": "auto",
                                "quantization_config": quantization_config,
                                "max_memory": {0: f"{int(gpu_free_gb * 0.8)}GB", "cpu": "4GB"}
                            })
                            return load_kwargs
                    logger.error("8bit量化不可用，请安装: pip install bitsandbytes")
                    raise RuntimeError("8bit量化不可用")
                elif self.quantization_mode == "none":
                    logger.info("强制禁用量化，使用原始精度")
                    load_kwargs.update({
                        "dtype": torch.bfloat16 if gpu_memory_gb >= 12 else torch.float16,
                        "device_map": "auto"
                    })
                    return load_kwargs
                
                # 根据显存大小选择加载策略（全量精度加载，无量化）
                if gpu_memory_gb < 4:
                    # 极小显存: 强制CPU模式
                    logger.warning("显存严重不足(<4GB)，强制使用CPU模式")
                    self.device = "cpu"
                    load_kwargs.update({
                        "dtype": torch.float32,
                        "device_map": "cpu",
                        "use_safetensors": True
                    })
                elif gpu_memory_gb < 6:
                    # 小显存: 使用float16精度
                    logger.info("显存较小(<6GB)，使用float16精度全量加载")
                    load_kwargs.update({
                        "dtype": torch.float16,
                        "device_map": "auto",
                        "max_memory": {0: f"{int(gpu_free_gb * 0.8)}GB", "cpu": "8GB"}
                    })
                elif gpu_memory_gb < 8:
                    # 中小显存: 使用float16精度
                    logger.info("中小显存(6-8GB)，使用float16精度全量加载")
                    load_kwargs.update({
                        "dtype": torch.float16,
                        "device_map": "auto",
                        "max_memory": {0: f"{int(gpu_free_gb * 0.85)}GB", "cpu": "4GB"}
                    })
                elif gpu_memory_gb < 12:
                    # 中等显存: 使用float16精度
                    logger.info("中等显存(8-12GB)，使用float16精度全量加载")
                    load_kwargs.update({
                        "dtype": torch.float16,
                        "device_map": "auto",
                        "max_memory": {0: f"{int(gpu_free_gb * 0.9)}GB", "cpu": "2GB"}
                    })
                else:
                    # 充足显存: 使用bfloat16精度
                    logger.info("充足显存(>=12GB)，使用bfloat16精度全量加载")
                    load_kwargs.update({
                        "dtype": torch.bfloat16,
                        "device_map": "auto"
                    })
                
                # flash_attention 是可选的（仅在非量化模式下）
                if not load_kwargs.get("quantization_config"):
                    try:
                        import flash_attn
                        load_kwargs["attn_implementation"] = "flash_attention_2"
                        logger.info("启用 FlashAttention2 加速")
                    except ImportError:
                        logger.info("FlashAttention2 不可用，使用默认注意力机制")
                    
            except Exception as e:
                logger.warning(f"获取GPU信息失败: {e}，使用默认GPU配置")
                load_kwargs.update({
                    "dtype": torch.float16,
                    "device_map": "auto"
                })
        else:
            # CPU模式优化
            logger.info("使用CPU模式，启用内存优化")
            load_kwargs.update({
                "dtype": torch.float32,
                "device_map": "cpu",
                "use_safetensors": True
            })
        
        return load_kwargs
        
    def _get_4bit_config(self):
        """获取4bit量化配置（针对SmolVLM2优化）"""
        try:
            from transformers import BitsAndBytesConfig
            # 针对SmolVLM2的4bit量化配置，增强兼容性
            # 添加更多兼容性配置以解决SCB属性错误
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                # 增加稳定性选项
                bnb_4bit_quant_storage=torch.uint8,
                # 添加更多兼容性选项
                bnb_4bit_skip_modules=["lm_head"],
            )
            logger.info("✅ 4bit量化配置创建成功")
            return config
        except ImportError:
            logger.error("BitsAndBytesConfig 不可用，请更新 transformers")
            return None
        except Exception as e:
            error_str = str(e)
            if "SCB" in error_str:
                logger.error(f"4bit量化配置创建失败（SCB错误）: {e}")
                logger.error("这是bitsandbytes版本兼容性问题，请更新到>=0.41.0")
            else:
                logger.error(f"4bit量化配置创建失败: {e}")
            return None
    
    def _get_8bit_config(self):
        """获取8bit量化配置（推荐用于SmolVLM2）"""
        try:
            from transformers import BitsAndBytesConfig
            # 8bit量化是SmolVLM2的推荐配置，兼容性更好
            # 添加更多兼容性配置以解决SCB属性错误
            config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=False,  # 禁用CPU卸载提高稳定性
                llm_int8_has_fp16_weight=True,
                # 添加更多兼容性选项
                llm_int8_skip_modules=["lm_head"],
                # 解决SCB属性错误的额外配置
                llm_int8_threshold=6.0,
            )
            logger.info("✅ 8bit量化配置创建成功")
            return config
        except ImportError:
            logger.error("BitsAndBytesConfig 不可用，请更新 transformers")
            return None
        except Exception as e:
            error_str = str(e)
            if "SCB" in error_str:
                logger.error(f"8bit量化配置创建失败（SCB错误）: {e}")
                logger.error("这是bitsandbytes版本兼容性问题，请更新到>=0.41.0")
            else:
                logger.error(f"8bit量化配置创建失败: {e}")
            return None
    
    def _check_bitsandbytes_compatibility(self) -> bool:
        """检查 bitsandbytes 量化库是否可用且兼容"""
        try:
            import bitsandbytes as bnb
            bnb_version = bnb.__version__
            logger.info(f"bitsandbytes 量化库可用，版本: {bnb_version}")
            
            # 检查版本兼容性
            import torch
            
            torch_version = torch.__version__.split('+')[0]  # 去除CUDA后缀
            
            # 检查PyTorch与bitsandbytes的兼容性
            if version is not None:
                try:
                    if version.parse(torch_version) >= version.parse("2.0.0"):
                        if version.parse(bnb_version) < version.parse("0.43.0"):
                            logger.warning(f"bitsandbytes版本{bnb_version}可能与PyTorch {torch_version}不兼容")
                            logger.warning("建议更新: pip install bitsandbytes>=0.43.0 --upgrade")
                            return False
                except Exception as ver_e:
                    logger.warning(f"版本检查失败: {ver_e}，继续进行功能性测试")
            else:
                logger.warning("无法进行版本兼容性检查，继续进行功能性测试")
            
            # 进行简单的功能性测试
            try:
                from transformers import BitsAndBytesConfig
                # 尝试创建一个简单的配置测试兼容性
                test_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                logger.info("✅ bitsandbytes功能性测试通过")
                return True
                
            except Exception as test_e:
                logger.error(f"❌ bitsandbytes功能性测试失败: {test_e}")
                if "SCB" in str(test_e):
                    logger.error("检测到SCB属性错误，这通常是版本兼容性问题")
                    logger.error("请尝试以下解决方案：")
                    logger.error("1. pip uninstall bitsandbytes")
                    logger.error("2. pip install bitsandbytes>=0.43.0")
                    logger.error("3. 或使用CPU模式: --device cpu")
                return False
                
        except ImportError:
            logger.warning("bitsandbytes 量化库不可用，无法4bit/8bit量化")
            return False
        except Exception as e:
            logger.error(f"bitsandbytes检查失败: {e}")
            return False
    
    def _check_bitsandbytes(self) -> bool:
        """检查 bitsandbytes 量化库是否可用"""
        try:
            import bitsandbytes as bnb
            logger.info(f"bitsandbytes 量化库可用，版本: {bnb.__version__}")
            return True
        except ImportError:
            logger.warning("bitsandbytes 量化库不可用，无法4bit/8bit量化")
            return False
        except Exception as e:
            logger.error(f"bitsandbytes检查失败: {e}")
            return False
    
    
    def _preprocess_image_standard(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """按照ModelScope标准预处理图像"""
        # 处理图像输入
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
            logger.info(f"从文件加载图像: {image}")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert('RGB')
        else:
            pil_image = image.convert('RGB')
        
        # 按照ModelScope推荐尺寸：384x384
        max_size = 384
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = tuple(int(dim * ratio) for dim in pil_image.size)
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"图像已缩放至: {new_size}")
        
        return pil_image
    
    def _cleanup_memory(self):
        """清理内存和显存"""
        try:
            import gc
            gc.collect()
            
            if CUDA_AVAILABLE and self.device == "cuda":
                torch.cuda.empty_cache()
                # 强制同步显存操作
                torch.cuda.synchronize()
                
        except Exception as e:
            logger.warning(f"内存清理失败: {e}")
    
    def analyze_image(self, 
                     image: Union[str, Image.Image, np.ndarray],
                     prompt: str = "请详细分析这张古董文物图片，包括文物类型、年代、材质、工艺特点、真伪评估、保存状况和历史价值。请确保回答完全使用中文，不要使用英文。") -> str:
        """
        分析图像并生成描述
        
        Args:
            image: 输入图像
            prompt: 分析提示词
            
        Returns:
            分析结果文本
            
        Raises:
            RuntimeError: 当模型未正确加载时
        """
        if not self.model or not self.processor:
            raise RuntimeError("模型或处理器未正确加载，无法进行图像分析")
        
        return self._analyze_with_model(image, prompt)
    
    def analyze_image_stream(self, 
                           image: Union[str, Image.Image, np.ndarray],
                           prompt: str = "请详细分析这张古董文物图片，包括文物类型、年代、材质、工艺特点、真伪评估、保存状况和历史价值。请确保回答完全使用中文，不要使用英文。"):
        """
        流式分析图像并生成描述
        
        Args:
            image: 输入图像
            prompt: 分析提示词
            
        Yields:
            分析结果文本片段
            
        Raises:
            RuntimeError: 当模型未正确加载时
        """
        if not self.model or not self.processor:
            raise RuntimeError("模型或处理器未正确加载，无法进行图像分析")
        
        yield from self._analyze_with_model_stream(image, prompt)
    
    def _analyze_with_model(self, image: Union[str, Image.Image, np.ndarray], prompt: str) -> str:
        """使用 SmolVLM2 模型分析图像"""
        try:
            # 推理前清理内存
            self._cleanup_memory()
            
            # 按照ModelScope标准预处理图像
            pil_image = self._preprocess_image_standard(image)
            
            # 按照ModelScope标准构建消息格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 应用聊天模板
            text_input = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )
            
            # 按照ModelScope标准处理输入
            inputs = self.processor(
                text=text_input,
                images=pil_image,
                return_tensors="pt",
                padding=True,
                truncation=False
            ).to(self.device)
            
            # 生成配置
            generation_config = {
                "max_new_tokens": min(self.max_tokens, 512),
                "temperature": self.temperature,
                "do_sample": True if self.temperature > 0 else False,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": True,
                "repetition_penalty": 1.1,
            }
            
            # 生成回复
            with torch.no_grad():
                if CUDA_AVAILABLE and self.device == "cuda":
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        generated_ids = self.model.generate(**inputs, **generation_config)
                else:
                    generated_ids = self.model.generate(**inputs, **generation_config)
            
            # 清理输入
            del inputs
            self._cleanup_memory()
            
            # 解码生成的文本
            generated_text = self.processor.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            # 清理生成的ids
            del generated_ids
            self._cleanup_memory()
            
            # 提取回复部分（去除输入prompt）
            if "assistant" in generated_text.lower():
                parts = generated_text.split("assistant")
                if len(parts) > 1:
                    response = parts[-1].strip()
                    return response if response else "无法生成有效回复"
            
            # 如果没有找到assistant标记，尝试其他方式提取
            response = generated_text.replace(text_input, "").strip()
            return response if response else "无法生成有效回复"
            
        except Exception as e:
            logger.error(f"SmolVLM2 图像分析失败: {e}")
            self._cleanup_memory()
            return f"分析失败: {str(e)}"
    
    def _analyze_with_model_stream(self, image: Union[str, Image.Image, np.ndarray], prompt: str):
        """使用 SmolVLM2 模型流式分析图像"""
        try:
            # 推理前清理内存
            self._cleanup_memory()
            
            # 按照ModelScope标准预处理图像
            pil_image = self._preprocess_image_standard(image)
            
            # 按照ModelScope标准构建消息格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 应用聊天模板
            text_input = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )
            
            # 按照ModelScope标准处理输入
            inputs = self.processor(
                text=text_input,
                images=pil_image,
                return_tensors="pt",
                padding=True,
                truncation=False
            ).to(self.device)
            
            # 流式生成配置
            generation_config = {
                "max_new_tokens": min(self.max_tokens, 512),
                "temperature": self.temperature,
                "do_sample": True if self.temperature > 0 else False,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": True,
                "repetition_penalty": 1.1,
            }
            
            # 流式生成
            with torch.no_grad():
                if CUDA_AVAILABLE and self.device == "cuda":
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        # 使用流式生成
                        for new_token_id in self._generate_stream(**inputs, **generation_config):
                            if new_token_id is not None:
                                token_text = self.processor.tokenizer.decode([new_token_id], skip_special_tokens=True)
                                yield token_text
                else:
                    for new_token_id in self._generate_stream(**inputs, **generation_config):
                        if new_token_id is not None:
                            token_text = self.processor.tokenizer.decode([new_token_id], skip_special_tokens=True)
                            yield token_text
            
            # 清理内存
            del inputs
            self._cleanup_memory()
            
        except Exception as e:
            logger.error(f"SmolVLM2 流式分析失败: {e}")
            self._cleanup_memory()
            yield f"错误: {str(e)}"
    
    def _generate_stream(self, **kwargs):
        """流式生成token"""
        try:
            # 获取输入长度
            input_length = kwargs['input_ids'].shape[1]
            
            # 逐步生成
            for step in range(kwargs.get('max_new_tokens', 512)):
                # 生成下一个token
                with torch.no_grad():
                    outputs = self.model(**{k: v for k, v in kwargs.items() if k not in ['max_new_tokens']})
                    next_token_logits = outputs.logits[0, -1, :]
                    
                    # 应用温度
                    if kwargs.get('temperature', 1.0) != 1.0:
                        next_token_logits = next_token_logits / kwargs['temperature']
                    
                    # 采样下一个token
                    if kwargs.get('do_sample', False):
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # 检查是否结束
                    if next_token.item() == kwargs.get('eos_token_id'):
                        break
                    
                    # 更新输入
                    kwargs['input_ids'] = torch.cat([kwargs['input_ids'], next_token.unsqueeze(0)], dim=1)
                    if 'attention_mask' in kwargs:
                        kwargs['attention_mask'] = torch.cat([
                            kwargs['attention_mask'], 
                            torch.ones((1, 1), device=kwargs['attention_mask'].device)
                        ], dim=1)
                    
                    yield next_token.item()
                    
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield None
    
    def generate_appraisal_report(self,
                                image: Union[str, Image.Image, np.ndarray],
                                user_query: str = "") -> Dict[str, Any]:
        """
        生成古董鉴定报告
        
        Args:
            image: 古董图像
            user_query: 用户特别关注的问题
            
        Returns:
            鉴定报告字典
        """
        try:
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
            analysis_text = self.analyze_image(image, prompt)
            
            # 构建报告
            report = {
                "model_name": "SmolVLM2-256M",
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
                "model_name": "SmolVLM2-256M",
                "timestamp": self._get_timestamp()
            }
    
    def chat_about_antique(self,
                          image: Union[str, Image.Image, np.ndarray],
                          message: str) -> str:
        """
        关于古董的对话
        
        Args:
            image: 古董图像
            message: 用户消息
            
        Returns:
            回复文本
        """
        prompt = f"请作为专业的古董鉴定师，根据用户的要求：{message}，对这件古董文物进行详细分析。请用中文提供专业的鉴定报告，包含：\n1. 类别：文物类型和名称\n2. 朝代：年代判断和历史时期\n3. 材质：材质成分和制作工艺\n4. 常见外观介绍：典型外观特征和装饰风格\n5. 古代主要用途：实际用途和社会功能\n6. 工艺特点：制作技术和艺术特色\n7. 保存状况：当前状态评估\n8. 历史文化价值：文化意义和学术价值\n\n请直接提供分析结果，不要重复用户的问题。注意：本报告仅供学术研究使用，不提供市场价值评估。"
        return self.analyze_image(image, prompt)
    
    def text_chat(self, message: str) -> str:
        """
        纯文本对话（不需要图片）
        注意：SmolVLM2主要是视觉-语言模型，对于纯文本对话提供基础支持
        
        Args:
            message: 用户消息
            
        Returns:
            回复文本
        """
        try:
            # 推理前清理内存
            self._cleanup_memory()
            
            # 针对中文问题提供更好的回复
            # 检查是否是常见的中文问题类型
            if any(keyword in message for keyword in ["青花瓷", "瓷器", "古董", "文物", "艺术品"]):
                return self._handle_cultural_question(message)
            elif any(keyword in message for keyword in ["介绍", "是什么", "什么是", "解释"]):
                return self._handle_explanation_question(message)
            elif any(keyword in message for keyword in ["你好", "您好", "hello", "hi"]):
                return "您好！我是SmolVLM2，一个视觉-语言AI助手。我特别擅长分析图片和回答与图片相关的问题。如果您有图片需要分析，请使用图片对话功能。对于纯文本问题，我会尽力为您提供帮助。"
            elif "自己" in message and "介绍" in message:
                return "我是SmolVLM2，一个多模态AI助手，专门设计用于理解和分析图像内容。我可以：\n\n1. 🖼️ 分析图片内容和细节\n2. 🏺 识别古董文物并提供专业鉴赏\n3. 📝 生成详细的图像分析报告\n4. 💬 回答与图片相关的问题\n\n虽然我也能进行一些文本对话，但我的强项是视觉理解。如果您有图片需要分析，请使用 'image <图片路径> <问题>' 命令获得最佳体验！"
            
            # 对于其他问题，使用简化的生成方式
            return self._generate_simple_response(message)
            
        except Exception as e:
            logger.error(f"SmolVLM2 纯文本对话失败: {e}")
            self._cleanup_memory()
            return "抱歉，我在处理您的问题时遇到了困难。作为视觉-语言模型，我更适合分析图片。如果您有图片相关的问题，请使用图片对话功能。"
    
    def _handle_cultural_question(self, message: str) -> str:
        """处理文化艺术相关问题"""
        if "青花瓷" in message:
            return "青花瓷是中国传统陶瓷工艺的杰出代表，以白瓷为胎，钴料为色，在高温下烧制而成。其特点包括：\n\n🎨 **艺术特色**：\n- 色彩：蓝白相间，清雅脱俗\n- 图案：山水、花鸟、人物等传统题材\n- 工艺：釉下彩绘，永不褪色\n\n📅 **历史发展**：\n- 起源于唐宋，成熟于元代\n- 明清时期达到巅峰\n- 景德镇是主要产地\n\n💎 **收藏价值**：\n- 工艺精湛的古代青花瓷价值极高\n- 现代仿制品也有一定艺术价值\n\n如果您有青花瓷图片需要鉴赏，我可以提供更详细的分析！"
        return "这是一个很有趣的文化艺术问题。如果您有相关的图片，我可以为您提供更详细和准确的分析。"
    
    def _handle_explanation_question(self, message: str) -> str:
        """处理解释类问题"""
        return f"关于您询问的'{message}'，我可以提供一些基本信息。不过，如果您有相关的图片，我能给出更准确和详细的解释。作为视觉-语言模型，我在图像分析方面表现更佳。"
    
    def _generate_simple_response(self, message: str) -> str:
        """生成简单回复"""
        try:
            # 构建更适合的中文prompt
            if any(char in message for char in "你您我他她它"):
                # 中文输入
                prompt = f"问题：{message}\n回答："
            else:
                # 英文输入
                prompt = f"Question: {message}\nAnswer:"
            
            # 使用tokenizer处理
            inputs = self.processor.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 简化的生成配置
            generation_config = {
                "max_new_tokens": 128,
                "temperature": 0.8,
                "do_sample": True,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "repetition_penalty": 1.3,
                "top_p": 0.85,
                "top_k": 40
            }
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_config)
            
            del inputs
            self._cleanup_memory()
            
            generated_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            del generated_ids
            self._cleanup_memory()
            
            # 提取回复
            if "回答：" in generated_text:
                response = generated_text.split("回答：")[-1].strip()
            elif "Answer:" in generated_text:
                response = generated_text.split("Answer:")[-1].strip()
            else:
                response = generated_text.replace(prompt, "").strip()
            
            # 清理和验证回复
            if response and len(response.strip()) > 3 and not response.startswith("The image"):
                return response[:200] + ("..." if len(response) > 200 else "")
            else:
                return "我理解您的问题，但作为视觉-语言模型，我更擅长分析图片。如果您有相关图片，请使用图片对话功能获得更好的回答。"
                
        except Exception as e:
            logger.error(f"简单回复生成失败: {e}")
            return "抱歉，我在生成回复时遇到了问题。请尝试使用图片对话功能，这是我的强项。"
    
    def chat_with_messages(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """使用标准messages格式进行多模态对话
        
        Args:
            messages: 标准的messages格式，支持以下结构：
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "描述这张图片"},
                            {"type": "image", "image": image_obj}  # PIL Image对象
                        ]
                    }
                ]
            **kwargs: 额外的生成参数
            
        Returns:
            str: 模型的回复
        """
        try:
            # 清理内存
            self._cleanup_memory()
            
            # 提取图片和文本
            images = []
            text_parts = []
            
            for message in messages:
                if message.get("role") == "user":
                    content = message.get("content", [])
                    if isinstance(content, str):
                        # 纯文本消息
                        text_parts.append(content)
                    elif isinstance(content, list):
                        # 多模态消息
                        for item in content:
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "image":
                                image = item.get("image")
                                if image is not None:
                                    images.append(image)
            
            # 合并文本
            prompt = " ".join(text_parts) if text_parts else "描述这张图片"
            
            # 根据输入类型选择处理方式
            if images:
                # 有图片的多模态对话
                return self._process_multimodal_chat(prompt, images[0], **kwargs)
            else:
                # 纯文本对话
                return self._process_text_only_chat(prompt, **kwargs)
                
        except Exception as e:
            logger.error(f"Messages格式对话失败: {e}")
            self._cleanup_memory()
            return f"抱歉，处理您的请求时出现错误: {str(e)}"
    
    def _process_multimodal_chat(self, prompt: str, image: Image.Image, **kwargs) -> str:
        """处理多模态对话（文本+图片）"""
        try:
            # 使用processor的apply_chat_template方法
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 应用聊天模板
            prompt_text = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            
            # 处理输入
            inputs = self.processor(
                text=prompt_text,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # 生成配置
            generation_config = {
                "max_new_tokens": kwargs.get("max_new_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "do_sample": True,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "repetition_penalty": 1.2,
                "top_p": 0.9,
                "top_k": 50
            }
            
            # 生成回复
            input_length = inputs["input_ids"].shape[1]  # 保存输入长度
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        generated_ids = self.model.generate(**inputs, **generation_config)
                else:
                    generated_ids = self.model.generate(**inputs, **generation_config)
            
            # 清理输入
            del inputs
            self._cleanup_memory()
            
            # 解码回复
            generated_text = self.processor.decode(
                generated_ids[0][input_length:], 
                skip_special_tokens=True
            )
            
            del generated_ids
            self._cleanup_memory()
            
            return generated_text.strip() if generated_text.strip() else "我无法为这张图片生成描述。"
            
        except Exception as e:
            logger.error(f"多模态对话处理失败: {e}")
            self._cleanup_memory()
            return f"处理图片对话时出现错误: {str(e)}"
    
    def _process_text_only_chat(self, prompt: str, **kwargs) -> str:
        """处理纯文本对话"""
        try:
            # 使用messages格式进行纯文本对话
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            # 应用聊天模板
            prompt_text = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            
            # 处理输入
            inputs = self.processor.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 生成配置
            generation_config = {
                "max_new_tokens": kwargs.get("max_new_tokens", 256),
                "temperature": kwargs.get("temperature", 0.8),
                "do_sample": True,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "repetition_penalty": 1.3,
                "top_p": 0.85,
                "top_k": 40
            }
            
            # 生成回复
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_config)
            
            # 清理输入
            del inputs
            self._cleanup_memory()
            
            # 解码回复
            generated_text = self.processor.tokenizer.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            del generated_ids
            self._cleanup_memory()
            
            # 提取助手回复
            if "assistant" in generated_text.lower():
                parts = generated_text.split("assistant")
                if len(parts) > 1:
                    response = parts[-1].strip()
                    return response if response else "我无法生成合适的回复。"
            
            # 移除原始prompt
            response = generated_text.replace(prompt_text, "").strip()
            return response if response else "我无法生成合适的回复。"
            
        except Exception as e:
            logger.error(f"纯文本对话处理失败: {e}")
            self._cleanup_memory()
            return f"处理文本对话时出现错误: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "model_name": "SmolVLM2-256M",
            "model_path": self.model_path,
            "device": self.device,
            "transformers_version": self._get_transformers_version(),
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "cuda_available": CUDA_AVAILABLE,
            "loading_method": "AutoModelForVision2Seq",
            "model_type": "Idefics3ForConditionalGeneration",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        if self.processor:
            info.update({
                "tokenizer_max_length": getattr(self.processor.tokenizer, 'model_max_length', 'unknown'),
                "vocab_size": getattr(self.processor.tokenizer, 'vocab_size', 'unknown')
            })
        
        return info
    
    def _get_transformers_version(self) -> str:
        """获取 transformers 版本"""
        try:
            import transformers
            return transformers.__version__
        except:
            return "unknown"
    
    def cleanup(self):
        """清理资源"""
        try:
            logger.info("开始清理SmolVLM2模型资源...")
            
            if self.model:
                # 先将模型移动到CPU释放显存
                if hasattr(self.model, 'cpu'):
                    self.model = self.model.cpu()
                
                # 清除模型占用的内存
                del self.model
                self.model = None
                logger.info("模型资源已清理")
            
            if self.processor:
                del self.processor
                self.processor = None
                logger.info("处理器资源已清理")
            
            # 深度清理CUDA缓存
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                logger.info("CUDA缓存已清理")
            
            # 深度清理Python垃圾回收
            import gc
            for _ in range(3):  # 多次回收以确保彻底清理
                collected = gc.collect()
                if collected == 0:
                    break
            
            logger.info("✅ SmolVLM2 模型资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().isoformat()


# 便捷函数
def create_smolvlm2_client(device: Optional[str] = None, 
                          max_tokens: int = 512,
                          temperature: float = 0.7,
                          quantization_mode: Optional[str] = None) -> SmolVLM2Client:
    """
    创建 SmolVLM2 客户端的便捷函数
    
    Args:
        device: 设备类型，None 为自动检测
        max_tokens: 最大生成token数
        temperature: 生成温度
        quantization_mode: 量化模式 (auto/4bit/8bit/none)
        
    Returns:
        SmolVLM2 客户端实例
    """
    if device is None:
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        logger.info(f"自动选择设备: {device}")
    
    return SmolVLM2Client(
        device=device,
        max_tokens=max_tokens,
        temperature=temperature,
        quantization_mode=quantization_mode
    )