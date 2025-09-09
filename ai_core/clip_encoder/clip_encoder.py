"""
CLIP 编码器实现
基于 OpenAI CLIP 模型进行图像和文本的多模态编码
"""

import os
import sys
import torch
import clip
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Optional

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.logger_config import get_ai_logger

# 配置CLIP编码器模块日志
logger = get_ai_logger('clip_encoder')


class CLIPEncoder:
    """CLIP 多模态编码器"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        初始化 CLIP 编码器
        
        Args:
            model_name: CLIP 模型名称，默认使用 ViT-B/32
            device: 设备类型，默认自动选择（auto/cuda/cpu）
        """
        # 初始化文本特征缓存
        self._text_cache = {}
        self._max_cache_size = 1000
        
        self.model_name = model_name
        
        # GPU设备智能检测和配置
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
                logger.info(f"CLIP检测到GPU: {gpu_name}, 显存: {gpu_memory:.1f}GB")
                logger.info(f"CLIP将使用GPU加速图像编码")
            else:
                self.device = "cpu"
                logger.info("CLIP未检测到可用GPU，将使用CPU模式")
        else:
            self.device = device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CLIP指定使用GPU但未检测到CUDA，回退到CPU模式")
                self.device = "cpu"
        
        try:
            logger.info(f"正在加载 CLIP 模型: {model_name}")
            # 尝试加载本地缓存的模型，如果失败则使用模拟模式
            try:
                self.model, self.preprocess = clip.load(model_name, device=self.device, download_root=None)
                logger.info(f"CLIP 模型加载成功，设备: {self.device}")
                self.mock_mode = False
            except Exception as network_error:
                logger.warning(f"CLIP模型网络加载失败: {network_error}")
                logger.info("尝试使用模拟模式...")
                self._init_mock_mode()
        except Exception as e:
            logger.error(f"CLIP 模型初始化失败: {e}")
            self._init_mock_mode()
    
    def _init_mock_mode(self):
        """初始化模拟模式"""
        self.mock_mode = True
        self.model = None
        self.preprocess = None
        self.feature_dim = 512  # CLIP的标准维度
        logger.warning("警告：正在使用CLIP模拟模式，功能可能受限")
    
    def encode_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        编码图像
        
        Args:
            image: 图像路径、PIL Image 对象或 numpy 数组
            
        Returns:
            图像特征向量
        """
        try:
            if hasattr(self, 'mock_mode') and self.mock_mode:
                # 模拟模式：返回随机特征向量
                logger.info("使用模拟模式进行图像编码")
                return np.random.randn(self.feature_dim).astype(np.float32)
            
            # 正常模式：处理不同类型的输入
            if isinstance(image, str):
                # 图像路径
                pil_image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                # numpy 数组
                pil_image = Image.fromarray(image).convert('RGB')
            elif isinstance(image, Image.Image):
                # PIL Image
                pil_image = image.convert('RGB')
            else:
                raise ValueError(f"不支持的图像类型: {type(image)}")
            
            # 预处理图像
            processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # 编码
            with torch.no_grad():
                image_features = self.model.encode_image(processed_image)
                image_features = image_features.cpu().numpy()
            
            return image_features[0]  # 返回第一个（也是唯一的）特征向量
            
        except Exception as e:
            logger.error(f"图像编码失败: {e}")
            raise
    
    def encode_text(self, text: Union[str, List[str]], use_cache: bool = True) -> np.ndarray:
        """
        编码文本（带缓存优化）
        
        Args:
            text: 单个文本字符串或文本列表
            use_cache: 是否使用缓存
            
        Returns:
            文本特征向量
        """
        try:
            if hasattr(self, 'mock_mode') and self.mock_mode:
                # 模拟模式：返回随机特征向量
                logger.info("使用模拟模式进行文本编码")
                if isinstance(text, str):
                    return np.random.randn(self.feature_dim).astype(np.float32)
                else:
                    return np.random.randn(len(text), self.feature_dim).astype(np.float32)
            
            # 处理单个文本的缓存
            if isinstance(text, str):
                if use_cache and text in self._text_cache:
                    logger.debug(f"从缓存获取文本特征: {text[:50]}...")
                    return self._text_cache[text]
                
                # 编码单个文本
                text_tokens = clip.tokenize([text]).to(self.device)
                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                    text_features = text_features.cpu().numpy()[0]
                
                # 缓存结果
                if use_cache:
                    self._cache_text_feature(text, text_features)
                
                return text_features
            
            # 处理文本列表
            else:
                cached_features = []
                uncached_texts = []
                uncached_indices = []
                
                # 检查缓存
                if use_cache:
                    for i, t in enumerate(text):
                        if t in self._text_cache:
                            cached_features.append((i, self._text_cache[t]))
                        else:
                            uncached_texts.append(t)
                            uncached_indices.append(i)
                else:
                    uncached_texts = text
                    uncached_indices = list(range(len(text)))
                
                # 编码未缓存的文本
                if uncached_texts:
                    text_tokens = clip.tokenize(uncached_texts).to(self.device)
                    with torch.no_grad():
                        uncached_features = self.model.encode_text(text_tokens)
                        uncached_features = uncached_features.cpu().numpy()
                    
                    # 缓存新特征
                    if use_cache:
                        for t, f in zip(uncached_texts, uncached_features):
                            self._cache_text_feature(t, f)
                else:
                    uncached_features = np.array([])
                
                # 合并结果
                all_features = [None] * len(text)
                
                # 填入缓存的特征
                for i, feature in cached_features:
                    all_features[i] = feature
                
                # 填入新编码的特征
                for i, feature in zip(uncached_indices, uncached_features):
                    all_features[i] = feature
                
                return np.array(all_features)
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise
    
    def compute_similarity(self, image: Union[str, Image.Image, np.ndarray], 
                          text: Union[str, List[str]]) -> np.ndarray:
        """
        计算图像和文本之间的相似度
        
        Args:
            image: 图像输入
            text: 文本输入
            
        Returns:
            相似度分数
        """
        try:
            # 编码图像和文本
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)
            
            # 归一化特征向量
            image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
            text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
            
            # 计算余弦相似度
            similarity = np.dot(image_features, text_features.T)
            
            return similarity
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            raise
    
    def batch_encode_images(self, images: List[Union[str, Image.Image, np.ndarray]], batch_size: int = 8) -> np.ndarray:
        """
        批量编码图像（优化版本）
        
        Args:
            images: 图像列表
            batch_size: 批处理大小
            
        Returns:
            图像特征向量数组
        """
        try:
            if hasattr(self, 'mock_mode') and self.mock_mode:
                # 模拟模式：返回随机特征向量
                return np.random.randn(len(images), self.feature_dim).astype(np.float32)
            
            features = []
            # 分批处理以优化内存使用
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_features = []
                
                for image in batch_images:
                    feature = self.encode_image(image)
                    batch_features.append(feature)
                
                features.extend(batch_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"批量图像编码失败: {e}")
            raise
    
    def batch_encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            
        Returns:
            文本特征向量数组
        """
        try:
            return self.encode_text(texts)
            
        except Exception as e:
            logger.error(f"批量文本编码失败: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if hasattr(self, 'mock_mode') and self.mock_mode:
            return {
                "model_name": self.model_name,
                "device": self.device,
                "feature_dim": self.feature_dim,
                "max_text_length": 77,  # CLIP标准最大长度
                "mode": "mock"
            }
        else:
            return {
                "model_name": self.model_name,
                "device": self.device,
                "feature_dim": self.model.visual.output_dim,
                "max_text_length": self.model.context_length,
                "mode": "normal",
                "cache_size": len(self._text_cache),
                "max_cache_size": self._max_cache_size
            }
    
    def _cache_text_feature(self, text: str, feature: np.ndarray):
        """缓存文本特征"""
        # 管理缓存大小
        if len(self._text_cache) >= self._max_cache_size:
            # 删除最旧的缓存项（简单FIFO策略）
            oldest_key = next(iter(self._text_cache))
            del self._text_cache[oldest_key]
            logger.debug(f"缓存已满，删除最旧项: {oldest_key[:30]}...")
        
        self._text_cache[text] = feature
        logger.debug(f"缓存文本特征: {text[:30]}..., 缓存大小: {len(self._text_cache)}")
    
    def clear_cache(self):
        """清空文本特征缓存"""
        cache_size = len(self._text_cache)
        self._text_cache.clear()
        logger.info(f"已清空文本特征缓存，释放了 {cache_size} 个缓存项")
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        return {
            "cache_size": len(self._text_cache),
            "max_cache_size": self._max_cache_size,
            "cache_usage": len(self._text_cache) / self._max_cache_size * 100
        }
