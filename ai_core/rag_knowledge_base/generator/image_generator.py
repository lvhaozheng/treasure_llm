#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文物图片生成器
专门负责文物图片的生成和下载功能

功能:
1. 生成文物图像的提示词
2. 调用豆包模型生成文物图像
3. 从URL下载图片到本地
4. 图片文件管理和存储

作者: AI Assistant
创建时间: 2025-01-15
"""

import os
import logging
import requests
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from urllib.parse import urlparse
# 尝试导入豆包相关的SDK
try:
    from volcenginesdkarkruntime import Ark
    VOLCENGINE_AVAILABLE = True
except ImportError:
    VOLCENGINE_AVAILABLE = False
    # 如果没有安装volcengine SDK，使用OpenAI作为备选
    try:
        from openai import OpenAI
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False

# 导入文物信息数据结构
try:
    from ..data_structures import CulturalRelicInfo, Dynasty, RelicCategory
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from data_structures import CulturalRelicInfo, Dynasty, RelicCategory

class ImageGenerator:
    """文物图片生成器类"""
    
    def __init__(self, config=None, logger=None):
        """
        初始化图片生成器
        
        Args:
            config: 配置对象，包含豆包图像生成API配置
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger or self._setup_default_logger()
        
        # 设置豆包图像生成API参数
        if config and hasattr(config, 'model') and hasattr(config.model, 'doubao_image'):
            doubao_config = config.model.doubao_image
            self.doubao_image_api_key = doubao_config.api_key
            self.doubao_image_endpoint = doubao_config.base_url
            self.doubao_image_model = doubao_config.model_name
        else:
            # 使用默认配置或环境变量
            self.doubao_image_api_key = os.getenv('DOUBAO_API_KEY', 'your_doubao_api_key_here')
            self.doubao_image_endpoint = os.getenv('DOUBAO_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
            self.doubao_image_model = os.getenv('DOUBAO_IMAGE_MODEL', 'doubao-seedream-3-0-t2i-250415')
        
        # 初始化HTTP会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CulturalRelicGenerator/1.0'
        })
        
        # 设置输出目录
        self.output_dir = self._setup_output_directory()
        
    def _setup_default_logger(self):
        """设置默认日志记录器"""
        logger = logging.getLogger('ImageGenerator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _setup_output_directory(self) -> str:
        """设置输出目录"""
        # 创建images目录用于存储下载的图片
        base_dir = os.path.dirname(os.path.dirname(__file__))
        images_dir = os.path.join(base_dir, 'images')
        
        try:
            os.makedirs(images_dir, exist_ok=True)
            self.logger.info(f"图片输出目录已创建: {images_dir}")
            return images_dir
        except Exception as e:
            self.logger.error(f"创建图片输出目录失败: {str(e)}")
            # 使用当前目录作为备选
            return os.getcwd()
    
    def generate_image_prompt(self, relic_info: CulturalRelicInfo) -> str:
        """
        生成文物图像的提示词
        
        Args:
            relic_info: 文物信息对象
            
        Returns:
            str: 生成的图像提示词
        """
        try:
            # 根据朝代特色调整提示词
            dynasty_styles = {
                Dynasty.SHANG: "商代风格，古朴厚重，青铜纹饰精美",
                Dynasty.TANG: "唐代风格，雍容华贵，色彩绚丽",
                Dynasty.SONG: "宋代风格，简约雅致，线条流畅",
                Dynasty.MING: "明代风格，精工细作，装饰华丽",
                Dynasty.QING: "清代风格，工艺精湛，富丽堂皇"
            }
            
            style_desc = dynasty_styles.get(relic_info.dynasty, "传统中国古典风格")
            
            # 根据文物类别调整描述
            category_details = {
                RelicCategory.BRONZE: "青铜材质，表面有古朴的铜绿色泽",
                RelicCategory.POTTERY: "陶土材质，质朴的土色调",
                RelicCategory.JADE: "玉石材质，温润光泽",
                RelicCategory.PORCELAIN: "瓷器材质，光滑细腻的表面",
                RelicCategory.LACQUERWARE: "漆器工艺，光亮的漆面"
            }
            
            material_desc = category_details.get(relic_info.category, "精美的传统工艺")
            
            prompt = f"""
一件{relic_info.dynasty.value}时期的{relic_info.name}，{relic_info.category.value}类文物。

外观特征：{relic_info.appearance}

材质工艺：{relic_info.materials}，{relic_info.craftsmanship}

风格要求：{style_desc}，{material_desc}

画面要求：
- 文物居中展示，背景简洁
- 光线柔和，突出文物细节
- 高清晰度，展现工艺精美
- 符合博物馆展品摄影标准
- 体现{relic_info.dynasty.value}时期的历史韵味

注意：图像应真实反映文物特征，避免过度艺术化处理。
            """.strip()
            
            self.logger.info(f"已生成图像提示词: {relic_info.name}")
            return prompt
            
        except Exception as e:
            self.logger.error(f"生成图像提示词失败: {str(e)}")
            return f"一件{relic_info.dynasty.value}的{relic_info.name}，{relic_info.category.value}，传统中国古典风格，博物馆展品质量。"
    
    def generate_image_with_doubao(self, prompt: str) -> Dict[str, Any]:
        """
        使用豆包模型生成图像
        
        Args:
            prompt: 图像生成提示词
            
        Returns:
            Dict[str, Any]: 包含生成结果的字典
        """
        try:
            self.logger.info("开始调用豆包图像生成API")
            
            # 检查是否有可用的SDK
            if not VOLCENGINE_AVAILABLE and not OPENAI_AVAILABLE:
                error_msg = "未安装必要的SDK。请安装 volcengine-python-sdk 或 openai"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "model": self.doubao_image_model
                }
            
            # 优先使用火山引擎SDK
            if VOLCENGINE_AVAILABLE:
                return self._generate_with_volcengine_sdk(prompt)
            elif OPENAI_AVAILABLE:
                return self._generate_with_openai_sdk(prompt)
            
        except Exception as e:
            error_msg = f"调用豆包图像生成API失败: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": self.doubao_image_model
            }
    
    def _generate_with_volcengine_sdk(self, prompt: str) -> Dict[str, Any]:
        """
        使用火山引擎SDK调用豆包图像生成API
        """
        try:
            # 注意：豆包的图像生成API可能需要特殊的调用方式
            # 这里提供一个基本的实现框架，可能需要根据实际API文档调整
            self.logger.warning("豆包图像生成API调用方式可能需要根据官方文档进一步调整")
            
            # 暂时返回一个模拟的失败结果，提示用户检查API配置
            error_msg = "豆包图像生成API需要使用火山引擎的专用接口，当前实现可能不完整。请检查API密钥格式和调用方式。"
            self.logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "model": self.doubao_image_model,
                "suggestion": "请参考火山引擎官方文档配置图像生成API"
            }
            
        except Exception as e:
            error_msg = f"火山引擎SDK调用失败: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": self.doubao_image_model
            }
    
    def _generate_with_openai_sdk(self, prompt: str) -> Dict[str, Any]:
        """
        使用OpenAI SDK调用豆包图像生成API（兼容模式）
        """
        try:
            # 初始化客户端
            client = OpenAI(
                api_key=self.doubao_image_api_key,
                base_url=self.doubao_image_endpoint
            )
            
            # 调用图像生成API
            response = client.images.generate(
                model=self.doubao_image_model,
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="standard",
                style="natural"
            )
            
            if response.data and len(response.data) > 0:
                image_url = response.data[0].url
                self.logger.info(f"图像生成成功，URL: {image_url}")
                
                return {
                    "success": True,
                    "image_url": image_url,
                    "model": self.doubao_image_model,
                    "prompt": prompt
                }
            else:
                error_msg = "API返回空结果"
                self.logger.error(f"图像生成失败: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "model": self.doubao_image_model
                }
                
        except Exception as e:
            error_msg = f"OpenAI SDK调用豆包API失败: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model": self.doubao_image_model,
                "suggestion": "豆包图像生成可能不完全兼容OpenAI接口，建议使用火山引擎官方SDK"
            }
    
    def download_image_from_url(self, image_url: str, entry_id: str) -> Dict[str, Any]:
        """
        从URL下载图片到本地
        
        Args:
            image_url: 图片URL
            entry_id: 数据集条目ID，用于生成文件名
            
        Returns:
            Dict[str, Any]: 包含下载结果的字典
        """
        try:
            self.logger.info(f"开始下载图片: {image_url}")
            
            # 发送HTTP请求下载图片
            response = self.session.get(image_url, timeout=30)
            response.raise_for_status()
            
            # 获取文件扩展名
            parsed_url = urlparse(image_url)
            file_extension = os.path.splitext(parsed_url.path)[1]
            if not file_extension:
                file_extension = '.jpg'  # 默认扩展名
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{entry_id}_{timestamp}{file_extension}"
            
            # 构建完整的文件路径
            file_path = os.path.join(self.output_dir, filename)
            
            # 保存图片文件
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # 计算文件大小
            file_size = len(response.content)
            
            # 验证文件是否成功保存
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                self.logger.info(f"图片下载成功: {filename} ({file_size} bytes)")
                
                return {
                    "success": True,
                    "local_path": file_path,
                    "filename": filename,
                    "file_size": file_size,
                    "url": image_url
                }
            else:
                error_msg = "文件保存失败或文件为空"
                self.logger.error(f"图片下载失败: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "url": image_url
                }
                
        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            self.logger.error(f"图片下载失败: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "url": image_url
            }
        except Exception as e:
            error_msg = f"下载过程中发生错误: {str(e)}"
            self.logger.error(f"图片下载失败: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "url": image_url
            }
    
    def generate_and_download_image(self, relic_info: CulturalRelicInfo, entry_id: str) -> Dict[str, Any]:
        """
        生成并下载文物图片的完整流程
        
        Args:
            relic_info: 文物信息对象
            entry_id: 数据集条目ID
            
        Returns:
            Dict[str, Any]: 包含完整流程结果的字典
        """
        result = {
            "image_generation": {"success": False},
            "image_download": {"success": False}
        }
        
        try:
            # 生成图像提示词
            image_prompt = self.generate_image_prompt(relic_info)
            
            # 生成图像
            image_result = self.generate_image_with_doubao(image_prompt)
            result["image_generation"] = image_result
            result["image_prompt"] = image_prompt
            
            # 如果图像生成成功，则下载图片
            if image_result.get("success") and image_result.get("image_url"):
                download_result = self.download_image_from_url(image_result["image_url"], entry_id)
                result["image_download"] = download_result
            
            return result
            
        except Exception as e:
            error_msg = f"图片生成和下载流程失败: {str(e)}"
            self.logger.error(error_msg)
            result["error"] = error_msg
            return result
    
    def get_output_directory(self) -> str:
        """
        获取图片输出目录
        
        Returns:
            str: 输出目录路径
        """
        return self.output_dir
    
    def cleanup_old_images(self, days: int = 30) -> Dict[str, Any]:
        """
        清理旧的图片文件
        
        Args:
            days: 保留天数，超过此天数的文件将被删除
            
        Returns:
            Dict[str, Any]: 清理结果
        """
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (days * 24 * 60 * 60)
            
            deleted_files = []
            total_size_freed = 0
            
            for filename in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, filename)
                
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    
                    if file_mtime < cutoff_time:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        deleted_files.append(filename)
                        total_size_freed += file_size
            
            self.logger.info(f"清理完成: 删除了 {len(deleted_files)} 个文件，释放空间 {total_size_freed} bytes")
            
            return {
                "success": True,
                "deleted_count": len(deleted_files),
                "deleted_files": deleted_files,
                "size_freed": total_size_freed
            }
            
        except Exception as e:
            error_msg = f"清理旧文件失败: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }