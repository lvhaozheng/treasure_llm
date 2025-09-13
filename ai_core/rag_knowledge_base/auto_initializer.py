#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动初始化模块 - RAG知识库向量数据库自动初始化

功能：
1. 系统启动时自动扫描images和metadata目录
2. 检查向量数据库中是否存在对应数据
3. 自动将缺失的数据插入到向量数据库中
4. 使用CLIP编码器对图片进行编码

作者: AI Assistant
创建时间: 2025-09-13
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 使用ai_core的标准日志系统
try:
    from utils.logger_config import get_logger
    logger = get_logger('auto_initializer')
except ImportError:
    # 如果无法导入标准日志系统，使用简化版本
    import logging
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ai_core.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('auto_initializer')

class RAGAutoInitializer:
    """RAG知识库自动初始化器"""
    
    def __init__(self, images_dir: str = None, metadata_dir: str = None):
        """
        初始化自动初始化器
        
        Args:
            images_dir: 图片目录路径
            metadata_dir: 元数据目录路径
        """
        self.base_dir = Path(__file__).parent
        
        # 设置默认目录路径
        if images_dir is None:
            self.images_dir = self.base_dir / "images"
        else:
            self.images_dir = Path(images_dir)
            
        if metadata_dir is None:
            self.metadata_dir = self.base_dir / "metadata"
        else:
            self.metadata_dir = Path(metadata_dir)
            
        # 初始化组件
        self.clip_encoder = None
        self.milvus_client = None
        self.collection_name = "antique_vectors"
        
        # 初始化各个组件
        self._init_components()
    
    def _init_components(self):
        """初始化各个组件"""
        # 初始化CLIP编码器
        self._init_clip_encoder()
        
        # 初始化Milvus客户端
        self._init_milvus_client()
    
    def _init_clip_encoder(self):
        """初始化CLIP编码器"""
        try:
            from ai_core.clip_encoder import CLIPEncoder
            self.clip_encoder = CLIPEncoder()
            logger.info("CLIP编码器初始化成功")
        except ImportError as e:
            logger.warning(f"无法导入CLIP编码器: {e}")
            logger.info("将使用模拟编码器")
            self.clip_encoder = None
        except Exception as e:
            logger.error(f"CLIP编码器初始化失败: {e}")
            self.clip_encoder = None
    
    def _init_milvus_client(self):
        """初始化Milvus客户端"""
        try:
            from ai_core.vector_db.milvus_client import MilvusClient
            self.milvus_client = MilvusClient(
                collection_name=self.collection_name,
                dim=512  # CLIP特征维度
            )
            logger.info("Milvus客户端初始化成功")
        except ImportError as e:
            logger.warning(f"无法导入Milvus客户端: {e}")
            self.milvus_client = None
        except Exception as e:
            logger.error(f"Milvus客户端初始化失败: {e}")
            self.milvus_client = None
    
    def scan_directories(self) -> Tuple[List[Dict], List[Dict]]:
        """扫描images和metadata目录"""
        logger.info("开始扫描目录...")
        
        # 扫描图片目录
        image_files = []
        if self.images_dir.exists():
            for img_file in self.images_dir.glob("*.jpeg"):
                image_files.append({
                    "file_path": str(img_file),
                    "file_name": img_file.name,
                    "file_id": img_file.stem,  # 不包含扩展名的文件名
                    "modified_time": img_file.stat().st_mtime
                })
            
            # 也扫描其他图片格式
            for ext in ["*.jpg", "*.png", "*.gif", "*.bmp"]:
                for img_file in self.images_dir.glob(ext):
                    image_files.append({
                        "file_path": str(img_file),
                        "file_name": img_file.name,
                        "file_id": img_file.stem,
                        "modified_time": img_file.stat().st_mtime
                    })
        
        logger.info(f"找到 {len(image_files)} 个图片文件")
        
        # 扫描元数据目录
        metadata_files = []
        if self.metadata_dir.exists():
            for meta_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    metadata_files.append({
                        "file_path": str(meta_file),
                        "file_name": meta_file.name,
                        "file_id": meta_file.stem,
                        "metadata": metadata,
                        "modified_time": meta_file.stat().st_mtime
                    })
                except Exception as e:
                    logger.warning(f"读取元数据文件失败 {meta_file}: {e}")
        
        logger.info(f"找到 {len(metadata_files)} 个元数据文件")
        
        return image_files, metadata_files
    
    async def check_existing_data(self, file_ids: List[str]) -> List[str]:
        """检查向量数据库中已存在的数据"""
        if not self.milvus_client:
            logger.warning("Milvus客户端未初始化，无法检查现有数据")
            return []
        
        existing_ids = []
        
        try:
            if hasattr(self.milvus_client, 'collection') and self.milvus_client.collection:
                # 获取所有记录的metadata字段
                query_result = self.milvus_client.collection.query(
                    expr="antique_id >= 0",  # 查询所有记录
                    output_fields=["metadata"]
                )
                
                # 解析metadata中的file_id
                for record in query_result:
                    try:
                        import json
                        metadata = json.loads(record.get('metadata', '{}'))
                        record_file_id = metadata.get('file_id')
                        
                        if record_file_id in file_ids and record_file_id not in existing_ids:
                            existing_ids.append(record_file_id)
                            logger.info(f"数据已存在: {record_file_id}")
                            
                    except (json.JSONDecodeError, AttributeError) as e:
                        continue
                        
            else:
                logger.warning("Milvus collection未初始化，跳过查询")
            
            logger.info(f"检查了 {len(file_ids)} 个文件ID，发现 {len(existing_ids)} 个已存在")
            return existing_ids
            
        except Exception as e:
            logger.error(f"检查现有数据失败: {e}")
            return []
    
    def encode_image(self, image_path: str) -> Optional[List[float]]:
        """编码图片"""
        if self.clip_encoder:
            try:
                return self.clip_encoder.encode_image(image_path)
            except Exception as e:
                logger.error(f"图片编码失败 {image_path}: {e}")
                return None
        else:
            # 使用模拟编码器
            return self._simulate_image_encoding(image_path)
    
    def _simulate_image_encoding(self, image_path: str) -> List[float]:
        """模拟图片编码（用于测试）"""
        import hashlib
        
        # 使用文件路径生成固定的向量
        path_hash = hashlib.md5(image_path.encode('utf-8')).hexdigest()
        
        # 将哈希转换为512维向量
        vector = []
        for i in range(0, len(path_hash), 2):
            hex_val = path_hash[i:i+2]
            vector.append(int(hex_val, 16) / 255.0)  # 归一化到[0,1]
        
        # 扩展到512维
        while len(vector) < 512:
            vector.extend(vector[:min(512-len(vector), len(vector))])
        
        return vector[:512]
    
    def encode_text(self, text: str) -> Optional[List[float]]:
        """编码文本"""
        if self.clip_encoder:
            try:
                return self.clip_encoder.encode_text(text)
            except Exception as e:
                logger.error(f"文本编码失败: {e}")
                return None
        else:
            # 使用模拟编码器
            return self._simulate_text_encoding(text)
    
    def _simulate_text_encoding(self, text: str) -> List[float]:
        """模拟文本编码（用于测试）"""
        import hashlib
        
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # 将哈希转换为512维向量
        vector = []
        for i in range(0, len(text_hash), 2):
            hex_val = text_hash[i:i+2]
            vector.append(int(hex_val, 16) / 255.0)  # 归一化到[0,1]
        
        # 扩展到512维
        while len(vector) < 512:
            vector.extend(vector[:min(512-len(vector), len(vector))])
        
        return vector[:512]
    
    async def insert_missing_data(self, image_files: List[Dict], metadata_files: List[Dict]) -> Dict[str, Any]:
        """插入缺失的数据到向量数据库"""
        if not self.milvus_client:
            logger.warning("Milvus客户端未初始化，无法插入数据")
            return {"success": False, "error": "Milvus客户端未初始化"}
        
        # 创建文件ID到文件信息的映射
        image_map = {img["file_id"]: img for img in image_files}
        metadata_map = {meta["file_id"]: meta for meta in metadata_files}
        
        # 找到匹配的图片和元数据对
        matched_pairs = []
        
        # 对于图片文件，尝试匹配对应的元数据文件
        for img_id, img_info in image_map.items():
            # 图片文件名可能包含重复的时间戳，需要特殊处理
            # 例如: 商朝_青铜器_20250913_154134_20250913_154134.jpeg
            # 对应的元数据: 商朝_青铜器_20250913_154134.json
            
            # 尝试直接匹配
            if img_id in metadata_map:
                matched_pairs.append({
                    "file_id": img_id,
                    "image_info": img_info,
                    "metadata_info": metadata_map[img_id]
                })
            else:
                # 如果直接匹配失败，尝试去掉重复的时间戳部分
                # 分割文件名，查找可能的匹配
                parts = img_id.split('_')
                if len(parts) >= 6:  # 至少包含朝代_类别_日期_时间_日期_时间格式
                    # 尝试构建简化的ID（去掉最后的重复时间戳）
                    simplified_id = '_'.join(parts[:-2])  # 去掉最后两个部分
                    if simplified_id in metadata_map:
                        matched_pairs.append({
                            "file_id": simplified_id,  # 使用简化的ID作为匹配键
                            "image_info": img_info,
                            "metadata_info": metadata_map[simplified_id]
                        })
        
        logger.info(f"找到 {len(matched_pairs)} 对匹配的图片和元数据")
        
        # 检查现有数据
        file_ids = [pair["file_id"] for pair in matched_pairs]
        existing_ids = await self.check_existing_data(file_ids)
        
        # 过滤出需要插入的数据
        pairs_to_insert = [pair for pair in matched_pairs if pair["file_id"] not in existing_ids]
        
        logger.info(f"需要插入 {len(pairs_to_insert)} 条新数据")
        
        if not pairs_to_insert:
            return {"success": True, "inserted_count": 0, "message": "没有新数据需要插入"}
        
        # 准备插入数据
        inserted_count = 0
        failed_count = 0
        
        for i, pair in enumerate(pairs_to_insert):
            # 每处理10个文件后添加小延迟，避免过度占用资源
            if i > 0 and i % 10 == 0:
                await asyncio.sleep(0.1)
                logger.info(f"已处理 {i}/{len(pairs_to_insert)} 个文件...")
            try:
                # 编码图片
                image_vector = self.encode_image(pair["image_info"]["file_path"])
                if image_vector is None:
                    logger.error(f"图片编码失败: {pair['file_id']}")
                    failed_count += 1
                    continue
                
                # 准备元数据文本 - 使用简洁描述
                metadata = pair["metadata_info"]["metadata"]
                
                # 构建简洁的文本描述，避免超过CLIP的77 token限制
                if "description" in metadata:
                    desc = metadata["description"]
                    if isinstance(desc, dict):
                        # 只使用最关键的信息：朝代 + 类别 + 简短名称
                        dynasty = desc.get("dynasty", "")[:50]  # 朝代最多6字符
                        category = desc.get("category", "")[:50]  # 类别最多6字符
                        name = desc.get("name", "")[:50]  # 名称最多10字符
                        
                        parts = [p for p in [dynasty, category, name] if p]
                        text_content = " ".join(parts)
                    else:
                        text_content = str(desc)[:50]  # 直接描述限制30字符
                else:
                    text_content = pair["file_id"][:50]  # 使用文件ID，限制30字符
                
                # 最终确保不超过30字符（约15个中文token）
                if len(text_content) > 50:
                    text_content = text_content[:50]
                
                # 编码文本
                text_vector = self.encode_text(text_content)
                if text_vector is None:
                    logger.error(f"文本编码失败: {pair['file_id']}")
                    failed_count += 1
                    continue
                
                # 插入图片向量
                try:
                    antique_id = hash(pair["file_id"]) % 1000000  # 生成一个数字ID
                    
                    # 插入图片向量
                    self.milvus_client.insert_vectors(
                        antique_ids=[antique_id],
                        vectors=[image_vector],
                        vector_types=["image"],
                        metadata=[{
                            "file_id": pair["file_id"],
                            "file_path": pair["image_info"]["file_path"],
                            "metadata_path": pair["metadata_info"]["file_path"],
                            "type": "image",
                            "inserted_at": datetime.now().isoformat()
                        }]
                    )
                    
                    # 插入文本向量
                    self.milvus_client.insert_vectors(
                        antique_ids=[antique_id],
                        vectors=[text_vector],
                        vector_types=["text"],
                        metadata=[{
                            "file_id": pair["file_id"],
                            "text_content": text_content,
                            "metadata_path": pair["metadata_info"]["file_path"],
                            "type": "text",
                            "inserted_at": datetime.now().isoformat()
                        }]
                    )
                    
                    inserted_count += 1
                    logger.info(f"成功插入数据: {pair['file_id']}")
                    
                except Exception as e:
                    logger.error(f"插入向量数据失败 {pair['file_id']}: {e}")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"处理数据失败 {pair['file_id']}: {e}")
                failed_count += 1
        
        return {
            "success": True,
            "inserted_count": inserted_count,
            "failed_count": failed_count,
            "total_processed": len(pairs_to_insert)
        }
    
    async def initialize(self) -> Dict[str, Any]:
        """执行完整的自动初始化流程"""
        logger.info("开始RAG知识库自动初始化...")
        
        start_time = time.time()
        
        try:
            # 1. 扫描目录
            image_files, metadata_files = self.scan_directories()
            
            if not image_files and not metadata_files:
                logger.info("没有找到任何文件，初始化完成")
                return {"success": True, "message": "没有找到任何文件"}
            
            # 2. 插入缺失的数据
            result = await self.insert_missing_data(image_files, metadata_files)
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"自动初始化完成，耗时: {duration:.2f}秒")
            
            result["duration"] = duration
            return result
            
        except Exception as e:
            logger.error(f"自动初始化失败: {e}")
            return {"success": False, "error": str(e)}

async def main():
    """主函数"""
    initializer = RAGAutoInitializer()
    result = await initializer.initialize()
    
    if result["success"]:
        logger.info("自动初始化成功完成")
        if "inserted_count" in result:
            logger.info(f"插入了 {result['inserted_count']} 条新数据")
        if "failed_count" in result and result["failed_count"] > 0:
            logger.warning(f"失败了 {result['failed_count']} 条数据")
    else:
        logger.error(f"自动初始化失败: {result.get('error', '未知错误')}")
    
    return result

def start_async_initialization():
    """启动异步初始化"""
    return asyncio.run(main())

async def initialize_in_background():
    """后台异步初始化，不阻塞主程序"""
    try:
        logger.info("开始后台异步初始化...")
        result = await main()
        logger.info("后台异步初始化完成")
        return result
    except Exception as e:
        logger.error(f"后台异步初始化失败: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    start_async_initialization()