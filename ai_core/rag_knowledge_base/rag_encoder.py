# -*- coding: utf-8 -*-
"""
RAG知识库编码器
使用CLIP模型对古董图文数据进行批量编码
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 简化的日志记录
class SimpleLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    
    def error(self, msg):
        print(f"[ERROR] {msg}")
    
    def warning(self, msg):
        print(f"[WARNING] {msg}")

logger = SimpleLogger()

@dataclass
class VectorData:
    """向量数据结构"""
    vector_id: str
    antique_id: int
    vector_type: str  # 'image' or 'text'
    vector: List[float]
    source_id: str  # image_id 或 text_id
    metadata: Dict[str, Any]

class RAGEncoder:
    """RAG知识库编码器"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.clip_encoder = None
        self.device = "cpu"  # 默认使用CPU
        
        # 尝试初始化CLIP编码器
        self._init_clip_encoder()
    
    def _init_clip_encoder(self):
        """初始化CLIP编码器"""
        try:
            # 尝试导入CLIP编码器
            from ai_core.clip_encoder import CLIPEncoder
            self.clip_encoder = CLIPEncoder()
            logger.info("CLIP编码器初始化成功")
        except ImportError as e:
            logger.warning(f"无法导入CLIP编码器: {e}")
            logger.info("将使用模拟编码器")
            self.clip_encoder = None
    
    def _simulate_text_encoding(self, text: str) -> List[float]:
        """模拟文本编码（用于测试）"""
        # 使用文本哈希生成固定长度的向量
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
    
    def _simulate_image_encoding(self, image_path: str) -> List[float]:
        """模拟图像编码（用于测试）"""
        # 使用图像路径生成固定向量
        import hashlib
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
    
    def encode_text(self, text: str) -> List[float]:
        """编码文本"""
        if self.clip_encoder:
            try:
                # 使用真实的CLIP编码器
                vector = self.clip_encoder.encode_text(text)
                return vector.tolist() if hasattr(vector, 'tolist') else vector
            except Exception as e:
                logger.error(f"CLIP文本编码失败: {e}")
                return self._simulate_text_encoding(text)
        else:
            # 使用模拟编码器
            return self._simulate_text_encoding(text)
    
    def encode_image(self, image_path: str) -> List[float]:
        """编码图像"""
        if self.clip_encoder:
            try:
                # 使用真实的CLIP编码器
                vector = self.clip_encoder.encode_image(image_path)
                return vector.tolist() if hasattr(vector, 'tolist') else vector
            except Exception as e:
                logger.error(f"CLIP图像编码失败: {e}")
                return self._simulate_image_encoding(image_path)
        else:
            # 使用模拟编码器
            return self._simulate_image_encoding(image_path)
    
    def encode_rag_entry(self, entry_dict: Dict[str, Any]) -> List[VectorData]:
        """编码单个RAG知识库条目"""
        vectors = []
        antique_id = entry_dict['antique_metadata']['antique_id']
        
        # 编码图像数据
        for image_data in entry_dict.get('images', []):
            try:
                image_vector = self.encode_image(image_data['file_path'])
                
                vector_data = VectorData(
                    vector_id=f"vec_{antique_id:03d}_img_{image_data['image_id']}",
                    antique_id=antique_id,
                    vector_type="image",
                    vector=image_vector,
                    source_id=image_data['image_id'],
                    metadata={
                        "description": image_data['description'],
                        "view_angle": image_data['view_angle'],
                        "format": image_data['format'],
                        "width": image_data['width'],
                        "height": image_data['height']
                    }
                )
                vectors.append(vector_data)
                
            except Exception as e:
                logger.error(f"编码图像 {image_data['image_id']} 失败: {e}")
        
        # 编码文本数据
        for text_data in entry_dict.get('texts', []):
            try:
                text_vector = self.encode_text(text_data['content'])
                
                vector_data = VectorData(
                    vector_id=f"vec_{antique_id:03d}_txt_{text_data['text_id']}",
                    antique_id=antique_id,
                    vector_type="text",
                    vector=text_vector,
                    source_id=text_data['text_id'],
                    metadata={
                        "text_type": text_data['text_type'],
                        "language": text_data['language'],
                        "content_length": len(text_data['content'])
                    }
                )
                vectors.append(vector_data)
                
            except Exception as e:
                logger.error(f"编码文本 {text_data['text_id']} 失败: {e}")
        
        return vectors
    
    def encode_dataset(self, dataset_path: str) -> Tuple[List[VectorData], Dict[str, Any]]:
        """编码整个数据集"""
        logger.info(f"开始编码数据集: {dataset_path}")
        
        # 加载数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        all_vectors = []
        stats = {
            "total_entries": len(dataset['entries']),
            "total_vectors": 0,
            "image_vectors": 0,
            "text_vectors": 0,
            "failed_encodings": 0,
            "processing_time": 0
        }
        
        start_time = datetime.now()
        
        # 批量编码
        for i, entry in enumerate(dataset['entries']):
            try:
                vectors = self.encode_rag_entry(entry)
                all_vectors.extend(vectors)
                
                # 统计
                for vector in vectors:
                    if vector.vector_type == "image":
                        stats["image_vectors"] += 1
                    else:
                        stats["text_vectors"] += 1
                
                if (i + 1) % 50 == 0:
                    logger.info(f"已处理 {i + 1}/{stats['total_entries']} 条数据")
                    
            except Exception as e:
                logger.error(f"编码条目 {i} 失败: {e}")
                stats["failed_encodings"] += 1
        
        end_time = datetime.now()
        stats["processing_time"] = (end_time - start_time).total_seconds()
        stats["total_vectors"] = len(all_vectors)
        
        logger.info(f"编码完成: {stats['total_vectors']} 个向量")
        logger.info(f"图像向量: {stats['image_vectors']}, 文本向量: {stats['text_vectors']}")
        logger.info(f"处理时间: {stats['processing_time']:.2f} 秒")
        
        return all_vectors, stats
    
    def save_vectors(self, vectors: List[VectorData], output_path: str) -> str:
        """保存向量数据"""
        # 转换为可序列化的格式
        vector_data = {
            "metadata": {
                "total_vectors": len(vectors),
                "created_at": datetime.now().isoformat(),
                "encoder_version": "1.0",
                "vector_dimension": 512
            },
            "vectors": [
                {
                    "vector_id": v.vector_id,
                    "antique_id": v.antique_id,
                    "vector_type": v.vector_type,
                    "vector": v.vector,
                    "source_id": v.source_id,
                    "metadata": v.metadata
                }
                for v in vectors
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vector_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"向量数据已保存到: {output_path}")
        return output_path
    
    def process_dataset(self, dataset_path: str) -> str:
        """处理数据集的完整流程"""
        # 编码数据集
        vectors, stats = self.encode_dataset(dataset_path)
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"rag_vectors_{timestamp}.json"
        output_path = os.path.join(self.base_dir, output_filename)
        
        # 保存向量数据
        self.save_vectors(vectors, output_path)
        
        # 保存统计信息
        stats_filename = f"encoding_stats_{timestamp}.json"
        stats_path = os.path.join(self.base_dir, stats_filename)
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"统计信息已保存到: {stats_path}")
        
        return output_path

def main():
    """主函数"""
    encoder = RAGEncoder()
    
    # 查找最新的数据集文件
    base_dir = os.path.dirname(__file__)
    dataset_files = [f for f in os.listdir(base_dir) if f.startswith('rag_sample_dataset_') and f.endswith('.json')]
    
    if not dataset_files:
        logger.error("未找到数据集文件")
        return
    
    # 使用最新的数据集文件
    latest_dataset = sorted(dataset_files)[-1]
    dataset_path = os.path.join(base_dir, latest_dataset)
    
    logger.info(f"使用数据集: {latest_dataset}")
    
    # 处理数据集
    output_path = encoder.process_dataset(dataset_path)
    
    logger.info(f"编码完成，向量数据保存在: {output_path}")
    
    return output_path

if __name__ == "__main__":
    main()