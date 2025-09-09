#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 知识库数据结构设计
用于定义古董图文数据的存储格式和元数据结构

作者: AI Assistant
创建时间: 2025-09-07
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class AntiqueCategory(Enum):
    """古董类别枚举"""
    PORCELAIN = "porcelain"  # 瓷器
    BRONZE = "bronze"  # 青铜器
    JADE = "jade"  # 玉器
    CALLIGRAPHY = "calligraphy"  # 书法
    PAINTING = "painting"  # 绘画
    FURNITURE = "furniture"  # 家具
    JEWELRY = "jewelry"  # 首饰
    SCULPTURE = "sculpture"  # 雕塑
    TEXTILE = "textile"  # 纺织品
    OTHER = "other"  # 其他


class Dynasty(Enum):
    """朝代枚举"""
    SHANG = "shang"  # 商朝
    ZHOU = "zhou"  # 周朝
    QIN = "qin"  # 秦朝
    HAN = "han"  # 汉朝
    TANG = "tang"  # 唐朝
    SONG = "song"  # 宋朝
    YUAN = "yuan"  # 元朝
    MING = "ming"  # 明朝
    QING = "qing"  # 清朝
    REPUBLIC = "republic"  # 民国
    MODERN = "modern"  # 现代
    UNKNOWN = "unknown"  # 未知


class Condition(Enum):
    """保存状况枚举"""
    EXCELLENT = "excellent"  # 完好
    GOOD = "good"  # 良好
    FAIR = "fair"  # 一般
    POOR = "poor"  # 较差
    DAMAGED = "damaged"  # 损坏
    UNKNOWN = "unknown"  # 未知


@dataclass
class AntiqueMetadata:
    """古董元数据结构"""
    # 基本信息
    antique_id: int
    name: str
    category: AntiqueCategory
    dynasty: Dynasty
    estimated_year: Optional[str] = None  # 估计年代
    
    # 物理特征
    material: str = ""  # 材质
    dimensions: str = ""  # 尺寸
    weight: Optional[float] = None  # 重量(克)
    color: str = ""  # 颜色
    
    # 艺术特征
    style: str = ""  # 风格
    pattern: str = ""  # 纹饰
    technique: str = ""  # 工艺技法
    inscription: str = ""  # 铭文
    
    # 状况和价值
    condition: Condition = Condition.UNKNOWN
    estimated_value: Optional[float] = None  # 估价(元)
    rarity: str = ""  # 稀有程度
    
    # 历史背景
    historical_context: str = ""  # 历史背景
    cultural_significance: str = ""  # 文化意义
    provenance: str = ""  # 来源
    
    # 鉴定信息
    authenticity: str = ""  # 真伪判断
    expert_opinion: str = ""  # 专家意见
    authentication_method: str = ""  # 鉴定方法
    
    # 数据来源
    source: str = ""  # 数据来源
    source_url: str = ""  # 来源链接
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换枚举值为字符串
        data['category'] = self.category.value
        data['dynasty'] = self.dynasty.value
        data['condition'] = self.condition.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AntiqueMetadata':
        """从字典创建实例"""
        # 转换字符串为枚举
        if 'category' in data:
            data['category'] = AntiqueCategory(data['category'])
        if 'dynasty' in data:
            data['dynasty'] = Dynasty(data['dynasty'])
        if 'condition' in data:
            data['condition'] = Condition(data['condition'])
        return cls(**data)


@dataclass
class ImageData:
    """图像数据结构"""
    image_id: str
    antique_id: int
    file_path: str
    file_name: str
    file_size: int  # 字节
    width: int
    height: int
    format: str  # jpg, png, etc.
    description: str = ""  # 图像描述
    view_angle: str = ""  # 拍摄角度
    lighting: str = ""  # 光照条件
    quality_score: Optional[float] = None  # 图像质量评分
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class TextData:
    """文本数据结构"""
    text_id: str
    antique_id: int
    content: str
    text_type: str  # description, history, expert_analysis, etc.
    language: str = "zh"  # 语言
    word_count: int = 0
    source: str = ""
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.word_count:
            self.word_count = len(self.content)


@dataclass
class VectorData:
    """向量数据结构"""
    vector_id: str
    antique_id: int
    vector_type: str  # image, text
    vector: List[float]
    dimension: int
    model_name: str  # CLIP模型名称
    source_id: str  # 对应的image_id或text_id
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.dimension:
            self.dimension = len(self.vector)


@dataclass
class RAGKnowledgeEntry:
    """RAG知识库条目"""
    antique_metadata: AntiqueMetadata
    images: List[ImageData]
    texts: List[TextData]
    vectors: List[VectorData]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'antique_metadata': self.antique_metadata.to_dict(),
            'images': [asdict(img) for img in self.images],
            'texts': [asdict(txt) for txt in self.texts],
            'vectors': [asdict(vec) for vec in self.vectors]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGKnowledgeEntry':
        """从字典创建实例"""
        return cls(
            antique_metadata=AntiqueMetadata.from_dict(data['antique_metadata']),
            images=[ImageData(**img) for img in data['images']],
            texts=[TextData(**txt) for txt in data['texts']],
            vectors=[VectorData(**vec) for vec in data['vectors']]
        )


class RAGDataValidator:
    """RAG数据验证器"""
    
    @staticmethod
    def validate_metadata(metadata: AntiqueMetadata) -> List[str]:
        """验证元数据"""
        errors = []
        
        if not metadata.name:
            errors.append("古董名称不能为空")
        
        if metadata.estimated_value and metadata.estimated_value < 0:
            errors.append("估价不能为负数")
        
        if metadata.weight and metadata.weight < 0:
            errors.append("重量不能为负数")
        
        return errors
    
    @staticmethod
    def validate_image(image: ImageData) -> List[str]:
        """验证图像数据"""
        errors = []
        
        if not image.file_path:
            errors.append("图像文件路径不能为空")
        
        if image.width <= 0 or image.height <= 0:
            errors.append("图像尺寸必须大于0")
        
        if image.file_size <= 0:
            errors.append("文件大小必须大于0")
        
        return errors
    
    @staticmethod
    def validate_text(text: TextData) -> List[str]:
        """验证文本数据"""
        errors = []
        
        if not text.content:
            errors.append("文本内容不能为空")
        
        if len(text.content) < 10:
            errors.append("文本内容过短，至少需要10个字符")
        
        return errors
    
    @staticmethod
    def validate_vector(vector: VectorData) -> List[str]:
        """验证向量数据"""
        errors = []
        
        if not vector.vector:
            errors.append("向量不能为空")
        
        if vector.dimension != len(vector.vector):
            errors.append("向量维度与实际长度不匹配")
        
        return errors


# 示例数据生成器
def create_sample_antique_data() -> RAGKnowledgeEntry:
    """创建示例古董数据"""
    metadata = AntiqueMetadata(
        antique_id=1,
        name="青花瓷花瓶",
        category=AntiqueCategory.PORCELAIN,
        dynasty=Dynasty.MING,
        estimated_year="明代永乐年间",
        material="瓷器",
        dimensions="高30cm，口径12cm",
        color="青花",
        pattern="缠枝莲纹",
        technique="釉下彩",
        condition=Condition.GOOD,
        estimated_value=500000.0,
        historical_context="明代永乐年间官窑精品",
        source="故宫博物院"
    )
    
    images = [
        ImageData(
            image_id="img_001",
            antique_id=1,
            file_path="/data/images/ming_vase_001.jpg",
            file_name="ming_vase_001.jpg",
            file_size=2048576,
            width=1920,
            height=1080,
            format="jpg",
            description="青花瓷花瓶正面图",
            view_angle="正面"
        )
    ]
    
    texts = [
        TextData(
            text_id="txt_001",
            antique_id=1,
            content="这是一件明代永乐年间的青花瓷花瓶，器型端庄，釉色纯正，纹饰精美。",
            text_type="description",
            language="zh"
        )
    ]
    
    vectors = []  # 向量数据将在编码时生成
    
    return RAGKnowledgeEntry(
        antique_metadata=metadata,
        images=images,
        texts=texts,
        vectors=vectors
    )


def main():
    """主函数 - 用于测试数据结构"""
    # 创建示例数据
    sample_entry = create_sample_antique_data()
    
    # 验证数据
    validator = RAGDataValidator()
    
    metadata_errors = validator.validate_metadata(sample_entry.antique_metadata)
    if metadata_errors:
        print("元数据验证错误:", metadata_errors)
    
    for image in sample_entry.images:
        image_errors = validator.validate_image(image)
        if image_errors:
            print(f"图像 {image.image_id} 验证错误:", image_errors)
    
    for text in sample_entry.texts:
        text_errors = validator.validate_text(text)
        if text_errors:
            print(f"文本 {text.text_id} 验证错误:", text_errors)
    
    # 转换为字典并打印
    entry_dict = sample_entry.to_dict()
    print("示例数据结构:")
    print(json.dumps(entry_dict, ensure_ascii=False, indent=2))
    
    print("\nRAG数据结构测试完成")


if __name__ == "__main__":
    main()