#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文物数据结构定义
包含文物信息、朝代、类别等数据类和枚举
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class Dynasty(Enum):
    """朝代枚举"""
    XIA = "夏朝"
    SHANG = "商朝"
    ZHOU = "周朝"
    QIN = "秦朝"
    HAN = "汉朝"
    THREE_KINGDOMS = "三国两晋南北朝"
    SUI = "隋朝"
    TANG = "唐朝"
    FIVE_DYNASTIES = "五代十国"
    SONG = "宋朝"
    YUAN = "元朝"
    MING = "明朝"
    QING = "清朝"

class RelicCategory(Enum):
    """文物类别枚举"""
    BRONZE = "青铜器"
    POTTERY = "陶器"
    JADE = "玉器"
    PORCELAIN = "瓷器"
    LACQUERWARE = "漆器"
    TEXTILE = "丝织品"
    CALLIGRAPHY = "书法"
    PAINTING = "绘画"
    SCULPTURE = "雕塑"
    METALWARE = "金银器"
    FURNITURE = "家具"
    ENAMEL = "珐琅器"

@dataclass
class CulturalRelicInfo:
    """文物信息数据类"""
    name: str  # 文物名称
    category: RelicCategory  # 文物类别
    dynasty: Dynasty  # 所属朝代
    period: str = ""  # 具体时期
    appearance: str = ""  # 外观描述
    description: str = ""  # 基本描述
    historical_significance: str = ""  # 历史意义
    materials: str = ""  # 材质
    craftsmanship: str = ""  # 工艺特点
    dimensions: str = ""  # 尺寸规格
    current_location: str = ""  # 现存位置
    discovery_info: str = ""  # 发现信息
    cultural_value: str = ""  # 文化价值
    artistic_features: str = ""  # 艺术特色
    preservation_status: str = ""  # 保存状况
    research_value: str = ""  # 研究价值
    related_artifacts: str = ""  # 相关文物
    exhibition_history: str = ""  # 展览历史
    literature_references: str = ""  # 文献记载
    expert_comments: str = ""  # 专家评价
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保枚举类型正确
        if isinstance(self.category, str):
            self.category = RelicCategory(self.category)
        if isinstance(self.dynasty, str):
            self.dynasty = Dynasty(self.dynasty)
    
    def get_full_name(self) -> str:
        """获取完整名称（包含朝代）"""
        return f"{self.dynasty.value}{self.name}"
    
    def get_category_name(self) -> str:
        """获取类别名称"""
        return self.category.value
    
    def get_dynasty_name(self) -> str:
        """获取朝代名称"""
        return self.dynasty.value
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'name': self.name,
            'category': self.category.value,
            'dynasty': self.dynasty.value,
            'period': self.period,
            'appearance': self.appearance,
            'description': self.description,
            'historical_significance': self.historical_significance,
            'materials': self.materials,
            'craftsmanship': self.craftsmanship,
            'dimensions': self.dimensions,
            'current_location': self.current_location,
            'discovery_info': self.discovery_info,
            'cultural_value': self.cultural_value,
            'artistic_features': self.artistic_features,
            'preservation_status': self.preservation_status,
            'research_value': self.research_value,
            'related_artifacts': self.related_artifacts,
            'exhibition_history': self.exhibition_history,
            'literature_references': self.literature_references,
            'expert_comments': self.expert_comments
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CulturalRelicInfo':
        """从字典创建实例"""
        # 处理枚举类型
        if 'category' in data and isinstance(data['category'], str):
            data['category'] = RelicCategory(data['category'])
        if 'dynasty' in data and isinstance(data['dynasty'], str):
            data['dynasty'] = Dynasty(data['dynasty'])
        
        return cls(**data)