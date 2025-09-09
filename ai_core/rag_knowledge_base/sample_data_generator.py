# -*- coding: utf-8 -*-
"""
RAG知识库样本数据生成器
用于生成包含500条高质量古董图文数据的样本数据集
"""

import os
import sys
import json
import random
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 定义数据结构
class AntiqueCategory(Enum):
    PORCELAIN = "瓷器"
    BRONZE = "青铜器"
    JADE = "玉器"
    PAINTING = "绘画"
    CALLIGRAPHY = "书法"
    FURNITURE = "家具"

class Dynasty(Enum):
    SHANG = "商代"
    ZHOU = "周代"
    HAN = "汉代"
    TANG = "唐代"
    SONG = "宋代"
    MING = "明代"
    QING = "清代"

class Condition(Enum):
    EXCELLENT = "完好"
    GOOD = "良好"
    FAIR = "一般"
    POOR = "较差"

@dataclass
class AntiqueMetadata:
    antique_id: int
    name: str
    category: AntiqueCategory
    dynasty: Dynasty
    estimated_year: str
    material: str
    pattern: str
    technique: str
    condition: Condition
    estimated_value: float
    historical_context: str
    source: str

@dataclass
class ImageData:
    image_id: str
    antique_id: int
    file_path: str
    file_name: str
    file_size: int
    width: int
    height: int
    format: str
    description: str
    view_angle: str

@dataclass
class TextData:
    text_id: str
    antique_id: int
    content: str
    text_type: str
    language: str

@dataclass
class RAGKnowledgeEntry:
    antique_metadata: AntiqueMetadata
    images: List[ImageData]
    texts: List[TextData]
    vectors: List = None
    
    def to_dict(self):
        return {
            "antique_metadata": {
                "antique_id": self.antique_metadata.antique_id,
                "name": self.antique_metadata.name,
                "category": self.antique_metadata.category.value,
                "dynasty": self.antique_metadata.dynasty.value,
                "estimated_year": self.antique_metadata.estimated_year,
                "material": self.antique_metadata.material,
                "pattern": self.antique_metadata.pattern,
                "technique": self.antique_metadata.technique,
                "condition": self.antique_metadata.condition.value,
                "estimated_value": self.antique_metadata.estimated_value,
                "historical_context": self.antique_metadata.historical_context,
                "source": self.antique_metadata.source
            },
            "images": [{
                "image_id": img.image_id,
                "antique_id": img.antique_id,
                "file_path": img.file_path,
                "file_name": img.file_name,
                "file_size": img.file_size,
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "description": img.description,
                "view_angle": img.view_angle
            } for img in self.images],
            "texts": [{
                "text_id": txt.text_id,
                "antique_id": txt.antique_id,
                "content": txt.content,
                "text_type": txt.text_type,
                "language": txt.language
            } for txt in self.texts],
            "vectors": self.vectors or []
        }

# 简化的日志记录
class SimpleLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")

logger = SimpleLogger()

class SampleDataGenerator:
    """RAG知识库样本数据生成器"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        
        # 瓷器样本数据
        self.porcelain_samples = [
            {
                "name": "青花缠枝莲纹梅瓶",
                "dynasty": Dynasty.MING,
                "year": "明永乐年间(1403-1424)",
                "pattern": "缠枝莲纹",
                "technique": "青花釉下彩",
                "value": 5000000.0,
                "description": "明代永乐年间青花瓷器的代表作品，器型优美，釉色纯正，纹饰精细。",
                "context": "永乐青花是明代瓷器的巅峰之作，以其纯正的钴蓝色和精湛的工艺闻名于世。"
            },
            {
                "name": "斗彩鸡缸杯",
                "dynasty": Dynasty.MING,
                "year": "明成化年间(1465-1487)",
                "pattern": "子母鸡图案",
                "technique": "斗彩工艺",
                "value": 28000000.0,
                "description": "成化斗彩的经典之作，色彩丰富，画工精细，是明代瓷器的珍品。",
                "context": "成化斗彩以其独特的釉上釉下结合技法，创造了瓷器装饰的新境界。"
            }
        ]
        
        # 青铜器样本数据
        self.bronze_samples = [
            {
                "name": "司母戊鼎",
                "dynasty": Dynasty.SHANG,
                "year": "商代晚期(约公元前13-11世纪)",
                "pattern": "饕餮纹",
                "technique": "青铜铸造",
                "value": 100000000.0,
                "description": "中国现存最大的青铜器，造型雄伟，纹饰精美，是商代青铜文明的杰出代表。",
                "context": "商代青铜器以其精湛的铸造工艺和神秘的纹饰图案，展现了古代中国的文明高度。"
            }
        ]
        
        # 玉器样本数据
        self.jade_samples = [
            {
                "name": "白玉龙凤纹璧",
                "dynasty": Dynasty.HAN,
                "year": "汉代(公元前206-公元220)",
                "pattern": "龙凤纹",
                "technique": "圆雕镂空",
                "value": 8000000.0,
                "description": "汉代玉器的精品，雕工精细，寓意吉祥，体现了汉代玉器的艺术水平。",
                "context": "汉代玉器承继了战国玉器的传统，在工艺和题材上都有所发展和创新。"
            }
        ]
    
    def generate_porcelain_entry(self, antique_id: int) -> RAGKnowledgeEntry:
        """生成瓷器数据条目"""
        sample = random.choice(self.porcelain_samples)
        
        # 创建变体名称
        patterns = ["缠枝莲纹", "云龙纹", "花鸟纹", "山水纹", "人物纹"]
        shapes = ["梅瓶", "玉壶春瓶", "盘", "碗", "罐"]
        
        name = f"青花{random.choice(patterns)}{random.choice(shapes)}"
        
        metadata = AntiqueMetadata(
            antique_id=antique_id,
            name=name,
            category=AntiqueCategory.PORCELAIN,
            dynasty=sample["dynasty"],
            estimated_year=sample["year"],
            material="瓷器",
            pattern=sample["pattern"],
            technique=sample["technique"],
            condition=random.choice(list(Condition)),
            estimated_value=sample["value"] * random.uniform(0.5, 2.0),
            historical_context=sample["context"],
            source="故宫博物院"
        )
        
        images = [
            ImageData(
                image_id=f"img_{antique_id:03d}_01",
                antique_id=antique_id,
                file_path=f"/rag_knowledge_base/images/{antique_id:03d}_front.jpg",
                file_name=f"{antique_id:03d}_front.jpg",
                file_size=2048576,
                width=1920,
                height=1080,
                format="jpg",
                description=f"{name}正面图",
                view_angle="正面"
            )
        ]
        
        texts = [
            TextData(
                text_id=f"txt_{antique_id:03d}_desc",
                antique_id=antique_id,
                content=sample["description"],
                text_type="description",
                language="zh"
            )
        ]
        
        return RAGKnowledgeEntry(
            antique_metadata=metadata,
            images=images,
            texts=texts,
            vectors=[]
        )
    
    def generate_bronze_entry(self, antique_id: int) -> RAGKnowledgeEntry:
        """生成青铜器数据条目"""
        sample = random.choice(self.bronze_samples)
        
        shapes = ["鼎", "簋", "觚", "爵", "卣"]
        patterns = ["饕餮纹", "夔龙纹", "凤鸟纹", "几何纹"]
        
        name = f"{random.choice(patterns)}{random.choice(shapes)}"
        
        metadata = AntiqueMetadata(
            antique_id=antique_id,
            name=name,
            category=AntiqueCategory.BRONZE,
            dynasty=sample["dynasty"],
            estimated_year=sample["year"],
            material="青铜",
            pattern=sample["pattern"],
            technique=sample["technique"],
            condition=random.choice(list(Condition)),
            estimated_value=sample["value"] * random.uniform(0.3, 1.5),
            historical_context=sample["context"],
            source="中国国家博物馆"
        )
        
        images = [
            ImageData(
                image_id=f"img_{antique_id:03d}_01",
                antique_id=antique_id,
                file_path=f"/rag_knowledge_base/images/{antique_id:03d}_front.jpg",
                file_name=f"{antique_id:03d}_front.jpg",
                file_size=1800000,
                width=1600,
                height=1200,
                format="jpg",
                description=f"{name}正面图",
                view_angle="正面"
            )
        ]
        
        texts = [
            TextData(
                text_id=f"txt_{antique_id:03d}_desc",
                antique_id=antique_id,
                content=sample["description"],
                text_type="description",
                language="zh"
            )
        ]
        
        return RAGKnowledgeEntry(
            antique_metadata=metadata,
            images=images,
            texts=texts,
            vectors=[]
        )
    
    def generate_jade_entry(self, antique_id: int) -> RAGKnowledgeEntry:
        """生成玉器数据条目"""
        sample = random.choice(self.jade_samples)
        
        shapes = ["璧", "环", "佩", "琮", "圭"]
        materials = ["白玉", "青玉", "碧玉", "黄玉"]
        
        name = f"{random.choice(materials)}{random.choice(shapes)}"
        
        metadata = AntiqueMetadata(
            antique_id=antique_id,
            name=name,
            category=AntiqueCategory.JADE,
            dynasty=sample["dynasty"],
            estimated_year=sample["year"],
            material="玉石",
            pattern=sample["pattern"],
            technique=sample["technique"],
            condition=random.choice(list(Condition)),
            estimated_value=sample["value"] * random.uniform(0.4, 1.8),
            historical_context=sample["context"],
            source="上海博物馆"
        )
        
        images = [
            ImageData(
                image_id=f"img_{antique_id:03d}_01",
                antique_id=antique_id,
                file_path=f"/rag_knowledge_base/images/{antique_id:03d}_front.jpg",
                file_name=f"{antique_id:03d}_front.jpg",
                file_size=1500000,
                width=1400,
                height=1000,
                format="jpg",
                description=f"{name}正面图",
                view_angle="正面"
            )
        ]
        
        texts = [
            TextData(
                text_id=f"txt_{antique_id:03d}_desc",
                antique_id=antique_id,
                content=sample["description"],
                text_type="description",
                language="zh"
            )
        ]
        
        return RAGKnowledgeEntry(
            antique_metadata=metadata,
            images=images,
            texts=texts,
            vectors=[]
        )
    
    def generate_extended_dataset(self, target_count: int = 500) -> List[RAGKnowledgeEntry]:
        """生成扩展数据集"""
        entries = []
        
        # 按比例分配不同类别的数据
        porcelain_count = int(target_count * 0.5)  # 50% 瓷器
        bronze_count = int(target_count * 0.3)     # 30% 青铜器
        jade_count = target_count - porcelain_count - bronze_count  # 20% 玉器
        
        antique_id = 1
        
        # 生成瓷器数据
        for _ in range(porcelain_count):
            entries.append(self.generate_porcelain_entry(antique_id))
            antique_id += 1
        
        # 生成青铜器数据
        for _ in range(bronze_count):
            entries.append(self.generate_bronze_entry(antique_id))
            antique_id += 1
        
        # 生成玉器数据
        for _ in range(jade_count):
            entries.append(self.generate_jade_entry(antique_id))
            antique_id += 1
        
        return entries
    
    def save_dataset(self, entries: List[RAGKnowledgeEntry]) -> str:
        """保存数据集到JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_sample_dataset_{timestamp}.json"
        filepath = os.path.join(self.base_dir, filename)
        
        # 转换为字典格式
        dataset = {
            "metadata": {
                "total_count": len(entries),
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "entries": [entry.to_dict() for entry in entries]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集已保存到: {filepath}")
        return filepath

def main():
    """主函数"""
    generator = SampleDataGenerator()
    
    # 生成500条数据
    logger.info("开始生成RAG知识库样本数据...")
    entries = generator.generate_extended_dataset(target_count=500)
    
    # 保存数据集
    filepath = generator.save_dataset(entries)
    
    # 统计信息
    categories = {}
    dynasties = {}
    for entry in entries:
        cat = entry.antique_metadata.category.value
        dyn = entry.antique_metadata.dynasty.value
        categories[cat] = categories.get(cat, 0) + 1
        dynasties[dyn] = dynasties.get(dyn, 0) + 1
    
    logger.info("数据集统计信息:")
    logger.info(f"类别分布: {categories}")
    logger.info(f"朝代分布: {dynasties}")
    
    return filepath

if __name__ == "__main__":
    main()