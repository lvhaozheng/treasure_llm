from app import db
from datetime import datetime

class Antique(db.Model):
    """古董文物模型"""
    __tablename__ = 'antiques'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(100), nullable=False)  # 瓷器、玉器、书画等
    dynasty = db.Column(db.String(100))  # 朝代
    material = db.Column(db.String(100))  # 材质
    description = db.Column(db.Text)
    image_url = db.Column(db.String(500))
    estimated_value = db.Column(db.Float)  # 预估价值
    authenticity_score = db.Column(db.Float)  # 真伪评分
    condition = db.Column(db.String(50))  # 保存状况
    dimensions = db.Column(db.String(100))  # 尺寸
    weight = db.Column(db.Float)  # 重量
    provenance = db.Column(db.Text)  # 来源
    expert_opinion = db.Column(db.Text)  # 专家意见
    ai_analysis = db.Column(db.Text)  # AI分析结果
    
    # 外键
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关联关系
    favorites = db.relationship('Favorite', backref='antique', lazy=True)
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'dynasty': self.dynasty,
            'material': self.material,
            'description': self.description,
            'image_url': self.image_url,
            'estimated_value': self.estimated_value,
            'authenticity_score': self.authenticity_score,
            'condition': self.condition,
            'dimensions': self.dimensions,
            'weight': self.weight,
            'provenance': self.provenance,
            'expert_opinion': self.expert_opinion,
            'ai_analysis': self.ai_analysis,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Favorite(db.Model):
    """收藏模型"""
    __tablename__ = 'favorites'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    antique_id = db.Column(db.Integer, db.ForeignKey('antiques.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'antique_id': self.antique_id,
            'created_at': self.created_at.isoformat()
        }
