import os
import sys
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token
from app import db
from models.user import User
from models.antique import Antique, Favorite
from services.ai_service import AIService
from services.image_service import ImageService

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.logger_config import get_backend_logger

# 配置路由模块日志
logger = get_backend_logger('routes')

# 创建蓝图
auth_bp = Blueprint('auth', __name__)
antique_bp = Blueprint('antique', __name__)
ai_bp = Blueprint('ai', __name__)

# 初始化服务
ai_service = AIService()
image_service = ImageService()

# 认证相关路由
@auth_bp.route('/register', methods=['POST'])
def register():
    """用户注册"""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'error': '缺少必要参数'}), 400
    
    # 检查用户是否已存在
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': '用户名已存在'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': '邮箱已存在'}), 400
    
    # 创建新用户
    user = User(
        username=data['username'],
        email=data['email']
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'message': '注册成功', 'user': user.to_dict()}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    """用户登录"""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': '缺少必要参数'}), 400
    
    user = User.query.filter_by(username=data['username']).first()
    
    if user and user.check_password(data['password']):
        access_token = create_access_token(identity=user.id)
        return jsonify({
            'message': '登录成功',
            'access_token': access_token,
            'user': user.to_dict()
        }), 200
    
    return jsonify({'error': '用户名或密码错误'}), 401

# 古董相关路由
@antique_bp.route('/', methods=['GET'])
@jwt_required()
def get_antiques():
    """获取用户的古董列表"""
    user_id = get_jwt_identity()
    antiques = Antique.query.filter_by(user_id=user_id).all()
    return jsonify([antique.to_dict() for antique in antiques]), 200

@antique_bp.route('/', methods=['POST'])
@jwt_required()
def create_antique():
    """创建古董记录"""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    if not data or not data.get('name') or not data.get('category'):
        return jsonify({'error': '缺少必要参数'}), 400
    
    antique = Antique(
        name=data['name'],
        category=data['category'],
        dynasty=data.get('dynasty'),
        material=data.get('material'),
        description=data.get('description'),
        image_url=data.get('image_url'),
        estimated_value=data.get('estimated_value'),
        authenticity_score=data.get('authenticity_score'),
        condition=data.get('condition'),
        dimensions=data.get('dimensions'),
        weight=data.get('weight'),
        provenance=data.get('provenance'),
        expert_opinion=data.get('expert_opinion'),
        ai_analysis=data.get('ai_analysis'),
        user_id=user_id
    )
    
    db.session.add(antique)
    db.session.commit()
    
    return jsonify({'message': '创建成功', 'antique': antique.to_dict()}), 201

@antique_bp.route('/<int:antique_id>', methods=['GET'])
@jwt_required()
def get_antique(antique_id):
    """获取单个古董详情"""
    user_id = get_jwt_identity()
    antique = Antique.query.filter_by(id=antique_id, user_id=user_id).first()
    
    if not antique:
        return jsonify({'error': '古董不存在'}), 404
    
    return jsonify(antique.to_dict()), 200

@antique_bp.route('/<int:antique_id>', methods=['PUT'])
@jwt_required()
def update_antique(antique_id):
    """更新古董信息"""
    user_id = get_jwt_identity()
    antique = Antique.query.filter_by(id=antique_id, user_id=user_id).first()
    
    if not antique:
        return jsonify({'error': '古董不存在'}), 404
    
    data = request.get_json()
    
    # 更新字段
    for field in ['name', 'category', 'dynasty', 'material', 'description', 
                  'image_url', 'estimated_value', 'authenticity_score', 
                  'condition', 'dimensions', 'weight', 'provenance', 
                  'expert_opinion', 'ai_analysis']:
        if field in data:
            setattr(antique, field, data[field])
    
    db.session.commit()
    
    return jsonify({'message': '更新成功', 'antique': antique.to_dict()}), 200

@antique_bp.route('/<int:antique_id>', methods=['DELETE'])
@jwt_required()
def delete_antique(antique_id):
    """删除古董"""
    user_id = get_jwt_identity()
    antique = Antique.query.filter_by(id=antique_id, user_id=user_id).first()
    
    if not antique:
        return jsonify({'error': '古董不存在'}), 404
    
    db.session.delete(antique)
    db.session.commit()
    
    return jsonify({'message': '删除成功'}), 200

# 收藏相关路由
@antique_bp.route('/<int:antique_id>/favorite', methods=['POST'])
@jwt_required()
def add_favorite(antique_id):
    """添加收藏"""
    user_id = get_jwt_identity()
    
    # 检查是否已收藏
    existing_favorite = Favorite.query.filter_by(
        user_id=user_id, antique_id=antique_id
    ).first()
    
    if existing_favorite:
        return jsonify({'error': '已收藏'}), 400
    
    favorite = Favorite(user_id=user_id, antique_id=antique_id)
    db.session.add(favorite)
    db.session.commit()
    
    return jsonify({'message': '收藏成功'}), 201

@antique_bp.route('/<int:antique_id>/favorite', methods=['DELETE'])
@jwt_required()
def remove_favorite(antique_id):
    """取消收藏"""
    user_id = get_jwt_identity()
    favorite = Favorite.query.filter_by(
        user_id=user_id, antique_id=antique_id
    ).first()
    
    if not favorite:
        return jsonify({'error': '未收藏'}), 404
    
    db.session.delete(favorite)
    db.session.commit()
    
    return jsonify({'message': '取消收藏成功'}), 200

# AI相关路由
@ai_bp.route('/analyze', methods=['POST'])
@jwt_required()
def analyze_antique():
    """AI分析古董"""
    user_id = get_jwt_identity()
    
    if 'image' not in request.files:
        return jsonify({'error': '请上传图片'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    # 保存图片
    image_path = image_service.save_image(file)
    
    try:
        # AI分析
        analysis_result = ai_service.analyze_antique(image_path)
        
        return jsonify({
            'message': '分析完成',
            'analysis': analysis_result
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'分析失败: {str(e)}'}), 500

@ai_bp.route('/search', methods=['POST'])
@jwt_required()
def search_antiques():
    """搜索相似古董"""
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': '请输入搜索内容'}), 400
    
    try:
        # AI搜索
        search_results = ai_service.search_antiques(query)
        
        return jsonify({
            'message': '搜索完成',
            'results': search_results
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'搜索失败: {str(e)}'}), 500
