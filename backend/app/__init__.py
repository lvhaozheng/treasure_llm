import os
import sys
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.logger_config import get_backend_logger

# 配置后端初始化模块日志
logger = get_backend_logger('app_init')

# 加载环境变量
load_dotenv()

# 初始化扩展
db = SQLAlchemy()
jwt = JWTManager()

def create_app():
    """创建Flask应用"""
    app = Flask(__name__)
    
    # 配置
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///antique_app.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key')
    
    # 初始化扩展
    CORS(app)
    db.init_app(app)
    jwt.init_app(app)
    
    # 注册蓝图
    from .routes import auth_bp, antique_bp, ai_bp
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(antique_bp, url_prefix='/api/antique')
    app.register_blueprint(ai_bp, url_prefix='/api/ai')
    
    # 创建数据库表
    with app.app_context():
        db.create_all()
    
    return app
