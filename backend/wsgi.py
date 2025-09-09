#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSGI入口文件
用于生产环境部署Flask应用
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

# 设置环境变量
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('PYTHONPATH', str(project_root))

try:
    # 直接执行app.py模块来获取Flask应用
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "app.py")
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    
    # 获取Flask应用实例
    application = app_module.app
    
    # 初始化AI核心
    if hasattr(app_module, 'init_ai_core'):
        if app_module.init_ai_core():
            print("✅ AI核心初始化成功")
        else:
            print("❌ AI核心初始化失败")
            sys.exit(1)
    else:
        print("⚠️ 未找到AI核心初始化函数")
    
    # 配置生产环境设置
    application.config['DEBUG'] = False
    application.config['TESTING'] = False
    
    print("✅ WSGI应用加载成功")
    
except ImportError as e:
    print(f"❌ 导入Flask应用失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ WSGI应用初始化失败: {e}")
    sys.exit(1)

if __name__ == "__main__":
    # 开发环境直接运行
    application.run(host='0.0.0.0', port=5000, debug=False)