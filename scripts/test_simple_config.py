"""
OmniVision配置测试
逐步测试各个组件，避免依赖问题
"""

import os
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_imports():
    """测试基础库导入"""
    print("=== 测试基础库导入 ===")
    
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU 数量: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"❌ PyTorch 导入失败: {e}")
    
    try:
        import transformers
        print(f"✅ Transformers 版本: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers 导入失败: {e}")
    
    try:
        from PIL import Image
        print("✅ Pillow (PIL) 可用")
    except ImportError as e:
        print(f"❌ Pillow 导入失败: {e}")
    
    try:
        import numpy as np
        print(f"✅ NumPy 版本: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy 导入失败: {e}")

def test_clip_import():
    """测试CLIP导入"""
    print("\n=== 测试CLIP导入 ===")
    
    # 测试不同的CLIP包
    clip_packages = [
        ("clip", "import clip"),
        ("openai-clip", "import clip"), 
        ("open_clip", "import open_clip"),
    ]
    
    for package_name, import_cmd in clip_packages:
        try:
            exec(import_cmd)
            print(f"✅ {package_name} 导入成功")
            
            # 测试load方法
            if package_name in ["clip", "openai-clip"]:
                import clip
                if hasattr(clip, 'load'):
                    print(f"✅ {package_name} 有 load 方法")
                    # 尝试获取可用模型
                    try:
                        available_models = clip.available_models()
                        print(f"✅ 可用CLIP模型: {available_models[:3]}...")  # 只显示前3个
                    except Exception as e:
                        print(f"⚠️ 获取可用模型失败: {e}")
                else:
                    print(f"❌ {package_name} 没有 load 方法")
                    
        except ImportError as e:
            print(f"❌ {package_name} 导入失败: {e}")

def test_transformers_omnivision():
    """测试Transformers中的OmniVision模型"""
    print("\n=== 测试Transformers OmniVision ===")
    
    omnivision_classes = [
        ("AutoProcessor", "from transformers import AutoProcessor"),
        ("AutoModelForCausalLM", "from transformers import AutoModelForCausalLM"),
        ("Qwen2VLForConditionalGeneration", "from transformers import Qwen2VLForConditionalGeneration"),
        ("Qwen2VLProcessor", "from transformers import Qwen2VLProcessor"),
    ]
    
    for class_name, import_cmd in omnivision_classes:
        try:
            exec(import_cmd)
            print(f"✅ {class_name} 可用")
        except ImportError as e:
            print(f"❌ {class_name} 不可用: {e}")

def test_model_path():
    """测试模型路径"""
    print("\n=== 测试模型路径 ===")
    
    # 检查默认模型路径
    default_model_path = os.path.join(os.path.dirname(__file__), "ai_core", "models")
    print(f"默认模型路径: {default_model_path}")
    print(f"路径存在: {os.path.exists(default_model_path)}")
    
    if os.path.exists(default_model_path):
        files = os.listdir(default_model_path)
        print(f"模型文件: {files}")
        
        # 检查关键文件
        required_files = ['config.json', 'tokenizer_config.json']
        for file in required_files:
            if file in files:
                print(f"✅ {file} 存在")
            else:
                print(f"❌ {file} 缺失")
                
        # 读取config.json检查模型类型
        config_path = os.path.join(default_model_path, 'config.json')
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                model_type = config.get('model_type', 'unknown')
                architectures = config.get('architectures', [])
                print(f"✅ 模型类型: {model_type}")
                print(f"✅ 架构: {architectures}")
            except Exception as e:
                print(f"❌ 读取config.json失败: {e}")
    else:
        print("❌ 模型路径不存在")

def test_omnivision_client_simple():
    """简化测试OmniVision客户端"""
    print("\n=== 测试OmniVision客户端（简化） ===")
    
    try:
        # 只测试导入
        sys.path.append(os.path.dirname(__file__))
        
        # 先测试单独导入OmniVision客户端
        from ai_core.omnivision.omnivision_client import OmniVisionClient
        print("✅ OmniVisionClient 类导入成功")
        
        # 测试初始化（不加载模型）
        try:
            # 创建一个API模式的客户端（避免加载模型）
            client = OmniVisionClient(api_url="http://dummy")  # 使用假的API URL
            print("✅ OmniVisionClient 初始化成功（API模式）")
            
            # 获取模型信息
            info = client.get_model_info()
            print(f"✅ 模型信息: {info}")
            
        except Exception as e:
            print(f"❌ OmniVisionClient 初始化失败: {e}")
            
    except ImportError as e:
        print(f"❌ OmniVisionClient 导入失败: {e}")

def main():
    """主测试函数"""
    print("开始简化配置测试...")
    print("=" * 60)
    
    # 1. 测试基础库导入
    test_basic_imports()
    
    # 2. 测试CLIP导入
    test_clip_import()
    
    # 3. 测试Transformers OmniVision
    test_transformers_omnivision()
    
    # 4. 测试模型路径
    test_model_path()
    
    # 5. 测试OmniVision客户端
    test_omnivision_client_simple()
    
    print("\n" + "=" * 60)
    print("简化测试完成！")
    print("\n建议：")
    print("1. 如果CLIP有问题，请安装：pip install git+https://github.com/openai/CLIP.git")
    print("2. 如果Transformers Qwen2VL类缺失，请更新：pip install transformers>=4.35.0")
    print("3. 如果模型文件缺失，请检查ai_core/omnivision目录")

if __name__ == "__main__":
    main()