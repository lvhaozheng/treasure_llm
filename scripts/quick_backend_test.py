#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速后端接口测试脚本
专注于测试核心功能，快速验证后端服务状态
"""

import requests
import json
import time
from datetime import datetime

# 后端服务配置
BACKEND_BASE_URL = "http://localhost:5000"
TIMEOUT = 5  # 较短的超时时间

def test_endpoint(name, endpoint, method="GET", data=None):
    """测试单个接口"""
    url = f"{BACKEND_BASE_URL}{endpoint}"
    
    try:
        print(f"测试 {name}: {method} {endpoint}")
        start_time = time.time()
        
        if method.upper() == "GET":
            response = requests.get(url, timeout=TIMEOUT)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=TIMEOUT)
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"  ✅ 成功 ({response_time:.2f}秒)")
                return True, result, response_time
            except:
                print(f"  ✅ 成功 - 非JSON响应 ({response_time:.2f}秒)")
                return True, {"text": response.text[:100]}, response_time
        else:
            print(f"  ❌ HTTP {response.status_code} ({response_time:.2f}秒)")
            try:
                error = response.json()
                print(f"     错误: {error.get('error', '未知错误')}")
            except:
                print(f"     错误: {response.text[:100]}")
            return False, {}, response_time
            
    except requests.exceptions.Timeout:
        print(f"  ⏱️  超时 (>{TIMEOUT}秒)")
        return False, {}, TIMEOUT
    except requests.exceptions.ConnectionError:
        print(f"  🔌 连接失败")
        return False, {}, 0
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False, {}, 0

def main():
    print("="*60)
    print("快速后端接口测试")
    print("="*60)
    print(f"后端地址: {BACKEND_BASE_URL}")
    print(f"超时时间: {TIMEOUT}秒")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 核心测试用例
    tests = [
        ("健康检查", "/health", "GET", None),
        ("系统状态", "/api/status", "GET", None),
        ("调试状态", "/api/debug/status", "GET", None),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, endpoint, method, data in tests:
        success, result, response_time = test_endpoint(name, endpoint, method, data)
        
        if success:
            passed += 1
            # 显示关键信息
            if isinstance(result, dict):
                if "ai_core_ready" in result:
                    status = "就绪" if result["ai_core_ready"] else "未就绪"
                    print(f"     AI核心: {status}")
                if "status" in result:
                    print(f"     状态: {result['status']}")
                if "ai_core" in result and isinstance(result["ai_core"], dict):
                    ai_info = result["ai_core"]
                    init_status = "已初始化" if ai_info.get("initialized") else "未初始化"
                    print(f"     AI初始化: {init_status}")
        
        print()
    
    # 可选的轻量级分析测试
    print("可选测试 - 轻量级分析:")
    light_data = {
        "type": "text",
        "description": "青花瓷"
    }
    
    # 使用更长的超时时间进行分析测试
    print("测试 轻量级分析: POST /api/analyze")
    try:
        start_time = time.time()
        response = requests.post(f"{BACKEND_BASE_URL}/api/analyze", 
                               json=light_data, timeout=10)  # 10秒超时
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"  ✅ 分析成功 ({response_time:.2f}秒)")
            try:
                result = response.json()
                if "analysis" in result:
                    analysis = result["analysis"]
                    if isinstance(analysis, str):
                        preview = analysis[:100] + "..." if len(analysis) > 100 else analysis
                        print(f"     分析结果预览: {preview}")
            except:
                pass
        else:
            print(f"  ❌ 分析失败: HTTP {response.status_code}")
            try:
                error = response.json()
                print(f"     错误: {error.get('error', '未知错误')}")
            except:
                pass
    except requests.exceptions.Timeout:
        print(f"  ⏱️  分析超时 (>10秒) - 这可能是正常的，AI处理需要时间")
    except Exception as e:
        print(f"  ❌ 分析异常: {e}")
    
    print()
    
    # 结果摘要
    print("="*60)
    print("测试摘要")
    print("="*60)
    print(f"核心测试: {passed}/{total} 通过")
    print(f"通过率: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 所有核心测试通过！后端服务基本功能正常。")
        print("\n💡 提示:")
        print("   - 如果分析功能超时，这可能是正常的，AI处理需要较长时间")
        print("   - 可以通过前端界面进一步测试完整功能")
        return True
    else:
        print("\n⚠️  部分核心测试失败，请检查后端服务配置。")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)