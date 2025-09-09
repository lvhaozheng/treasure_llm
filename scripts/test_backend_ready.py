#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后端接口就绪状态测试脚本
测试后端服务的各个关键接口是否正常工作
"""

import requests
import json
import time
import sys
from typing import Dict, Any, Tuple

# 后端服务配置
BACKEND_BASE_URL = "http://localhost:5000"
TIMEOUT = 10  # 请求超时时间（秒）
RETRY_COUNT = 3  # 重试次数
RETRY_DELAY = 2  # 重试间隔（秒）

# 测试端点配置
TEST_ENDPOINTS = {
    "健康检查": "/health",
    "系统状态": "/api/status",
    "调试状态": "/api/debug/status",
    "图像分析接口": "/api/analyze",  # POST请求，需要特殊处理
}

class BackendTester:
    def __init__(self):
        self.results = {}
        self.session = requests.Session()
        self.session.timeout = TIMEOUT
    
    def test_endpoint(self, name: str, endpoint: str, method: str = "GET", data: Dict = None) -> Tuple[bool, str, Dict]:
        """
        测试单个端点
        
        Args:
            name: 端点名称
            endpoint: 端点路径
            method: HTTP方法
            data: 请求数据（用于POST请求）
        
        Returns:
            (是否成功, 错误信息, 响应数据)
        """
        url = f"{BACKEND_BASE_URL}{endpoint}"
        
        for attempt in range(RETRY_COUNT):
            try:
                print(f"  尝试 {attempt + 1}/{RETRY_COUNT}: {method} {url}")
                
                if method.upper() == "GET":
                    response = self.session.get(url)
                elif method.upper() == "POST":
                    response = self.session.post(url, json=data)
                else:
                    return False, f"不支持的HTTP方法: {method}", {}
                
                # 检查响应状态
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        return True, "成功", response_data
                    except json.JSONDecodeError:
                        return True, "成功（非JSON响应）", {"text": response.text[:200]}
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    if attempt < RETRY_COUNT - 1:
                        print(f"    失败: {error_msg}，{RETRY_DELAY}秒后重试...")
                        time.sleep(RETRY_DELAY)
                        continue
                    return False, error_msg, {}
                    
            except requests.exceptions.ConnectionError:
                error_msg = "连接被拒绝 - 后端服务可能未启动"
                if attempt < RETRY_COUNT - 1:
                    print(f"    失败: {error_msg}，{RETRY_DELAY}秒后重试...")
                    time.sleep(RETRY_DELAY)
                    continue
                return False, error_msg, {}
            except requests.exceptions.Timeout:
                error_msg = f"请求超时（>{TIMEOUT}秒）"
                if attempt < RETRY_COUNT - 1:
                    print(f"    失败: {error_msg}，{RETRY_DELAY}秒后重试...")
                    time.sleep(RETRY_DELAY)
                    continue
                return False, error_msg, {}
            except Exception as e:
                error_msg = f"未知错误: {str(e)}"
                if attempt < RETRY_COUNT - 1:
                    print(f"    失败: {error_msg}，{RETRY_DELAY}秒后重试...")
                    time.sleep(RETRY_DELAY)
                    continue
                return False, error_msg, {}
        
        return False, "所有重试均失败", {}
    
    def test_image_analyze_endpoint(self) -> Tuple[bool, str, Dict]:
        """
        测试图像分析接口（需要特殊处理的POST请求）
        """
        # 创建一个简单的测试请求数据（文本分析）
        test_data = {
            "type": "text",
            "description": "一个青花瓷花瓶，表面有蓝色花纹装饰"
        }
        
        return self.test_endpoint("图像分析接口", "/api/analyze", "POST", test_data)
    
    def run_all_tests(self) -> bool:
        """
        运行所有测试
        
        Returns:
            是否所有测试都通过
        """
        print("="*60)
        print("后端接口就绪状态测试")
        print("="*60)
        print(f"后端服务地址: {BACKEND_BASE_URL}")
        print(f"请求超时时间: {TIMEOUT}秒")
        print(f"重试次数: {RETRY_COUNT}")
        print()
        
        all_passed = True
        
        # 测试GET端点
        for name, endpoint in TEST_ENDPOINTS.items():
            if name == "图像分析接口":
                continue  # 跳过，单独处理
            
            print(f"测试 {name} ({endpoint})...")
            success, message, data = self.test_endpoint(name, endpoint)
            
            self.results[name] = {
                "success": success,
                "message": message,
                "data": data
            }
            
            if success:
                print(f"  ✅ 成功: {message}")
                # 显示关键信息
                if isinstance(data, dict):
                    if "ai_core_ready" in data:
                        print(f"    AI核心状态: {'就绪' if data['ai_core_ready'] else '未就绪'}")
                    if "status" in data:
                        print(f"    服务状态: {data['status']}")
                    if "ai_core" in data and isinstance(data["ai_core"], dict):
                        print(f"    AI核心初始化: {'是' if data['ai_core'].get('initialized') else '否'}")
            else:
                print(f"  ❌ 失败: {message}")
                all_passed = False
            
            print()
        
        # 测试图像分析接口（POST）
        print("测试 图像分析接口 (/api/analyze)...")
        success, message, data = self.test_image_analyze_endpoint()
        
        self.results["图像分析接口"] = {
            "success": success,
            "message": message,
            "data": data
        }
        
        if success:
            print(f"  ✅ 成功: {message}")
        else:
            print(f"  ❌ 失败: {message}")
            # 对于图像分析接口，如果是因为缺少参数而失败，可能是正常的
            if "400" in message or "参数" in message or "image" in message.lower():
                print(f"    注意: 这可能是正常的，因为我们发送的是测试数据")
        
        print()
        
        return all_passed
    
    def print_summary(self):
        """
        打印测试结果摘要
        """
        print("="*60)
        print("测试结果摘要")
        print("="*60)
        
        passed_count = sum(1 for result in self.results.values() if result["success"])
        total_count = len(self.results)
        
        print(f"总测试数: {total_count}")
        print(f"通过数: {passed_count}")
        print(f"失败数: {total_count - passed_count}")
        print(f"通过率: {passed_count/total_count*100:.1f}%")
        print()
        
        # 详细结果
        for name, result in self.results.items():
            status = "✅ 通过" if result["success"] else "❌ 失败"
            print(f"{status} {name}: {result['message']}")
        
        print()
        
        if passed_count == total_count:
            print("🎉 所有测试通过！后端服务已就绪。")
            return True
        else:
            print("⚠️  部分测试失败，请检查后端服务状态。")
            return False

def main():
    """
    主函数
    """
    tester = BackendTester()
    
    try:
        # 运行所有测试
        all_passed = tester.run_all_tests()
        
        # 打印摘要
        final_result = tester.print_summary()
        
        # 设置退出码
        sys.exit(0 if final_result else 1)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()