#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面的后端接口测试脚本
测试后端服务的各个关键接口，包括错误处理和性能测试
"""

import requests
import json
import time
import sys
import os
from typing import Dict, Any, Tuple, List
from datetime import datetime

# 后端服务配置
BACKEND_BASE_URL = "http://localhost:5000"
TIMEOUT = 15  # 请求超时时间（秒）
RETRY_COUNT = 3  # 重试次数
RETRY_DELAY = 2  # 重试间隔（秒）

class ComprehensiveBackendTester:
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def make_request(self, method: str, endpoint: str, data: Dict = None, 
                    headers: Dict = None, files: Dict = None) -> Tuple[bool, str, Dict, float]:
        """
        发送HTTP请求
        
        Returns:
            (success, message, response_data, response_time)
        """
        url = f"{BACKEND_BASE_URL}{endpoint}"
        
        for attempt in range(RETRY_COUNT):
            try:
                print(f"  尝试 {attempt + 1}/{RETRY_COUNT}: {method} {url}")
                
                start_time = time.time()
                
                if method.upper() == "GET":
                    response = requests.get(url, timeout=TIMEOUT, headers=headers)
                elif method.upper() == "POST":
                    if files:
                        response = requests.post(url, files=files, data=data, 
                                               timeout=TIMEOUT, headers=headers)
                    else:
                        response = requests.post(url, json=data, timeout=TIMEOUT, 
                                               headers=headers)
                else:
                    return False, f"不支持的HTTP方法: {method}", {}, 0
                
                response_time = time.time() - start_time
                
                # 检查响应状态
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        return True, "成功", response_data, response_time
                    except json.JSONDecodeError:
                        return True, "成功（非JSON响应）", {"text": response.text}, response_time
                else:
                    try:
                        error_data = response.json()
                        return False, f"HTTP {response.status_code}: {error_data.get('error', '未知错误')}", error_data, response_time
                    except json.JSONDecodeError:
                        return False, f"HTTP {response.status_code}: {response.text}", {}, response_time
                        
            except requests.exceptions.Timeout:
                print(f"    请求超时 (>{TIMEOUT}秒)")
                if attempt < RETRY_COUNT - 1:
                    print(f"    等待 {RETRY_DELAY} 秒后重试...")
                    time.sleep(RETRY_DELAY)
                else:
                    return False, f"请求超时 (>{TIMEOUT}秒)", {}, 0
                    
            except requests.exceptions.ConnectionError:
                print(f"    连接错误")
                if attempt < RETRY_COUNT - 1:
                    print(f"    等待 {RETRY_DELAY} 秒后重试...")
                    time.sleep(RETRY_DELAY)
                else:
                    return False, "连接失败，请检查后端服务是否启动", {}, 0
                    
            except Exception as e:
                print(f"    请求异常: {e}")
                if attempt < RETRY_COUNT - 1:
                    print(f"    等待 {RETRY_DELAY} 秒后重试...")
                    time.sleep(RETRY_DELAY)
                else:
                    return False, f"请求异常: {str(e)}", {}, 0
        
        return False, "所有重试都失败", {}, 0
    
    def test_health_check(self) -> Tuple[bool, str, Dict, float]:
        """测试健康检查接口"""
        return self.make_request("GET", "/health")
    
    def test_system_status(self) -> Tuple[bool, str, Dict, float]:
        """测试系统状态接口"""
        return self.make_request("GET", "/api/status")
    
    def test_debug_status(self) -> Tuple[bool, str, Dict, float]:
        """测试调试状态接口"""
        return self.make_request("GET", "/api/debug/status")
    
    def test_text_analysis(self) -> Tuple[bool, str, Dict, float]:
        """测试文本分析接口"""
        test_data = {
            "type": "text",
            "description": "一个明代青花瓷碗，碗口直径约15厘米，高度约8厘米，表面绘有精美的龙纹图案，釉色青白，底部有'大明宣德年制'款识。"
        }
        return self.make_request("POST", "/api/analyze", test_data)
    
    def test_invalid_analysis_request(self) -> Tuple[bool, str, Dict, float]:
        """测试无效的分析请求（错误处理）"""
        test_data = {
            "invalid_field": "test"
        }
        return self.make_request("POST", "/api/analyze", test_data)
    
    def test_model_status(self) -> Tuple[bool, str, Dict, float]:
        """测试模型状态接口"""
        return self.make_request("GET", "/api/model/status")
    
    def test_chat_interface(self) -> Tuple[bool, str, Dict, float]:
        """测试聊天接口"""
        test_data = {
            "message": "你好，请介绍一下你的功能。"
        }
        return self.make_request("POST", "/api/chat", test_data)
    
    def test_nonexistent_endpoint(self) -> Tuple[bool, str, Dict, float]:
        """测试不存在的接口（404错误处理）"""
        return self.make_request("GET", "/api/nonexistent")
    
    def run_performance_test(self, endpoint: str, method: str = "GET", 
                           data: Dict = None, iterations: int = 5) -> Dict:
        """运行性能测试"""
        print(f"\n性能测试: {method} {endpoint} ({iterations}次请求)")
        
        response_times = []
        success_count = 0
        
        for i in range(iterations):
            success, message, response_data, response_time = self.make_request(method, endpoint, data)
            response_times.append(response_time)
            if success:
                success_count += 1
            print(f"  第{i+1}次: {response_time:.3f}秒 {'✅' if success else '❌'}")
        
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        success_rate = (success_count / iterations) * 100
        
        performance_result = {
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "success_rate": success_rate,
            "total_requests": iterations
        }
        
        print(f"  平均响应时间: {avg_time:.3f}秒")
        print(f"  最快响应时间: {min_time:.3f}秒")
        print(f"  最慢响应时间: {max_time:.3f}秒")
        print(f"  成功率: {success_rate:.1f}%")
        
        return performance_result
    
    def run_all_tests(self) -> bool:
        """
        运行所有测试
        
        Returns:
            是否所有关键测试都通过
        """
        print("="*80)
        print("全面后端接口测试")
        print("="*80)
        print(f"后端服务地址: {BACKEND_BASE_URL}")
        print(f"请求超时时间: {TIMEOUT}秒")
        print(f"重试次数: {RETRY_COUNT}")
        print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.start_time = time.time()
        
        # 定义测试用例
        test_cases = [
            ("健康检查", self.test_health_check, True),  # 关键测试
            ("系统状态", self.test_system_status, True),  # 关键测试
            ("调试状态", self.test_debug_status, True),   # 关键测试
            ("文本分析", self.test_text_analysis, True),  # 关键测试
            ("模型状态", self.test_model_status, False), # 非关键测试
            ("聊天接口", self.test_chat_interface, False), # 非关键测试
            ("无效请求处理", self.test_invalid_analysis_request, False), # 错误处理测试
            ("404错误处理", self.test_nonexistent_endpoint, False), # 错误处理测试
        ]
        
        critical_passed = 0
        critical_total = 0
        all_passed = 0
        total_tests = len(test_cases)
        
        # 运行功能测试
        for test_name, test_func, is_critical in test_cases:
            print(f"测试 {test_name}...")
            
            success, message, data, response_time = test_func()
            
            self.results[test_name] = {
                "success": success,
                "message": message,
                "data": data,
                "response_time": response_time,
                "is_critical": is_critical
            }
            
            if is_critical:
                critical_total += 1
                if success:
                    critical_passed += 1
            
            if success:
                all_passed += 1
                print(f"  ✅ 成功: {message} ({response_time:.3f}秒)")
                
                # 显示关键信息
                if isinstance(data, dict):
                    if "ai_core_ready" in data:
                        print(f"    AI核心状态: {'就绪' if data['ai_core_ready'] else '未就绪'}")
                    if "status" in data:
                        print(f"    服务状态: {data['status']}")
                    if "ai_core" in data and isinstance(data["ai_core"], dict):
                        ai_info = data["ai_core"]
                        print(f"    AI核心初始化: {'是' if ai_info.get('initialized') else '否'}")
                        if "system_status" in ai_info:
                            sys_status = ai_info["system_status"]
                            if isinstance(sys_status, dict) and "model_loaded" in sys_status:
                                print(f"    模型加载状态: {'已加载' if sys_status['model_loaded'] else '未加载'}")
            else:
                print(f"  ❌ 失败: {message} ({response_time:.3f}秒)")
                if is_critical:
                    print(f"    ⚠️  这是关键测试失败！")
            
            print()
        
        # 运行性能测试
        print("\n" + "="*50)
        print("性能测试")
        print("="*50)
        
        performance_tests = [
            ("/health", "GET", None),
            ("/api/status", "GET", None),
            ("/api/analyze", "POST", {"type": "text", "description": "测试描述"})
        ]
        
        for endpoint, method, data in performance_tests:
            perf_result = self.run_performance_test(endpoint, method, data, 3)
            self.results[f"性能_{endpoint}"] = perf_result
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        # 输出测试摘要
        print("\n" + "="*80)
        print("测试结果摘要")
        print("="*80)
        print(f"测试总时间: {total_time:.2f}秒")
        print(f"总测试数: {total_tests}")
        print(f"通过数: {all_passed}")
        print(f"失败数: {total_tests - all_passed}")
        print(f"总通过率: {(all_passed / total_tests) * 100:.1f}%")
        print()
        print(f"关键测试: {critical_passed}/{critical_total} 通过")
        print(f"关键测试通过率: {(critical_passed / critical_total) * 100:.1f}%")
        print()
        
        # 详细结果
        for test_name, result in self.results.items():
            if test_name.startswith("性能_"):
                continue  # 跳过性能测试结果
                
            status_icon = "✅" if result["success"] else "❌"
            critical_mark = " [关键]" if result["is_critical"] else ""
            print(f"{status_icon} {test_name}{critical_mark}: {result['message']}")
        
        print()
        
        # 判断整体状态
        if critical_passed == critical_total:
            print("🎉 所有关键测试通过！后端服务已就绪。")
            return True
        else:
            print("⚠️  部分关键测试失败，后端服务可能存在问题。")
            return False
    
    def save_results(self, filename: str = None):
        """保存测试结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backend_test_results_{timestamp}.json"
        
        results_data = {
            "test_time": datetime.now().isoformat(),
            "backend_url": BACKEND_BASE_URL,
            "total_time": self.end_time - self.start_time if self.end_time and self.start_time else 0,
            "results": self.results
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            print(f"测试结果已保存到: {filename}")
        except Exception as e:
            print(f"保存测试结果失败: {e}")

def main():
    """主函数"""
    tester = ComprehensiveBackendTester()
    
    try:
        success = tester.run_all_tests()
        
        # 保存结果
        tester.save_results()
        
        # 返回适当的退出代码
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(2)
    except Exception as e:
        print(f"\n测试过程中发生异常: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()