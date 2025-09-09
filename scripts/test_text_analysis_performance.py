#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本分析性能测试脚本
测试优化后的文本分析功能性能
"""

import requests
import time
import json
from datetime import datetime

# 配置
BASE_URL = "http://localhost:5000"
TEST_TEXTS = [
    "青花瓷",
    "明代花瓶",
    "清朝瓷器",
    "古代陶瓷",
    "宋代青瓷"
]

def test_text_analysis_performance():
    """测试文本分析性能"""
    print("=" * 60)
    print("文本分析性能测试")
    print("=" * 60)
    print(f"后端地址: {BASE_URL}")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    for i, text in enumerate(TEST_TEXTS, 1):
        print(f"测试 {i}/{len(TEST_TEXTS)}: 分析文本 '{text}'")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 发送分析请求
            response = requests.post(
                f"{BASE_URL}/api/analyze",
                json={"text": text},
                timeout=120  # 2分钟超时
            )
            
            # 记录结束时间
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                from_cache = data.get('from_cache', False)
                cache_status = "(来自缓存)" if from_cache else "(新分析)"
                
                print(f"  ✅ 成功 ({duration:.2f}秒) {cache_status}")
                
                # 显示部分结果
                if 'analysis' in data:
                    analysis = data['analysis']
                    if isinstance(analysis, dict):
                        antique_type = analysis.get('type', '未知')
                        print(f"     类型: {antique_type}")
                
                results.append({
                    'text': text,
                    'duration': duration,
                    'success': True,
                    'from_cache': from_cache,
                    'status_code': response.status_code
                })
            else:
                print(f"  ❌ 失败 ({duration:.2f}秒) - HTTP {response.status_code}")
                results.append({
                    'text': text,
                    'duration': duration,
                    'success': False,
                    'from_cache': False,
                    'status_code': response.status_code,
                    'error': response.text
                })
                
        except requests.exceptions.Timeout:
            duration = time.time() - start_time
            print(f"  ⏱️  超时 ({duration:.2f}秒)")
            results.append({
                'text': text,
                'duration': duration,
                'success': False,
                'from_cache': False,
                'error': 'timeout'
            })
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"  ❌ 错误 ({duration:.2f}秒): {str(e)}")
            results.append({
                'text': text,
                'duration': duration,
                'success': False,
                'from_cache': False,
                'error': str(e)
            })
        
        print()
        
        # 短暂延迟避免过载
        if i < len(TEST_TEXTS):
            time.sleep(1)
    
    # 分析结果
    print("=" * 60)
    print("性能分析结果")
    print("=" * 60)
    
    successful_tests = [r for r in results if r['success']]
    cached_tests = [r for r in successful_tests if r['from_cache']]
    new_analysis_tests = [r for r in successful_tests if not r['from_cache']]
    
    print(f"总测试数: {len(results)}")
    print(f"成功测试: {len(successful_tests)}")
    print(f"缓存命中: {len(cached_tests)}")
    print(f"新分析: {len(new_analysis_tests)}")
    print()
    
    if successful_tests:
        durations = [r['duration'] for r in successful_tests]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        print(f"平均响应时间: {avg_duration:.2f}秒")
        print(f"最快响应时间: {min_duration:.2f}秒")
        print(f"最慢响应时间: {max_duration:.2f}秒")
        print()
        
        if cached_tests:
            cached_durations = [r['duration'] for r in cached_tests]
            avg_cached = sum(cached_durations) / len(cached_durations)
            print(f"缓存平均响应时间: {avg_cached:.2f}秒")
        
        if new_analysis_tests:
            new_durations = [r['duration'] for r in new_analysis_tests]
            avg_new = sum(new_durations) / len(new_durations)
            print(f"新分析平均响应时间: {avg_new:.2f}秒")
    
    print()
    
    # 性能评估
    if successful_tests:
        success_rate = len(successful_tests) / len(results) * 100
        cache_hit_rate = len(cached_tests) / len(successful_tests) * 100 if successful_tests else 0
        
        print("性能评估:")
        print(f"  成功率: {success_rate:.1f}%")
        print(f"  缓存命中率: {cache_hit_rate:.1f}%")
        
        if new_analysis_tests:
            avg_new_time = sum(r['duration'] for r in new_analysis_tests) / len(new_analysis_tests)
            if avg_new_time < 30:
                print(f"  新分析性能: 优秀 (平均{avg_new_time:.1f}秒)")
            elif avg_new_time < 60:
                print(f"  新分析性能: 良好 (平均{avg_new_time:.1f}秒)")
            else:
                print(f"  新分析性能: 需要优化 (平均{avg_new_time:.1f}秒)")
    
    # 保存详细结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"performance_test_results_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_config': {
                'base_url': BASE_URL,
                'test_texts': TEST_TEXTS,
                'timeout': 120
            },
            'results': results,
            'summary': {
                'total_tests': len(results),
                'successful_tests': len(successful_tests),
                'cached_tests': len(cached_tests),
                'new_analysis_tests': len(new_analysis_tests),
                'success_rate': len(successful_tests) / len(results) * 100 if results else 0,
                'cache_hit_rate': len(cached_tests) / len(successful_tests) * 100 if successful_tests else 0
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {result_file}")
    
    return results

if __name__ == "__main__":
    try:
        results = test_text_analysis_performance()
        
        # 简单的性能判断
        successful = [r for r in results if r['success']]
        if successful:
            avg_time = sum(r['duration'] for r in successful) / len(successful)
            if avg_time < 30:
                print("\n🎉 性能测试通过！文本分析响应时间良好。")
            else:
                print(f"\n⚠️  性能需要进一步优化，平均响应时间: {avg_time:.1f}秒")
        else:
            print("\n❌ 所有测试都失败了，请检查后端服务状态。")
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")