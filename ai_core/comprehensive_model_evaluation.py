#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合模型评测脚本
对Qwen和SmolVLM2模型进行全面的性能和质量评测
包括响应质量、推理速度、资源占用、准确性等多个维度
"""

import os
import sys
import json
import time
import psutil
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入模型客户端
try:
    from models.client.qwen3_client import Qwen3Client
    QWEN3_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Qwen3客户端导入失败: {e}")
    QWEN3_AVAILABLE = False

try:
    from models.client.smolvlm2_client import SmolVLM2Client
    SMOLVLM2_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  SmolVLM2客户端导入失败: {e}")
    SMOLVLM2_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveModelEvaluator:
    """综合模型评测器"""
    
    def __init__(self):
        self.qwen3_client = None
        self.smolvlm2_client = None
        
        # 评测用例
        self.test_cases = {
            'basic_qa': [
                "请简单介绍一下青花瓷的特点",
                "明代瓷器有哪些主要特征？",
                "如何鉴别古董的真伪？",
                "宋代官窑瓷器的价值如何？",
                "什么是釉里红工艺？"
            ],
            'complex_analysis': [
                "请详细分析唐三彩的制作工艺、历史背景和收藏价值，并说明其在中国陶瓷史上的地位",
                "比较分析明代青花瓷和清代青花瓷在工艺、纹饰、胎质等方面的差异",
                "从考古学角度分析汝窑瓷器的发现意义和研究价值",
                "论述景德镇在中国瓷器发展史上的重要作用和历史地位"
            ],
            'professional_terms': [
                "什么是'开片'现象？它对古瓷鉴定有什么意义？",
                "请解释'胎骨'、'釉色'、'火候'这三个陶瓷术语",
                "'窑变'是什么？它是如何形成的？",
                "什么是'支钉痕'？它能说明什么问题？"
            ],
            'reasoning_tasks': [
                "如果一件瓷器底部有'大清康熙年制'款识，但釉色偏现代，胎质较轻，你会如何判断其真伪？请说明理由。",
                "某收藏家声称拥有一件汝窑天青釉洗，但专家质疑其真实性。请分析可能的鉴定要点。",
                "在拍卖会上看到一件标注为'宋代定窑白瓷'的器物，价格异常便宜，可能存在什么问题？"
            ]
        }
        
        # 评测指标
        self.evaluation_metrics = {
            'response_time': [],
            'response_length': [],
            'memory_usage': [],
            'cpu_usage': [],
            'success_rate': 0,
            'quality_scores': []
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            import torch
            gpu_info = {
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
            }
        except:
            gpu_info = {'cuda_available': False, 'gpu_count': 0, 'gpu_names': []}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'python_version': sys.version,
            'platform': sys.platform,
            'gpu_info': gpu_info
        }
    
    def monitor_resources(self, duration: float = 1.0) -> Dict[str, float]:
        """监控系统资源使用情况"""
        start_time = time.time()
        cpu_samples = []
        memory_samples = []
        
        while time.time() - start_time < duration:
            cpu_samples.append(psutil.cpu_percent())
            memory_samples.append(psutil.virtual_memory().percent)
            time.sleep(0.1)
        
        return {
            'avg_cpu_usage': np.mean(cpu_samples),
            'max_cpu_usage': np.max(cpu_samples),
            'avg_memory_usage': np.mean(memory_samples),
            'max_memory_usage': np.max(memory_samples)
        }
    
    def evaluate_response_quality(self, prompt: str, response: str) -> Dict[str, Any]:
        """评估响应质量"""
        quality_metrics = {
            'length': len(response),
            'word_count': len(response.split()),
            'has_professional_terms': 0,
            'completeness_score': 0,
            'relevance_score': 0,
            'quality_score': 0
        }
        
        # 专业术语检测
        professional_terms = [
            '青花', '釉色', '胎质', '窑口', '款识', '纹饰', '工艺', '朝代',
            '瓷器', '陶器', '官窑', '民窑', '开片', '釉面', '胎骨', '火候',
            '景德镇', '汝窑', '定窑', '钧窑', '哥窑', '唐三彩', '宋瓷'
        ]
        
        term_count = sum(1 for term in professional_terms if term in response)
        quality_metrics['has_professional_terms'] = term_count
        
        # 完整性评分（基于长度和结构）
        if len(response) > 200:
            quality_metrics['completeness_score'] = min(100, len(response) / 5)
        else:
            quality_metrics['completeness_score'] = len(response) / 2
        
        # 相关性评分（基于关键词匹配）
        prompt_keywords = set(prompt.lower().split())
        response_keywords = set(response.lower().split())
        relevance = len(prompt_keywords.intersection(response_keywords)) / len(prompt_keywords) * 100
        quality_metrics['relevance_score'] = relevance
        
        # 综合质量评分
        quality_score = (
            min(quality_metrics['completeness_score'], 40) +  # 完整性 40%
            min(quality_metrics['relevance_score'], 30) +     # 相关性 30%
            min(term_count * 5, 30)                           # 专业性 30%
        )
        quality_metrics['quality_score'] = quality_score
        
        return quality_metrics
    
    def test_model_performance(self, model_client, model_name: str, test_cases: List[str]) -> Dict[str, Any]:
        """测试模型性能"""
        print(f"\n🔍 测试{model_name}模型性能...")
        print("="*60)
        
        results = {
            'model_name': model_name,
            'test_results': [],
            'performance_metrics': {
                'total_tests': len(test_cases),
                'successful_tests': 0,
                'failed_tests': 0,
                'avg_response_time': 0,
                'avg_response_length': 0,
                'avg_quality_score': 0,
                'resource_usage': {
                    'avg_cpu_usage': 0,
                    'avg_memory_usage': 0
                }
            }
        }
        
        response_times = []
        response_lengths = []
        quality_scores = []
        cpu_usages = []
        memory_usages = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 测试 {i}/{len(test_cases)}: {test_case[:50]}...")
            
            try:
                # 开始资源监控
                monitor_thread = threading.Thread(
                    target=lambda: self._monitor_during_inference(cpu_usages, memory_usages)
                )
                monitor_thread.daemon = True
                monitor_thread.start()
                
                # 执行推理
                start_time = time.time()
                
                if hasattr(model_client, 'text_chat'):
                    response = model_client.text_chat(test_case)
                else:
                    response = "模型不支持text_chat方法"
                
                response_time = time.time() - start_time
                
                # 停止监控
                monitor_thread.join(timeout=0.1)
                
                # 评估响应质量
                quality_metrics = self.evaluate_response_quality(test_case, response)
                
                # 记录结果
                test_result = {
                    'test_case': test_case,
                    'response': response,
                    'response_time': response_time,
                    'response_length': len(response),
                    'quality_metrics': quality_metrics,
                    'success': True
                }
                
                results['test_results'].append(test_result)
                results['performance_metrics']['successful_tests'] += 1
                
                # 收集统计数据
                response_times.append(response_time)
                response_lengths.append(len(response))
                quality_scores.append(quality_metrics['quality_score'])
                
                print(f"✅ 成功 - 响应时间: {response_time:.2f}秒, 质量评分: {quality_metrics['quality_score']:.1f}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ 失败: {error_msg}")
                
                results['test_results'].append({
                    'test_case': test_case,
                    'error': error_msg,
                    'success': False
                })
                results['performance_metrics']['failed_tests'] += 1
        
        # 计算平均指标
        if response_times:
            results['performance_metrics']['avg_response_time'] = np.mean(response_times)
            results['performance_metrics']['avg_response_length'] = np.mean(response_lengths)
            results['performance_metrics']['avg_quality_score'] = np.mean(quality_scores)
        
        if cpu_usages:
            results['performance_metrics']['resource_usage']['avg_cpu_usage'] = np.mean(cpu_usages)
        if memory_usages:
            results['performance_metrics']['resource_usage']['avg_memory_usage'] = np.mean(memory_usages)
        
        # 计算成功率
        success_rate = results['performance_metrics']['successful_tests'] / results['performance_metrics']['total_tests']
        results['performance_metrics']['success_rate'] = success_rate
        
        return results
    
    def _monitor_during_inference(self, cpu_usages: List[float], memory_usages: List[float]):
        """推理过程中监控资源使用"""
        start_time = time.time()
        while time.time() - start_time < 30:  # 最多监控30秒
            try:
                cpu_usages.append(psutil.cpu_percent())
                memory_usages.append(psutil.virtual_memory().percent)
                time.sleep(0.5)
            except:
                break
    
    def test_multimodal_performance(self, model_client, test_images: List[str]) -> Dict[str, Any]:
        """测试多模态性能"""
        if not test_images:
            return {'error': '没有找到测试图片'}
        
        print(f"\n🖼️  测试多模态性能（{len(test_images)}张图片）...")
        
        results = {
            'multimodal_tests': [],
            'avg_response_time': 0,
            'success_rate': 0
        }
        
        multimodal_prompts = [
            "请分析这张图片中的古董，描述其特征和可能的价值",
            "这件古董可能来自哪个历史时期？请说明判断依据",
            "从工艺角度分析这件古董的制作特点"
        ]
        
        response_times = []
        successful_tests = 0
        
        for i, image_path in enumerate(test_images[:3]):  # 最多测试3张图片
            try:
                image = Image.open(image_path)
                prompt = multimodal_prompts[i % len(multimodal_prompts)]
                
                print(f"📸 测试图片 {i+1}: {os.path.basename(image_path)}")
                
                start_time = time.time()
                if hasattr(model_client, 'chat_about_antique'):
                    response = model_client.chat_about_antique(image, prompt)
                else:
                    response = "模型不支持多模态功能"
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                successful_tests += 1
                
                results['multimodal_tests'].append({
                    'image_path': image_path,
                    'prompt': prompt,
                    'response': response,
                    'response_time': response_time,
                    'success': True
                })
                
                print(f"✅ 多模态测试成功 - 响应时间: {response_time:.2f}秒")
                
            except Exception as e:
                print(f"❌ 多模态测试失败: {e}")
                results['multimodal_tests'].append({
                    'image_path': image_path,
                    'error': str(e),
                    'success': False
                })
        
        if response_times:
            results['avg_response_time'] = np.mean(response_times)
        results['success_rate'] = successful_tests / len(test_images[:3])
        
        return results
    
    def find_test_images(self) -> List[str]:
        """查找测试图片"""
        test_image_dirs = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo_images'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend', 'uploads'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads'),
            os.path.join(os.path.dirname(__file__), 'test_images')
        ]
        
        test_images = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        
        for image_dir in test_image_dirs:
            if os.path.exists(image_dir):
                for file in os.listdir(image_dir):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        test_images.append(os.path.join(image_dir, file))
                        if len(test_images) >= 5:  # 最多5张
                            break
            if len(test_images) >= 5:
                break
        
        return test_images
    
    def compare_models(self, qwen3_results: Dict, smolvlm2_results: Dict) -> Dict[str, Any]:
        """比较两个模型的性能"""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'performance_comparison': {},
            'quality_comparison': {},
            'recommendation': ''
        }
        
        # 性能比较
        qwen3_perf = qwen3_results.get('performance_metrics', {})
        smolvlm2_perf = smolvlm2_results.get('performance_metrics', {})
        
        comparison['performance_comparison'] = {
            'response_time': {
                'qwen3_avg': qwen3_perf.get('avg_response_time', 0),
                'smolvlm2_avg': smolvlm2_perf.get('avg_response_time', 0),
                'winner': 'qwen3' if qwen3_perf.get('avg_response_time', float('inf')) < smolvlm2_perf.get('avg_response_time', float('inf')) else 'smolvlm2'
            },
            'success_rate': {
                'qwen3_rate': qwen3_perf.get('success_rate', 0),
                'smolvlm2_rate': smolvlm2_perf.get('success_rate', 0),
                'winner': 'qwen3' if qwen3_perf.get('success_rate', 0) > smolvlm2_perf.get('success_rate', 0) else 'smolvlm2'
            },
            'quality_score': {
                'qwen3_avg': qwen3_perf.get('avg_quality_score', 0),
                'smolvlm2_avg': smolvlm2_perf.get('avg_quality_score', 0),
                'winner': 'qwen3' if qwen3_perf.get('avg_quality_score', 0) > smolvlm2_perf.get('avg_quality_score', 0) else 'smolvlm2'
            }
        }
        
        # 生成建议
        qwen3_score = 0
        smolvlm2_score = 0
        
        # 响应时间评分
        if comparison['performance_comparison']['response_time']['winner'] == 'qwen3':
            qwen3_score += 1
        else:
            smolvlm2_score += 1
        
        # 成功率评分
        if comparison['performance_comparison']['success_rate']['winner'] == 'qwen3':
            qwen3_score += 1
        else:
            smolvlm2_score += 1
        
        # 质量评分
        if comparison['performance_comparison']['quality_score']['winner'] == 'qwen3':
            qwen3_score += 1
        else:
            smolvlm2_score += 1
        
        # 多模态能力
        if 'multimodal_results' in smolvlm2_results:
            smolvlm2_score += 2  # 多模态能力加分
        
        if qwen3_score > smolvlm2_score:
            comparison['recommendation'] = 'Qwen3在纯文本任务上表现更好，推荐用于文本分析任务'
        elif smolvlm2_score > qwen3_score:
            comparison['recommendation'] = 'SmolVLM2综合表现更好，特别是支持多模态任务，推荐作为主要模型'
        else:
            comparison['recommendation'] = '两个模型各有优势，建议根据具体任务选择：纯文本用Qwen3，图文结合用SmolVLM2'
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """保存评测结果"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'model_evaluation_results_{timestamp}.json'
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 评测结果已保存到: {filepath}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def run_comprehensive_evaluation(self):
        """运行综合评测"""
        print("🚀 开始综合模型评测")
        print("="*80)
        
        # 获取系统信息
        system_info = self.get_system_info()
        print(f"💻 系统信息: CPU核心数={system_info['cpu_count']}, 内存={system_info['memory_total']:.1f}GB")
        print(f"🖥️  GPU信息: CUDA可用={system_info['gpu_info']['cuda_available']}, GPU数量={system_info['gpu_info']['gpu_count']}")
        
        evaluation_results = {
            'system_info': system_info,
            'qwen3_available': QWEN3_AVAILABLE,
            'smolvlm2_available': SMOLVLM2_AVAILABLE
        }
        
        # 测试Qwen3
        if QWEN3_AVAILABLE:
            try:
                print("\n🔧 初始化Qwen3模型...")
                self.qwen3_client = Qwen3Client(device="auto", quantization="4bit")
                
                # 合并所有测试用例
                all_test_cases = []
                for category, cases in self.test_cases.items():
                    all_test_cases.extend(cases)
                
                qwen3_results = self.test_model_performance(self.qwen3_client, "Qwen3", all_test_cases)
                evaluation_results['qwen3_results'] = qwen3_results
                
            except Exception as e:
                print(f"❌ Qwen3测试失败: {e}")
                evaluation_results['qwen3_results'] = {'error': str(e)}
        else:
            print("\n⚠️  跳过Qwen3测试（不可用）")
        
        # 测试SmolVLM2
        if SMOLVLM2_AVAILABLE:
            try:
                print("\n🔧 初始化SmolVLM2模型...")
                self.smolvlm2_client = SmolVLM2Client(
                    device="auto",
                    max_tokens=512,
                    temperature=0.7
                )
                
                # 文本测试
                all_test_cases = []
                for category, cases in self.test_cases.items():
                    all_test_cases.extend(cases)
                
                smolvlm2_results = self.test_model_performance(self.smolvlm2_client, "SmolVLM2", all_test_cases)
                
                # 多模态测试
                test_images = self.find_test_images()
                if test_images:
                    multimodal_results = self.test_multimodal_performance(self.smolvlm2_client, test_images)
                    smolvlm2_results['multimodal_results'] = multimodal_results
                
                evaluation_results['smolvlm2_results'] = smolvlm2_results
                
            except Exception as e:
                print(f"❌ SmolVLM2测试失败: {e}")
                evaluation_results['smolvlm2_results'] = {'error': str(e)}
        else:
            print("\n⚠️  跳过SmolVLM2测试（不可用）")
        
        # 模型比较
        if (QWEN3_AVAILABLE and 'qwen3_results' in evaluation_results and 'error' not in evaluation_results['qwen3_results'] and
            SMOLVLM2_AVAILABLE and 'smolvlm2_results' in evaluation_results and 'error' not in evaluation_results['smolvlm2_results']):
            
            comparison_results = self.compare_models(
                evaluation_results['qwen3_results'],
                evaluation_results['smolvlm2_results']
            )
            evaluation_results['comparison'] = comparison_results
        
        # 打印总结
        self.print_evaluation_summary(evaluation_results)
        
        # 保存结果
        self.save_results(evaluation_results)
        
        return evaluation_results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """打印评测总结"""
        print("\n" + "="*80)
        print("🎯 综合评测总结")
        print("="*80)
        
        # Qwen3结果
        if 'qwen3_results' in results and 'error' not in results['qwen3_results']:
            qwen3_perf = results['qwen3_results']['performance_metrics']
            print(f"\n📊 Qwen3性能指标:")
            print(f"  ✅ 成功率: {qwen3_perf['success_rate']:.1%}")
            print(f"  ⏱️  平均响应时间: {qwen3_perf['avg_response_time']:.2f}秒")
            print(f"  📄 平均响应长度: {qwen3_perf['avg_response_length']:.0f}字符")
            print(f"  🎯 平均质量评分: {qwen3_perf['avg_quality_score']:.1f}/100")
            print(f"  💻 平均CPU使用率: {qwen3_perf['resource_usage']['avg_cpu_usage']:.1f}%")
        
        # SmolVLM2结果
        if 'smolvlm2_results' in results and 'error' not in results['smolvlm2_results']:
            smolvlm2_perf = results['smolvlm2_results']['performance_metrics']
            print(f"\n📊 SmolVLM2性能指标:")
            print(f"  ✅ 文本成功率: {smolvlm2_perf['success_rate']:.1%}")
            print(f"  ⏱️  文本平均响应时间: {smolvlm2_perf['avg_response_time']:.2f}秒")
            print(f"  📄 文本平均响应长度: {smolvlm2_perf['avg_response_length']:.0f}字符")
            print(f"  🎯 文本平均质量评分: {smolvlm2_perf['avg_quality_score']:.1f}/100")
            print(f"  💻 平均CPU使用率: {smolvlm2_perf['resource_usage']['avg_cpu_usage']:.1f}%")
            
            if 'multimodal_results' in results['smolvlm2_results']:
                mm_results = results['smolvlm2_results']['multimodal_results']
                print(f"  🖼️  多模态成功率: {mm_results['success_rate']:.1%}")
                print(f"  ⏱️  多模态平均响应时间: {mm_results['avg_response_time']:.2f}秒")
        
        # 比较结果
        if 'comparison' in results:
            comparison = results['comparison']
            print(f"\n🏆 模型比较结果:")
            perf_comp = comparison['performance_comparison']
            print(f"  ⚡ 响应速度优胜者: {perf_comp['response_time']['winner'].upper()}")
            print(f"  ✅ 成功率优胜者: {perf_comp['success_rate']['winner'].upper()}")
            print(f"  🎯 质量评分优胜者: {perf_comp['quality_score']['winner'].upper()}")
            print(f"\n💡 推荐: {comparison['recommendation']}")
        
        print("\n✅ 综合评测完成！")


def main():
    """主函数"""
    try:
        evaluator = ComprehensiveModelEvaluator()
        results = evaluator.run_comprehensive_evaluation()
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠️  评测被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 评测过程中出现错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())