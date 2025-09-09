#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen和SmolVLM2模型效果评测脚本
支持多维度评测：响应质量、推理速度、资源占用、准确性等
"""

import os
import sys
import json
import time
import logging
import psutil
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入模型客户端
from models.client.qwen3_client import Qwen3Client
from models.client.smolvlm2_client import SmolVLM2Client
from utils.logger_config import get_ai_logger

# 配置日志
logger = get_ai_logger('model_evaluation')

class ModelEvaluator:
    """模型评测器"""
    
    def __init__(self, output_dir: str = None):
        """
        初始化模型评测器
        
        Args:
            output_dir: 输出目录，默认为当前目录
        """
        self.output_dir = output_dir or os.getcwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 模型客户端
        self.qwen3_client = None
        self.smolvlm2_client = None
        
        # 评测结果
        self.evaluation_results = {
            'timestamp': self.timestamp,
            'system_info': self._get_system_info(),
            'qwen3_results': {},
            'smolvlm2_results': {},
            'comparison': {}
        }
        
        # 测试数据
        self.test_cases = self._prepare_test_cases()
        
        logger.info(f"模型评测器初始化完成，输出目录: {self.output_dir}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total // (1024**3),  # GB
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'python_version': sys.version,
            'torch_version': torch.__version__
        }
    
    def _prepare_test_cases(self) -> Dict[str, List[Dict]]:
        """准备测试用例"""
        return {
            'text_analysis': [
                {
                    'id': 'text_001',
                    'prompt': '请分析这件青花瓷器的特征和价值',
                    'expected_keywords': ['青花', '瓷器', '价值', '特征'],
                    'difficulty': 'easy'
                },
                {
                    'id': 'text_002', 
                    'prompt': '这是一件明代宣德年间的青花缠枝莲纹梅瓶，请详细分析其工艺特点、历史价值和市场前景',
                    'expected_keywords': ['明代', '宣德', '青花', '缠枝莲', '梅瓶', '工艺', '历史', '市场'],
                    'difficulty': 'hard'
                },
                {
                    'id': 'text_003',
                    'prompt': '请比较唐三彩和宋代定窑白瓷的区别',
                    'expected_keywords': ['唐三彩', '宋代', '定窑', '白瓷', '区别', '比较'],
                    'difficulty': 'medium'
                }
            ],
            'multimodal_analysis': [
                {
                    'id': 'mm_001',
                    'image_path': None,  # 将在运行时查找测试图片
                    'prompt': '请分析这件古董的年代、材质和价值',
                    'expected_keywords': ['年代', '材质', '价值'],
                    'difficulty': 'medium'
                }
            ],
            'reasoning_tasks': [
                {
                    'id': 'reason_001',
                    'prompt': '如果一件瓷器底部有"大清康熙年制"款识，但釉色偏现代，胎质较轻，你会如何判断其真伪？',
                    'expected_keywords': ['款识', '釉色', '胎质', '真伪', '判断'],
                    'difficulty': 'hard'
                }
            ]
        }
    
    def initialize_models(self) -> bool:
        """初始化模型"""
        logger.info("开始初始化模型...")
        
        success = True
        
        # 初始化Qwen3
        try:
            logger.info("初始化Qwen3模型...")
            self.qwen3_client = Qwen3Client(
                device="auto",
                quantization="4bit"
            )
            logger.info("✅ Qwen3模型初始化成功")
        except Exception as e:
            logger.error(f"❌ Qwen3模型初始化失败: {e}")
            success = False
        
        # 初始化SmolVLM2
        try:
            logger.info("初始化SmolVLM2模型...")
            self.smolvlm2_client = SmolVLM2Client(
                device="auto",
                max_tokens=512,
                temperature=0.7
            )
            logger.info("✅ SmolVLM2模型初始化成功")
        except Exception as e:
            logger.error(f"❌ SmolVLM2模型初始化失败: {e}")
            success = False
        
        return success
    
    def _measure_performance(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """测量函数执行性能"""
        # 记录初始状态
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        initial_cpu = process.cpu_percent()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        
        # 执行函数
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = f"Error: {str(e)}"
            success = False
        end_time = time.time()
        
        # 记录最终状态
        final_memory = process.memory_info().rss / (1024**2)  # MB
        final_cpu = process.cpu_percent()
        
        performance_metrics = {
            'execution_time': end_time - start_time,
            'memory_usage': final_memory - initial_memory,
            'cpu_usage': final_cpu,
            'success': success
        }
        
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
            peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            performance_metrics.update({
                'gpu_memory_usage': final_gpu_memory - initial_gpu_memory,
                'peak_gpu_memory': peak_gpu_memory
            })
        
        return result, performance_metrics
    
    def _evaluate_response_quality(self, response: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """评估响应质量"""
        if isinstance(response, str) and response.startswith("Error:"):
            return {
                'keyword_coverage': 0.0,
                'response_length': 0,
                'has_error': True,
                'error_message': response
            }
        
        # 关键词覆盖率
        response_lower = response.lower()
        covered_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
        keyword_coverage = len(covered_keywords) / len(expected_keywords) if expected_keywords else 0.0
        
        # 响应长度
        response_length = len(response)
        
        # 简单的质量评分（基于长度和关键词覆盖率）
        length_score = min(response_length / 200, 1.0)  # 200字符为满分
        quality_score = (keyword_coverage * 0.7 + length_score * 0.3)
        
        return {
            'keyword_coverage': keyword_coverage,
            'covered_keywords': covered_keywords,
            'response_length': response_length,
            'quality_score': quality_score,
            'has_error': False
        }
    
    def evaluate_qwen3(self) -> Dict[str, Any]:
        """评测Qwen3模型"""
        logger.info("开始评测Qwen3模型...")
        
        if not self.qwen3_client:
            return {'error': 'Qwen3客户端未初始化'}
        
        results = {
            'model_info': self.qwen3_client.get_model_info(),
            'text_analysis': [],
            'reasoning_tasks': [],
            'performance_summary': {}
        }
        
        all_performance_metrics = []
        
        # 文本分析测试
        logger.info("执行文本分析测试...")
        for test_case in self.test_cases['text_analysis']:
            logger.info(f"测试用例: {test_case['id']}")
            
            response, performance = self._measure_performance(
                self.qwen3_client.text_chat,
                test_case['prompt'],
                512
            )
            
            quality = self._evaluate_response_quality(
                response, 
                test_case['expected_keywords']
            )
            
            test_result = {
                'test_id': test_case['id'],
                'difficulty': test_case['difficulty'],
                'prompt': test_case['prompt'],
                'response': response,
                'performance': performance,
                'quality': quality
            }
            
            results['text_analysis'].append(test_result)
            all_performance_metrics.append(performance)
        
        # 推理任务测试
        logger.info("执行推理任务测试...")
        for test_case in self.test_cases['reasoning_tasks']:
            logger.info(f"测试用例: {test_case['id']}")
            
            response, performance = self._measure_performance(
                self.qwen3_client.text_chat,
                test_case['prompt'],
                512
            )
            
            quality = self._evaluate_response_quality(
                response,
                test_case['expected_keywords']
            )
            
            test_result = {
                'test_id': test_case['id'],
                'difficulty': test_case['difficulty'],
                'prompt': test_case['prompt'],
                'response': response,
                'performance': performance,
                'quality': quality
            }
            
            results['reasoning_tasks'].append(test_result)
            all_performance_metrics.append(performance)
        
        # 计算性能摘要
        if all_performance_metrics:
            results['performance_summary'] = self._calculate_performance_summary(all_performance_metrics)
        
        logger.info("Qwen3模型评测完成")
        return results
    
    def evaluate_smolvlm2(self) -> Dict[str, Any]:
        """评测SmolVLM2模型"""
        logger.info("开始评测SmolVLM2模型...")
        
        if not self.smolvlm2_client:
            return {'error': 'SmolVLM2客户端未初始化'}
        
        results = {
            'model_info': self.smolvlm2_client.get_model_info(),
            'text_analysis': [],
            'multimodal_analysis': [],
            'reasoning_tasks': [],
            'performance_summary': {}
        }
        
        all_performance_metrics = []
        
        # 文本分析测试
        logger.info("执行文本分析测试...")
        for test_case in self.test_cases['text_analysis']:
            logger.info(f"测试用例: {test_case['id']}")
            
            response, performance = self._measure_performance(
                self.smolvlm2_client.text_chat,
                test_case['prompt']
            )
            
            quality = self._evaluate_response_quality(
                response,
                test_case['expected_keywords']
            )
            
            test_result = {
                'test_id': test_case['id'],
                'difficulty': test_case['difficulty'],
                'prompt': test_case['prompt'],
                'response': response,
                'performance': performance,
                'quality': quality
            }
            
            results['text_analysis'].append(test_result)
            all_performance_metrics.append(performance)
        
        # 多模态分析测试（如果有测试图片）
        logger.info("执行多模态分析测试...")
        test_images = self._find_test_images()
        
        for test_case in self.test_cases['multimodal_analysis']:
            if test_images:
                test_image = test_images[0]  # 使用第一张测试图片
                logger.info(f"测试用例: {test_case['id']} with image: {test_image}")
                
                try:
                    image = Image.open(test_image)
                    response, performance = self._measure_performance(
                        self.smolvlm2_client.chat_about_antique,
                        image,
                        test_case['prompt']
                    )
                    
                    quality = self._evaluate_response_quality(
                        response,
                        test_case['expected_keywords']
                    )
                    
                    test_result = {
                        'test_id': test_case['id'],
                        'difficulty': test_case['difficulty'],
                        'image_path': test_image,
                        'prompt': test_case['prompt'],
                        'response': response,
                        'performance': performance,
                        'quality': quality
                    }
                    
                    results['multimodal_analysis'].append(test_result)
                    all_performance_metrics.append(performance)
                    
                except Exception as e:
                    logger.error(f"多模态测试失败: {e}")
                    results['multimodal_analysis'].append({
                        'test_id': test_case['id'],
                        'error': str(e)
                    })
            else:
                logger.warning("未找到测试图片，跳过多模态测试")
        
        # 推理任务测试
        logger.info("执行推理任务测试...")
        for test_case in self.test_cases['reasoning_tasks']:
            logger.info(f"测试用例: {test_case['id']}")
            
            response, performance = self._measure_performance(
                self.smolvlm2_client.text_chat,
                test_case['prompt']
            )
            
            quality = self._evaluate_response_quality(
                response,
                test_case['expected_keywords']
            )
            
            test_result = {
                'test_id': test_case['id'],
                'difficulty': test_case['difficulty'],
                'prompt': test_case['prompt'],
                'response': response,
                'performance': performance,
                'quality': quality
            }
            
            results['reasoning_tasks'].append(test_result)
            all_performance_metrics.append(performance)
        
        # 计算性能摘要
        if all_performance_metrics:
            results['performance_summary'] = self._calculate_performance_summary(all_performance_metrics)
        
        logger.info("SmolVLM2模型评测完成")
        return results
    
    def _find_test_images(self) -> List[str]:
        """查找测试图片"""
        test_image_dirs = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo_images'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend', 'uploads'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
        ]
        
        test_images = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        
        for image_dir in test_image_dirs:
            if os.path.exists(image_dir):
                for file in os.listdir(image_dir):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        test_images.append(os.path.join(image_dir, file))
        
        return test_images[:3]  # 最多返回3张测试图片
    
    def _calculate_performance_summary(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """计算性能摘要"""
        if not metrics_list:
            return {}
        
        # 过滤成功的测试
        successful_metrics = [m for m in metrics_list if m.get('success', True)]
        
        if not successful_metrics:
            return {'success_rate': 0.0}
        
        summary = {
            'success_rate': len(successful_metrics) / len(metrics_list),
            'avg_execution_time': np.mean([m['execution_time'] for m in successful_metrics]),
            'max_execution_time': np.max([m['execution_time'] for m in successful_metrics]),
            'min_execution_time': np.min([m['execution_time'] for m in successful_metrics]),
            'avg_memory_usage': np.mean([m.get('memory_usage', 0) for m in successful_metrics]),
            'avg_cpu_usage': np.mean([m.get('cpu_usage', 0) for m in successful_metrics])
        }
        
        # GPU相关指标
        gpu_metrics = [m for m in successful_metrics if 'gpu_memory_usage' in m]
        if gpu_metrics:
            summary.update({
                'avg_gpu_memory_usage': np.mean([m['gpu_memory_usage'] for m in gpu_metrics]),
                'max_peak_gpu_memory': np.max([m.get('peak_gpu_memory', 0) for m in gpu_metrics])
            })
        
        return summary
    
    def compare_models(self) -> Dict[str, Any]:
        """比较两个模型的性能"""
        logger.info("开始模型比较分析...")
        
        qwen3_results = self.evaluation_results.get('qwen3_results', {})
        smolvlm2_results = self.evaluation_results.get('smolvlm2_results', {})
        
        comparison = {
            'performance_comparison': {},
            'quality_comparison': {},
            'capability_comparison': {},
            'recommendations': []
        }
        
        # 性能比较
        qwen3_perf = qwen3_results.get('performance_summary', {})
        smolvlm2_perf = smolvlm2_results.get('performance_summary', {})
        
        if qwen3_perf and smolvlm2_perf:
            comparison['performance_comparison'] = {
                'speed': {
                    'qwen3_avg_time': qwen3_perf.get('avg_execution_time', 0),
                    'smolvlm2_avg_time': smolvlm2_perf.get('avg_execution_time', 0),
                    'winner': 'qwen3' if qwen3_perf.get('avg_execution_time', float('inf')) < smolvlm2_perf.get('avg_execution_time', float('inf')) else 'smolvlm2'
                },
                'memory_efficiency': {
                    'qwen3_avg_memory': qwen3_perf.get('avg_memory_usage', 0),
                    'smolvlm2_avg_memory': smolvlm2_perf.get('avg_memory_usage', 0),
                    'winner': 'qwen3' if qwen3_perf.get('avg_memory_usage', float('inf')) < smolvlm2_perf.get('avg_memory_usage', float('inf')) else 'smolvlm2'
                },
                'success_rate': {
                    'qwen3_success_rate': qwen3_perf.get('success_rate', 0),
                    'smolvlm2_success_rate': smolvlm2_perf.get('success_rate', 0),
                    'winner': 'qwen3' if qwen3_perf.get('success_rate', 0) > smolvlm2_perf.get('success_rate', 0) else 'smolvlm2'
                }
            }
        
        # 质量比较
        qwen3_text_results = qwen3_results.get('text_analysis', [])
        smolvlm2_text_results = smolvlm2_results.get('text_analysis', [])
        
        if qwen3_text_results and smolvlm2_text_results:
            qwen3_avg_quality = np.mean([r['quality']['quality_score'] for r in qwen3_text_results if 'quality' in r and 'quality_score' in r['quality']])
            smolvlm2_avg_quality = np.mean([r['quality']['quality_score'] for r in smolvlm2_text_results if 'quality' in r and 'quality_score' in r['quality']])
            
            comparison['quality_comparison'] = {
                'qwen3_avg_quality': qwen3_avg_quality,
                'smolvlm2_avg_quality': smolvlm2_avg_quality,
                'winner': 'qwen3' if qwen3_avg_quality > smolvlm2_avg_quality else 'smolvlm2'
            }
        
        # 能力比较
        comparison['capability_comparison'] = {
            'text_analysis': {
                'qwen3': len(qwen3_results.get('text_analysis', [])) > 0,
                'smolvlm2': len(smolvlm2_results.get('text_analysis', [])) > 0
            },
            'multimodal_analysis': {
                'qwen3': False,  # Qwen3不支持多模态
                'smolvlm2': len(smolvlm2_results.get('multimodal_analysis', [])) > 0
            },
            'reasoning_tasks': {
                'qwen3': len(qwen3_results.get('reasoning_tasks', [])) > 0,
                'smolvlm2': len(smolvlm2_results.get('reasoning_tasks', [])) > 0
            }
        }
        
        # 生成建议
        recommendations = []
        
        if comparison['performance_comparison']:
            speed_winner = comparison['performance_comparison']['speed']['winner']
            memory_winner = comparison['performance_comparison']['memory_efficiency']['winner']
            
            if speed_winner == memory_winner:
                recommendations.append(f"{speed_winner}在速度和内存效率方面都表现更好")
            else:
                recommendations.append(f"{speed_winner}速度更快，{memory_winner}内存效率更高")
        
        if comparison['capability_comparison']['multimodal_analysis']['smolvlm2']:
            recommendations.append("SmolVLM2支持多模态分析，适合图文结合的古董鉴定任务")
        
        if comparison['quality_comparison']:
            quality_winner = comparison['quality_comparison']['winner']
            recommendations.append(f"{quality_winner}在文本分析质量方面表现更好")
        
        recommendations.extend([
            "对于纯文本分析任务，可以根据速度和质量需求选择模型",
            "对于图文结合的鉴定任务，建议使用SmolVLM2",
            "建议根据具体使用场景和资源限制选择合适的模型"
        ])
        
        comparison['recommendations'] = recommendations
        
        logger.info("模型比较分析完成")
        return comparison
    
    def run_evaluation(self) -> str:
        """运行完整的模型评测"""
        logger.info("="*60)
        logger.info("开始Qwen和SmolVLM2模型效果评测")
        logger.info("="*60)
        
        try:
            # 初始化模型
            if not self.initialize_models():
                raise Exception("模型初始化失败")
            
            # 评测Qwen3
            if self.qwen3_client:
                self.evaluation_results['qwen3_results'] = self.evaluate_qwen3()
            
            # 评测SmolVLM2
            if self.smolvlm2_client:
                self.evaluation_results['smolvlm2_results'] = self.evaluate_smolvlm2()
            
            # 模型比较
            self.evaluation_results['comparison'] = self.compare_models()
            
            # 保存结果
            report_file = self._save_evaluation_report()
            
            # 打印摘要
            self._print_evaluation_summary()
            
            logger.info("="*60)
            logger.info("模型评测完成")
            logger.info(f"详细报告: {report_file}")
            logger.info("="*60)
            
            return report_file
            
        except Exception as e:
            logger.error(f"评测过程中出现错误: {e}")
            raise
    
    def _save_evaluation_report(self) -> str:
        """保存评测报告"""
        report_file = os.path.join(
            self.output_dir,
            f"model_evaluation_report_{self.timestamp}.json"
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评测报告已保存: {report_file}")
        return report_file
    
    def _print_evaluation_summary(self):
        """打印评测摘要"""
        logger.info("\n评测结果摘要:")
        logger.info("="*40)
        
        # Qwen3结果
        qwen3_results = self.evaluation_results.get('qwen3_results', {})
        if qwen3_results and 'performance_summary' in qwen3_results:
            perf = qwen3_results['performance_summary']
            logger.info(f"\n📊 Qwen3模型:")
            logger.info(f"  成功率: {perf.get('success_rate', 0):.2%}")
            logger.info(f"  平均响应时间: {perf.get('avg_execution_time', 0):.2f}秒")
            logger.info(f"  平均内存使用: {perf.get('avg_memory_usage', 0):.1f}MB")
        
        # SmolVLM2结果
        smolvlm2_results = self.evaluation_results.get('smolvlm2_results', {})
        if smolvlm2_results and 'performance_summary' in smolvlm2_results:
            perf = smolvlm2_results['performance_summary']
            logger.info(f"\n📊 SmolVLM2模型:")
            logger.info(f"  成功率: {perf.get('success_rate', 0):.2%}")
            logger.info(f"  平均响应时间: {perf.get('avg_execution_time', 0):.2f}秒")
            logger.info(f"  平均内存使用: {perf.get('avg_memory_usage', 0):.1f}MB")
            
            # 多模态能力
            multimodal_tests = smolvlm2_results.get('multimodal_analysis', [])
            if multimodal_tests:
                logger.info(f"  多模态测试: {len(multimodal_tests)}个")
        
        # 比较结果
        comparison = self.evaluation_results.get('comparison', {})
        if comparison and 'recommendations' in comparison:
            logger.info(f"\n💡 建议:")
            for i, rec in enumerate(comparison['recommendations'][:3], 1):
                logger.info(f"  {i}. {rec}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Qwen和SmolVLM2模型效果评测脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认为当前目录）')
    
    args = parser.parse_args()
    
    try:
        evaluator = ModelEvaluator(output_dir=args.output_dir)
        report_file = evaluator.run_evaluation()
        print(f"\n✅ 评测完成！详细报告: {report_file}")
        return 0
    except Exception as e:
        print(f"\n❌ 评测失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())