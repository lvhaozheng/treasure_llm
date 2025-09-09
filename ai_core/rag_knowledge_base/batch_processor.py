#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理脚本 - RAG知识库自动化构建

功能：
1. 自动生成古董样本数据
2. 批量编码图文数据
3. 存储到向量数据库
4. 提供完整的处理流程监控

作者: AI Assistant
创建时间: 2025-09-07
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from sample_data_generator import SampleDataGenerator
    from rag_encoder import RAGEncoder
    from milvus_storage import RAGMilvusStorage
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有依赖模块都在正确的路径下")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGBatchProcessor:
    """RAG知识库批量处理器"""
    
    def __init__(self, base_dir=None):
        """初始化批量处理器
        
        Args:
            base_dir: 基础目录，默认为当前脚本所在目录
        """
        self.base_dir = base_dir or os.path.dirname(__file__)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 初始化各个组件
        self.data_generator = None
        self.encoder = None
        self.storage = None
        
        # 处理统计
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_samples': 0,
            'encoded_vectors': 0,
            'stored_vectors': 0,
            'failed_count': 0,
            'processing_time': 0
        }
        
        logger.info(f"批量处理器初始化完成，基础目录: {self.base_dir}")
    
    def initialize_components(self):
        """初始化所有组件"""
        try:
            # 初始化数据生成器
            self.data_generator = SampleDataGenerator()
            logger.info("数据生成器初始化成功")
            
            # 初始化编码器
            self.encoder = RAGEncoder()
            logger.info("CLIP编码器初始化成功")
            
            # 初始化存储器
            self.storage = RAGMilvusStorage()
            logger.info("Milvus存储器初始化成功")
            
            return True
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            return False
    
    def generate_sample_data(self, target_count=500):
        """生成样本数据
        
        Args:
            target_count: 目标数据条数
            
        Returns:
            str: 生成的数据文件路径
        """
        try:
            logger.info(f"开始生成 {target_count} 条样本数据")
            
            # 生成数据集
            entries = self.data_generator.generate_extended_dataset(target_count)
            
            # 转换为字典格式
            dataset = {
                "metadata": {
                    "total_count": len(entries),
                    "generated_at": datetime.now().isoformat(),
                    "version": "1.0"
                },
                "entries": [entry.to_dict() for entry in entries]
            }
            
            # 保存数据集
            data_file = os.path.join(
                self.base_dir, 
                f"rag_dataset_{self.timestamp}.json"
            )
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            self.stats['total_samples'] = len(dataset['entries'])
            logger.info(f"样本数据生成完成: {len(dataset['entries'])} 条，保存到 {data_file}")
            
            return data_file
            
        except Exception as e:
            logger.error(f"样本数据生成失败: {e}")
            raise
    
    def encode_data(self, data_file):
        """编码数据
        
        Args:
            data_file: 数据文件路径
            
        Returns:
            str: 编码后的向量文件路径
        """
        try:
            logger.info(f"开始编码数据文件: {data_file}")
            
            # 批量编码（直接传入文件路径）
            vectors, stats = self.encoder.encode_dataset(data_file)
            
            # 保存编码结果
            vector_file = os.path.join(
                self.base_dir,
                f"rag_vectors_{self.timestamp}.json"
            )
            
            # 转换向量数据为可序列化格式
            vector_data = {
                'metadata': {
                    'total_vectors': len(vectors),
                    'encoding_time': datetime.now().isoformat(),
                    'stats': stats
                },
                'vectors': [{
                    'vector_id': v.vector_id,
                    'antique_id': v.antique_id,
                    'vector_type': v.vector_type,
                    'vector': v.vector,
                    'source_id': v.source_id,
                    'metadata': v.metadata
                } for v in vectors]
            }
            
            with open(vector_file, 'w', encoding='utf-8') as f:
                json.dump(vector_data, f, ensure_ascii=False, indent=2)
            
            result = {
                'success': True,
                'total_vectors': len(vectors),
                'vector_file': vector_file,
                'stats': stats
            }
            
            if result['success']:
                self.stats['encoded_vectors'] = result['total_vectors']
                logger.info(f"数据编码完成: {result['total_vectors']} 个向量")
                return result['vector_file']
            else:
                raise Exception(f"编码失败: {result.get('error', '未知错误')}")
                
        except Exception as e:
            logger.error(f"数据编码失败: {e}")
            raise
    
    def store_vectors(self, vector_file):
        """存储向量
        
        Args:
            vector_file: 向量文件路径
            
        Returns:
            dict: 存储结果
        """
        try:
            logger.info(f"开始存储向量文件: {vector_file}")
            
            # 存储向量
            result = self.storage.load_and_store_vectors(vector_file)
            
            if result and result.get('success', False):
                self.stats['stored_vectors'] = result.get('inserted_count', 0)
                self.stats['failed_count'] = result.get('failed_count', 0)
                logger.info(f"向量存储完成: {result['inserted_count']} 条成功")
                
                if result.get('failed_count', 0) > 0:
                    logger.warning(f"存储失败: {result['failed_count']} 条")
                    
                return result
            else:
                raise Exception(f"存储失败: {result.get('message', '未知错误')}")
                
        except Exception as e:
            logger.error(f"向量存储失败: {e}")
            raise
    
    def run_full_pipeline(self, target_count=500):
        """运行完整的处理流程
        
        Args:
            target_count: 目标数据条数
            
        Returns:
            dict: 处理结果统计
        """
        self.stats['start_time'] = datetime.now()
        
        try:
            logger.info("="*60)
            logger.info("开始RAG知识库批量处理流程")
            logger.info(f"目标数据量: {target_count} 条")
            logger.info("="*60)
            
            # 1. 初始化组件
            if not self.initialize_components():
                raise Exception("组件初始化失败")
            
            # 2. 生成样本数据
            data_file = self.generate_sample_data(target_count)
            
            # 3. 编码数据
            vector_file = self.encode_data(data_file)
            
            # 4. 存储向量
            storage_result = self.store_vectors(vector_file)
            
            # 5. 计算处理时间
            self.stats['end_time'] = datetime.now()
            self.stats['processing_time'] = (
                self.stats['end_time'] - self.stats['start_time']
            ).total_seconds()
            
            # 6. 生成处理报告
            self.generate_report()
            
            logger.info("="*60)
            logger.info("RAG知识库批量处理完成")
            logger.info("="*60)
            
            return {
                'success': True,
                'stats': self.stats,
                'files': {
                    'data_file': data_file,
                    'vector_file': vector_file
                }
            }
            
        except Exception as e:
            self.stats['end_time'] = datetime.now()
            if self.stats['start_time']:
                self.stats['processing_time'] = (
                    self.stats['end_time'] - self.stats['start_time']
                ).total_seconds()
            
            logger.error(f"批量处理失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }
    
    def generate_report(self):
        """生成处理报告"""
        report_file = os.path.join(
            self.base_dir, 
            f"processing_report_{self.timestamp}.json"
        )
        
        report = {
            'timestamp': self.timestamp,
            'processing_stats': self.stats,
            'summary': {
                'total_samples_generated': self.stats['total_samples'],
                'vectors_encoded': self.stats['encoded_vectors'],
                'vectors_stored': self.stats['stored_vectors'],
                'failed_count': self.stats['failed_count'],
                'success_rate': (
                    self.stats['stored_vectors'] / max(self.stats['total_samples'], 1) * 100
                    if self.stats['total_samples'] > 0 else 0
                ),
                'processing_time_seconds': self.stats['processing_time']
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"处理报告已生成: {report_file}")
        
        # 打印摘要
        logger.info("\n处理摘要:")
        logger.info(f"  生成样本: {self.stats['total_samples']} 条")
        logger.info(f"  编码向量: {self.stats['encoded_vectors']} 个")
        logger.info(f"  存储向量: {self.stats['stored_vectors']} 个")
        logger.info(f"  失败数量: {self.stats['failed_count']} 个")
        logger.info(f"  成功率: {report['summary']['success_rate']:.2f}%")
        logger.info(f"  处理时间: {self.stats['processing_time']:.2f} 秒")

def main():
    """主函数"""
    # 创建批量处理器
    processor = RAGBatchProcessor()
    
    # 运行完整流程
    result = processor.run_full_pipeline(target_count=500)
    
    if result['success']:
        logger.info("批量处理成功完成")
        return 0
    else:
        logger.error(f"批量处理失败: {result.get('error', '未知错误')}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)