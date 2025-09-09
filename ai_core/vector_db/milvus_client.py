"""
Milvus 向量数据库客户端
用于高性能向量存储和检索
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.logger_config import get_ai_logger

# 配置向量数据库模块日志
logger = get_ai_logger('milvus_client')


class MilvusClient:
    """Milvus 向量数据库客户端"""
    
    def __init__(self, host: str = "localhost", port: str = "19530", 
                 collection_name: str = "antique_vectors", dim: int = 512):
        """
        初始化 Milvus 客户端
        
        Args:
            host: Milvus 服务器地址
            port: Milvus 服务器端口
            collection_name: 集合名称
            dim: 向量维度
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        
        try:
            # 连接 Milvus
            connections.connect(alias="default", host=host, port=port)
            logger.info(f"成功连接到 Milvus: {host}:{port}")
            
            # 初始化集合
            self._init_collection()
            
        except Exception as e:
            logger.error(f"Milvus 连接失败: {e}")
            raise
    
    def _init_collection(self):
        """初始化集合"""
        try:
            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"使用现有集合: {self.collection_name}")
            else:
                # 创建新集合
                self._create_collection()
                logger.info(f"创建新集合: {self.collection_name}")
            
            # 加载集合
            self.collection.load()
            
        except Exception as e:
            logger.error(f"集合初始化失败: {e}")
            raise
    
    def _create_collection(self):
        """创建集合"""
        try:
            # 定义字段（只使用一个向量字段）
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="antique_id", dtype=DataType.INT64, description="古董ID"),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim, description="特征向量"),
                FieldSchema(name="vector_type", dtype=DataType.VARCHAR, max_length=32, description="向量类型"), # image/text
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535, description="元数据")
            ]
            
            # 创建集合模式
            schema = CollectionSchema(fields=fields, description="古董向量集合")
            
            # 创建集合
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # 创建索引
            self._create_index()
            
        except Exception as e:
            logger.error(f"集合创建失败: {e}")
            raise
    
    def _create_index(self):
        """创建索引"""
        try:
            # 为向量字段创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="vector", index_params=index_params)
            
            logger.info("索引创建成功")
            
        except Exception as e:
            logger.error(f"索引创建失败: {e}")
            raise
    
    def insert_vectors(self, antique_ids: List[int], vectors: List[List[float]], 
                      vector_types: List[str], metadata: List[Dict[str, Any]]) -> List[int]:
        """
        插入向量数据
        
        Args:
            antique_ids: 古董ID列表
            vectors: 向量列表
            vector_types: 向量类型列表 ('image' 或 'text')
            metadata: 元数据列表
            
        Returns:
            插入的ID列表
        """
        try:
            # 准备数据
            data = [
                antique_ids,
                vectors,
                vector_types,
                [str(m) for m in metadata]  # 转换为字符串
            ]
            
            # 插入数据
            insert_result = self.collection.insert(data)
            
            # 刷新集合
            self.collection.flush()
            
            logger.info(f"成功插入 {len(antique_ids)} 条向量数据")
            return insert_result.primary_keys
            
        except Exception as e:
            logger.error(f"向量插入失败: {e}")
            raise
    
    def search_similar_images(self, query_vector: List[float], top_k: int = 10, 
                            score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        搜索相似图像
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            
        Returns:
            相似结果列表
        """
        try:
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索（只搜索图像向量）
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr="vector_type == 'image'",  # 只搜索图像向量
                output_fields=["antique_id", "vector_type", "metadata"]
            )
            
            # 处理结果
            similar_results = []
            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        similar_results.append({
                            "antique_id": hit.entity.get("antique_id"),
                            "score": hit.score,
                            "vector_type": hit.entity.get("vector_type"),
                            "metadata": eval(hit.entity.get("metadata")) if hit.entity.get("metadata") else {}
                        })
            
            return similar_results
            
        except Exception as e:
            logger.error(f"图像搜索失败: {e}")
            raise
    
    def search_similar_texts(self, query_vector: List[float], top_k: int = 10, 
                           score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        搜索相似文本
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            
        Returns:
            相似结果列表
        """
        try:
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索（只搜索文本向量）
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr="vector_type == 'text'",  # 只搜索文本向量
                output_fields=["antique_id", "vector_type", "metadata"]
            )
            
            # 处理结果
            similar_results = []
            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        similar_results.append({
                            "antique_id": hit.entity.get("antique_id"),
                            "score": hit.score,
                            "vector_type": hit.entity.get("vector_type"),
                            "metadata": eval(hit.entity.get("metadata")) if hit.entity.get("metadata") else {}
                        })
            
            return similar_results
            
        except Exception as e:
            logger.error(f"文本搜索失败: {e}")
            raise
    
    def delete_by_antique_id(self, antique_id: int) -> bool:
        """
        根据古董ID删除向量
        
        Args:
            antique_id: 古董ID
            
        Returns:
            是否删除成功
        """
        try:
            # 删除条件
            expr = f"antique_id == {antique_id}"
            
            # 执行删除
            self.collection.delete(expr)
            
            # 刷新集合
            self.collection.flush()
            
            logger.info(f"成功删除古董ID {antique_id} 的向量数据")
            return True
            
        except Exception as e:
            logger.error(f"向量删除失败: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典
        """
        try:
            # 将索引对象转换为可序列化的格式
            indexes_info = []
            for index in self.collection.indexes:
                index_info = {
                    "field_name": index.field_name,
                    "index_name": index.index_name,
                    "params": index.params
                }
                indexes_info.append(index_info)
            
            stats = {
                "collection_name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "schema": self.collection.schema.to_dict(),
                "indexes": indexes_info
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            raise
    
    def drop_collection(self) -> bool:
        """
        删除集合
        
        Returns:
            是否删除成功
        """
        try:
            utility.drop_collection(self.collection_name)
            logger.info(f"成功删除集合: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"集合删除失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        try:
            if self.collection:
                self.collection.release()
            connections.disconnect("default")
            logger.info("Milvus 连接已关闭")
            
        except Exception as e:
            logger.error(f"关闭连接失败: {e}")
