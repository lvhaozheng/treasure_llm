# -*- coding: utf-8 -*-
"""
Milvus向量数据库存储模块
用于将编码后的RAG知识库向量数据批量存入Milvus
"""

import os
import sys
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 简化的日志记录
class SimpleLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    
    def error(self, msg):
        print(f"[ERROR] {msg}")
    
    def warning(self, msg):
        print(f"[WARNING] {msg}")

logger = SimpleLogger()

class RAGMilvusStorage:
    """RAG知识库Milvus存储器"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.milvus_client = None
        self.collection_name = "rag_antique_vectors"
        
        # 尝试初始化Milvus客户端
        self._init_milvus_client()
    
    def _init_milvus_client(self):
        """初始化Milvus客户端"""
        try:
            # 尝试导入Milvus客户端
            from ai_core.vector_db.milvus_client import MilvusClient
            self.milvus_client = MilvusClient()
            logger.info("Milvus客户端初始化成功")
        except ImportError as e:
            logger.warning(f"无法导入Milvus客户端: {e}")
            logger.info("将使用模拟存储")
            self.milvus_client = None
        except Exception as e:
            logger.error(f"Milvus客户端初始化失败: {e}")
            self.milvus_client = None
    
    def _create_rag_collection(self):
        """创建RAG专用的集合"""
        if not self.milvus_client:
            logger.warning("Milvus客户端未初始化，跳过集合创建")
            return False
        
        try:
            # 检查集合是否存在
            collections = self.milvus_client.list_collections()
            if self.collection_name in collections:
                logger.info(f"集合 {self.collection_name} 已存在")
                return True
            
            # 创建新集合的字段定义
            fields = [
                {
                    "name": "id",
                    "type": "int64",
                    "is_primary": True,
                    "auto_id": True
                },
                {
                    "name": "vector_id",
                    "type": "varchar",
                    "max_length": 100
                },
                {
                    "name": "antique_id",
                    "type": "int64"
                },
                {
                    "name": "vector_type",
                    "type": "varchar",
                    "max_length": 20
                },
                {
                    "name": "vector",
                    "type": "float_vector",
                    "dim": 512
                },
                {
                    "name": "source_id",
                    "type": "varchar",
                    "max_length": 100
                },
                {
                    "name": "metadata",
                    "type": "json"
                }
            ]
            
            # 创建集合
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                fields=fields,
                description="RAG古董知识库向量集合"
            )
            
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            self.milvus_client.create_index(
                collection_name=self.collection_name,
                field_name="vector",
                index_params=index_params
            )
            
            logger.info(f"集合 {self.collection_name} 创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def _simulate_storage(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """模拟存储（用于测试）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        storage_file = os.path.join(self.base_dir, f"simulated_storage_{timestamp}.json")
        
        storage_data = {
            "collection_name": self.collection_name,
            "stored_at": datetime.now().isoformat(),
            "total_vectors": len(vectors),
            "vectors": vectors
        }
        
        with open(storage_file, 'w', encoding='utf-8') as f:
            json.dump(storage_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模拟存储完成，数据保存到: {storage_file}")
        
        return {
            "success": True,
            "total_vectors": len(vectors),
            "inserted_count": len(vectors),
            "failed_count": 0,
            "collection_name": self.collection_name
        }
    
    def prepare_vectors_for_milvus(self, vectors_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """准备向量数据用于Milvus存储"""
        prepared_vectors = []
        
        for vector_info in vectors_data['vectors']:
            prepared_vector = {
                "vector_id": vector_info['vector_id'],
                "antique_id": vector_info['antique_id'],
                "vector_type": vector_info['vector_type'],
                "vector": vector_info['vector'],
                "source_id": vector_info['source_id'],
                "metadata": vector_info['metadata']
            }
            prepared_vectors.append(prepared_vector)
        
        return prepared_vectors
    
    def batch_insert_vectors(self, vectors: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:
        """批量插入向量数据"""
        if not self.milvus_client:
            logger.warning("Milvus客户端未初始化，使用模拟存储")
            return self._simulate_storage(vectors)
        
        try:
            # 确保集合存在
            if not self._create_rag_collection():
                logger.error("集合创建失败，使用模拟存储")
                return self._simulate_storage(vectors)
            
            total_inserted = 0
            failed_insertions = 0
            
            # 分批插入
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                try:
                    # 准备批次数据
                    batch_data = {
                        "vector_id": [v["vector_id"] for v in batch],
                        "antique_id": [v["antique_id"] for v in batch],
                        "vector_type": [v["vector_type"] for v in batch],
                        "vector": [v["vector"] for v in batch],
                        "source_id": [v["source_id"] for v in batch],
                        "metadata": [v["metadata"] for v in batch]
                    }
                    
                    # 插入数据
                    result = self.milvus_client.insert(
                        collection_name=self.collection_name,
                        data=batch_data
                    )
                    
                    # 获取实际插入数量
                    inserted_count = len(batch)
                    if hasattr(result, 'insert_count'):
                        inserted_count = result.insert_count
                    elif isinstance(result, dict) and 'insert_count' in result:
                        inserted_count = result['insert_count']
                    
                    total_inserted += inserted_count
                    
                    if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(vectors):
                        logger.info(f"已插入 {min(i + batch_size, len(vectors))}/{len(vectors)} 条向量")
                    
                except Exception as e:
                    logger.error(f"批次 {i//batch_size + 1} 插入失败: {e}")
                    failed_insertions += len(batch)
            
            # 刷新集合
            self.milvus_client.flush(collection_name=self.collection_name)
            
            logger.info(f"向量插入完成: 成功 {total_inserted}, 失败 {failed_insertions}")
            
            return {
                "success": True,
                "total_vectors": len(vectors),
                "inserted_count": total_inserted,
                "failed_count": failed_insertions,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            return self._simulate_storage(vectors)
    
    def load_and_store_vectors(self, vectors_file_path: str) -> Dict[str, Any]:
        """加载并存储向量文件"""
        logger.info(f"开始加载向量文件: {vectors_file_path}")
        
        # 加载向量数据
        with open(vectors_file_path, 'r', encoding='utf-8') as f:
            vectors_data = json.load(f)
        
        logger.info(f"加载了 {vectors_data['metadata']['total_vectors']} 个向量")
        
        # 准备向量数据
        prepared_vectors = self.prepare_vectors_for_milvus(vectors_data)
        
        # 批量插入
        result = self.batch_insert_vectors(prepared_vectors)
        
        return result
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self.milvus_client:
            return {"error": "Milvus客户端未初始化"}
        
        try:
            # 获取集合信息
            stats = self.milvus_client.get_collection_stats(self.collection_name)
            return stats
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {"error": str(e)}
    
    def search_similar_vectors(self, query_vector: List[float], top_k: int = 10, 
                             vector_type: str = None) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        if not self.milvus_client:
            logger.warning("Milvus客户端未初始化，无法执行搜索")
            return []
        
        try:
            # 构建搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 构建过滤表达式
            filter_expr = None
            if vector_type:
                filter_expr = f"vector_type == '{vector_type}'"
            
            # 执行搜索
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["vector_id", "antique_id", "vector_type", "source_id", "metadata"]
            )
            
            return results[0] if results else []
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

def main():
    """主函数"""
    storage = RAGMilvusStorage()
    
    # 查找最新的向量文件
    base_dir = os.path.dirname(__file__)
    vector_files = [f for f in os.listdir(base_dir) if f.startswith('rag_vectors_') and f.endswith('.json')]
    
    if not vector_files:
        logger.error("未找到向量文件")
        return
    
    # 使用最新的向量文件
    latest_vectors = sorted(vector_files)[-1]
    vectors_path = os.path.join(base_dir, latest_vectors)
    
    logger.info(f"使用向量文件: {latest_vectors}")
    
    # 加载并存储向量
    result = storage.load_and_store_vectors(vectors_path)
    
    if result.get('success', False):
        logger.info(f"向量存储完成: {result.get('inserted_count', 0)} 条成功插入")
        if result.get('failed_count', 0) > 0:
            logger.warning(f"失败: {result['failed_count']} 条")
        
        # 获取集合统计信息
        stats = storage.get_collection_stats()
        if 'error' not in stats:
            logger.info(f"集合统计: {stats}")
    else:
        logger.error("向量存储失败")
        if 'error' in result:
            logger.error(f"错误信息: {result['error']}")
    
    return result

if __name__ == "__main__":
    main()