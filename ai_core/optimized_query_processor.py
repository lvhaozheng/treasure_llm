#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的查询处理器
实现图片+文字查询的优化流程：
1. 图片先查询Milvus获取top5相似结果
2. 检查相似度是否达到标准
3. 如果达标则使用metadata作为prompt的一部分查询大模型
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logger_config import get_ai_logger

# 添加ai_core路径
ai_core_path = os.path.dirname(__file__)
if ai_core_path not in sys.path:
    sys.path.insert(0, ai_core_path)

from clip_encoder.clip_encoder import CLIPEncoder
from vector_db.milvus_client import MilvusClient
from ai_core.models.client.internvl3_5_client import InternVL3_5Client

# 配置日志
logger = get_ai_logger('optimized_query_processor')


class OptimizedQueryProcessor:
    """优化的查询处理器"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 top_k: int = 5,
                 clip_encoder=None,
                 milvus_client=None,
                 internvl3_5_client=None):
        """
        初始化优化查询处理器
        
        Args:
            similarity_threshold: 相似度阈值，超过此值才使用metadata增强prompt
            top_k: 从Milvus检索的top结果数量
            clip_encoder: 外部传入的CLIP编码器实例（可选）
            milvus_client: 外部传入的Milvus客户端实例（可选）
            internvl3_5_client: 外部传入的InternVL3.5客户端实例（可选）
        """
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        
        # 使用外部传入的组件或初始化新组件
        self.clip_encoder = clip_encoder
        self.milvus_client = milvus_client
        self.internvl3_5_client = internvl3_5_client
        
        # 只初始化未提供的组件
        self._init_components()
    
    def _init_components(self):
        """初始化各个组件（只初始化未提供的组件）"""
        try:
            # 初始化CLIP编码器（如果未提供）
            if self.clip_encoder is None:
                logger.info("初始化CLIP编码器...")
                self.clip_encoder = CLIPEncoder()
            else:
                logger.info("使用外部传入的CLIP编码器")
            
            # 初始化Milvus客户端（如果未提供）
            if self.milvus_client is None:
                logger.info("初始化Milvus客户端...")
                self.milvus_client = MilvusClient()
            else:
                logger.info("使用外部传入的Milvus客户端")
            
            # 初始化InternVL3.5客户端（如果未提供）
            if self.internvl3_5_client is None:
                logger.info("初始化InternVL3.5客户端...")
                self.internvl3_5_client = InternVL3_5Client()
            else:
                logger.info("使用外部传入的InternVL3.5客户端")
            
            logger.info("所有组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def process_image_text_query(self, 
                                image: Union[str, Image.Image, np.ndarray],
                                text_query: str = "") -> Dict[str, Any]:
        """
        处理图片+文字查询的优化流程
        
        Args:
            image: 输入图片
            text_query: 文字查询内容
            
        Returns:
            处理结果字典
        """
        try:
            logger.info("🚀 开始处理图片+文字查询")
            logger.info(f"📝 原始查询: {text_query if text_query else '(无文字查询)'}")
            
            # 步骤1: 使用CLIP编码图片
            logger.info("🔍 步骤1: 编码图片特征")
            image_features = self.clip_encoder.encode_image(image)
            logger.info(f"✅ 图片特征编码完成，维度: {len(image_features)}")
            
            # 步骤2: 在Milvus中搜索top5相似图片
            logger.info(f"🔎 步骤2: 在向量数据库中搜索top{self.top_k}相似图片")
            similar_results = self.milvus_client.search_similar_images(
                query_vector=image_features.tolist(),
                top_k=self.top_k,
                score_threshold=0.3  # 设置较低阈值以获取更多候选
            )
            
            # 步骤3: 检查最高相似度是否达到标准
            use_metadata_enhancement = False
            best_match = None
            metadata_context = ""
            
            if similar_results:
                best_match = similar_results[0]
                best_similarity = best_match.get('score', 0)
                
                logger.info(f"📊 相似度检查结果:")
                logger.info(f"   - 找到相似图片数量: {len(similar_results)}")
                logger.info(f"   - 最高相似度: {best_similarity:.4f}")
                logger.info(f"   - 设定阈值: {self.similarity_threshold}")
                logger.info(f"   - 是否达标: {'✅ 是' if best_similarity >= self.similarity_threshold else '❌ 否'}")
                
                # 打印所有相似结果的详细信息
                for i, result in enumerate(similar_results, 1):
                    score = result.get('score', 0)
                    metadata = result.get('metadata', {})
                    name = metadata.get('name', '未知') if isinstance(metadata, dict) else '未知'
                    logger.info(f"   - 相似图片{i}: {name} (相似度: {score:.4f})")
                
                if best_similarity >= self.similarity_threshold:
                    use_metadata_enhancement = True
                    logger.info("🎯 相似度达标，启用metadata增强分析")
                    
                    # 构建metadata上下文
                    metadata_context = self._build_metadata_context(similar_results)
                    logger.info(f"📋 构建的metadata上下文长度: {len(metadata_context)}字符")
                else:
                    logger.info("⚠️ 相似度未达标，使用标准分析")
            else:
                logger.warning("❌ 未找到任何相似图片")
            
            # 步骤4: 根据相似度结果选择分析策略
            logger.info(f"🔄 步骤4: 选择分析策略 - {'metadata增强' if use_metadata_enhancement else '标准分析'}")
            
            if use_metadata_enhancement:
                analysis_result = self._analyze_with_metadata_enhancement(
                    image, text_query, metadata_context, similar_results
                )
            else:
                analysis_result = self._analyze_standard(
                    image, text_query
                )
            
            # 整合最终结果
            final_result = {
                "query_timestamp": datetime.now().isoformat(),
                "image_features": image_features.tolist(),
                "text_query": text_query,
                "similar_results": similar_results,
                "best_similarity": best_match.get('score', 0) if best_match else 0,
                "similarity_threshold": self.similarity_threshold,
                "metadata_enhanced": use_metadata_enhancement,
                "analysis": analysis_result,
                "processing_strategy": "metadata_enhanced" if use_metadata_enhancement else "standard"
            }
            
            # 处理完成的详细日志记录
            logger.info("\n" + "🎉"*20 + " 处理完成 " + "🎉"*20)
            logger.info("📊 处理结果统计:")
            logger.info(f"   - 处理策略: {'metadata增强' if use_metadata_enhancement else '标准分析'}")
            logger.info(f"   - 相似图片数量: {len(similar_results) if similar_results else 0}")
            if similar_results:
                logger.info(f"   - 最高相似度: {similar_results[0].get('score', 0):.4f}")
            logger.info(f"   - 结果包含字段: {list(final_result.keys())}")
            if analysis_result:
                logger.info(f"   - 分析文本长度: {len(analysis_result.get('analysis_text', ''))}字符")
                logger.info(f"   - 使用模型: {analysis_result.get('model_used', '未知')}")
            else:
                logger.warning("   - 分析结果为空")
            logger.info("="*60)
            
            return final_result
            
        except Exception as e:
            logger.error("\n" + "❌"*20 + " 处理失败 " + "❌"*20)
            logger.error(f"💥 错误信息: {str(e)}")
            logger.error(f"🔍 错误类型: {type(e).__name__}")
            import traceback
            logger.error(f"📋 错误堆栈:\n{traceback.format_exc()}")
            logger.error("="*60)
            
            # 返回错误信息而不是抛出异常
            return {
                'error': True,
                'error_message': str(e),
                'error_type': type(e).__name__,
                'processing_failed': True,
                'query_timestamp': datetime.now().isoformat(),
                'text_query': text_query if 'text_query' in locals() else '',
                'processing_strategy': 'failed'
            }
    
    def _build_metadata_context(self, similar_results: List[Dict[str, Any]]) -> str:
        """
        构建metadata上下文信息
        
        Args:
            similar_results: 相似搜索结果
            
        Returns:
            构建的上下文字符串
        """
        if not similar_results:
            logger.warning("⚠️ 没有相似结果，无法构建metadata上下文")
            return ""
        
        logger.info(f"📋 开始构建metadata上下文，使用前{min(3, len(similar_results))}个最相似结果")
        
        context_parts = []
        context_parts.append("\n=== 相似古董参考信息 ===")
        
        valid_references = 0
        for i, result in enumerate(similar_results[:3], 1):  # 只使用前3个最相似的
            metadata = result.get('metadata', {})
            similarity = result.get('score', 0)
            
            logger.info(f"🔍 处理相似结果{i}:")
            logger.info(f"   - 相似度: {similarity:.4f}")
            logger.info(f"   - metadata类型: {type(metadata)}")
            logger.info(f"   - metadata内容: {metadata}")
            logger.info(f"   - metadata键: {list(metadata.keys()) if isinstance(metadata, dict) else 'N/A'}")
            
            context_parts.append(f"\n【参考古董 {i}】（相似度: {similarity:.3f}）")
            
            # 提取关键metadata信息
            if isinstance(metadata, dict):
                # 尝试多种可能的字段名
                name = (metadata.get('name') or metadata.get('title') or 
                       metadata.get('artifact_name') or metadata.get('文物名称') or '未知')
                dynasty = (metadata.get('dynasty') or metadata.get('period') or 
                          metadata.get('era') or metadata.get('朝代') or '未知朝代')
                category = (metadata.get('category') or metadata.get('type') or 
                           metadata.get('classification') or metadata.get('类别') or '未知类别')
                material = (metadata.get('material') or metadata.get('materials') or 
                           metadata.get('composition') or metadata.get('材质') or '未知材质')
                description = (metadata.get('description') or metadata.get('desc') or 
                              metadata.get('details') or metadata.get('描述') or '无描述')
                
                logger.info(f"   - 名称: {name}")
                logger.info(f"   - 朝代: {dynasty}")
                logger.info(f"   - 类别: {category}")
                logger.info(f"   - 材质: {material}")
                logger.info(f"   - 描述长度: {len(description)}字符")
                
                context_parts.append(f"名称: {name}")
                context_parts.append(f"朝代: {dynasty}")
                context_parts.append(f"类别: {category}")
                context_parts.append(f"材质: {material}")
                context_parts.append(f"描述: {description}")
                valid_references += 1
            else:
                logger.warning(f"   - ⚠️ metadata格式无效: {metadata}")
            
        context_parts.append("\n=== 参考信息结束 ===")
        
        context = "\n".join(context_parts)
        logger.info(f"✅ metadata上下文构建完成:")
        logger.info(f"   - 有效参考数量: {valid_references}")
        logger.info(f"   - 上下文总长度: {len(context)}字符")
        logger.info(f"   - 上下文行数: {len(context.split())}行")
        
        return context
    
    def _analyze_with_metadata_enhancement(self, 
                                         image: Union[str, Image.Image, np.ndarray],
                                         text_query: str,
                                         metadata_context: str,
                                         similar_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        使用metadata增强的分析
        
        Args:
            image: 输入图片
            text_query: 文字查询
            metadata_context: metadata上下文
            similar_results: 相似搜索结果
            
        Returns:
            分析结果
        """
        logger.info("🎯 开始metadata增强分析")
        
        # 构建增强的prompt
        enhanced_prompt = self._build_enhanced_prompt(text_query, metadata_context)
        
        # 记录prompt优化对比
        original_prompt = text_query if text_query else "请分析这件古董"
        logger.info("\n" + "="*80)
        logger.info("📋 PROMPT优化对比:")
        logger.info("="*80)
        logger.info(f"🔸 原始prompt ({len(original_prompt)}字符):")
        logger.info(f"   {original_prompt}")
        logger.info("-"*40)
        logger.info(f"🔹 增强prompt ({len(enhanced_prompt)}字符):")
        # 分行显示增强prompt，避免过长
        for i, line in enumerate(enhanced_prompt.split('\n')[:10], 1):
            if line.strip():
                logger.info(f"   {i:2d}| {line[:100]}{'...' if len(line) > 100 else ''}")
        if len(enhanced_prompt.split('\n')) > 10:
            remaining_lines = len(enhanced_prompt.split('\n')) - 10
            logger.info(f"   ... (还有{remaining_lines}行)")
        logger.info("="*80)
        
        # 使用InternVL3.5进行分析（流式输出）
        logger.info("🤖 调用InternVL3.5进行增强分析（流式输出）...")
        analysis_text = ""
        for chunk in self.internvl3_5_client.chat_about_antique_stream(enhanced_prompt, image):
            analysis_text += chunk
            # 这里可以添加实时输出逻辑
            logger.debug(f"流式输出: {chunk}")
        model_used = "InternVL3.5"
        
        logger.info("✅ metadata增强分析完成")
        logger.info(f"📊 分析结果长度: {len(analysis_text)}字符")
        
        return {
            "analysis_text": analysis_text,
            "model_used": model_used,
            "enhancement_type": "metadata_enhanced",
            "reference_count": len(similar_results),
            "prompt_used": enhanced_prompt,
            "prompt_enhancement": {
                "original_length": len(original_prompt),
                "enhanced_length": len(enhanced_prompt),
                "enhancement_ratio": len(enhanced_prompt) / len(original_prompt) if original_prompt else 0
            }
        }
        
    def _build_enhanced_prompt(self, text_query: str, metadata_context: str) -> str:
        """
        构建增强的prompt
        
        Args:
            text_query: 用户文字查询
            metadata_context: metadata上下文信息
            
        Returns:
            构建的增强prompt
        """
        logger.info("🔧 开始构建增强prompt")
        logger.info(f"   - 原始查询长度: {len(text_query) if text_query else 0}字符")
        logger.info(f"   - metadata上下文长度: {len(metadata_context)}字符")
        
        user_query = text_query if text_query else "请分析这件古董"
        
        enhanced_prompt = f"""
作为专业的古董鉴定专家，请基于以下信息对这件古董进行详细分析：

用户查询: {user_query}

{metadata_context}

请结合上述相似古董的参考信息，从以下方面进行专业分析：
1. 外观特征对比：与参考古董的相似性和差异性
2. 朝代风格分析：基于参考信息推测可能的历史时期
3. 工艺技法评估：分析制作工艺和技术特点
4. 材质鉴定：结合参考信息分析可能的材质
5. 真伪初判：基于风格对比进行真伪评估
6. 价值评估：参考同类古董给出价值判断
7. 收藏建议：提供专业的收藏和保养建议

请提供详细、专业的分析报告。
"""
        
        logger.info("✅ 增强prompt构建完成")
        logger.info(f"   - 最终prompt长度: {len(enhanced_prompt)}字符")
        logger.info(f"   - 增强比例: {len(enhanced_prompt) / len(user_query):.2f}x")
        logger.info(f"   - 包含参考信息: {'是' if metadata_context else '否'}")
        
        return enhanced_prompt
    
    def _analyze_standard(self, 
                         image: Union[str, Image.Image, np.ndarray],
                         text_query: str) -> Dict[str, Any]:
        """
        标准分析（无metadata增强）
        
        Args:
            image: 输入图片
            text_query: 文字查询
            
        Returns:
            分析结果
        """
        logger.info("⚡ 开始标准分析")
        
        # 构建标准prompt
        standard_prompt = f"""
作为专业的古董鉴定专家，请对这件古董进行详细分析：

用户查询: {text_query if text_query else '请分析这件古董'}

请从以下方面进行专业分析：
1. 外观特征：描述古董的形状、颜色、纹饰等视觉特征
2. 材质推测：分析可能的制作材料
3. 工艺特征：观察制作工艺和技法
4. 年代推测：根据风格特征推测可能的历史时期
5. 真伪评估：基于观察到的特征进行初步真伪判断
6. 保存状况：评估古董的保存状态
7. 收藏价值：分析其潜在的收藏和市场价值

请提供专业、详细的分析报告。
"""
        
        # 记录标准分析的prompt信息
        original_query = text_query if text_query else "请分析这件古董"
        logger.info("\n" + "="*60)
        logger.info("📝 标准分析PROMPT信息:")
        logger.info("="*60)
        logger.info(f"🔸 用户查询: {original_query}")
        logger.info(f"🔹 标准prompt长度: {len(standard_prompt)}字符")
        logger.info(f"🔹 分析策略: 基于图像的直接分析（无参考信息）")
        logger.info("="*60)
        
        # 使用InternVL3.5进行分析（流式输出）
        logger.info("🤖 调用InternVL3.5进行标准分析（流式输出）...")
        analysis_text = ""
        for chunk in self.internvl3_5_client.chat_about_antique_stream(standard_prompt, image):
            analysis_text += chunk
            # 这里可以添加实时输出逻辑
            logger.debug(f"流式输出: {chunk}")
        model_used = "InternVL3.5"
        
        logger.info("✅ 标准分析完成")
        logger.info(f"📊 分析结果长度: {len(analysis_text)}字符")
        
        return {
            "analysis_text": analysis_text,
            "model_used": model_used,
            "enhancement_type": "standard",
            "reference_count": 0,
            "prompt_used": standard_prompt,
            "prompt_info": {
                "original_query": original_query,
                "prompt_length": len(standard_prompt),
                "analysis_strategy": "direct_image_analysis"
            }
        }
    
    def process_image_text_query_stream(self, 
                                       image: Union[str, Image.Image, np.ndarray],
                                       text_query: str = ""):
        """
        流式处理图片+文字查询的优化流程
        
        Args:
            image: 输入图片（文件路径、PIL图像或numpy数组）
            text_query: 文字查询内容
            
        Yields:
            分析结果文本片段
        """
        logger.info(f"🚀 开始流式优化查询处理 - 查询: {text_query[:50]}...")
        
        try:
            # 1. 图片编码
            logger.info("📸 开始图片编码...")
            image_embedding = self.clip_encoder.encode_image(image)
            logger.info(f"✅ 图片编码完成，向量维度: {image_embedding.shape}")
            
            # 2. 向量数据库检索
            logger.info(f"🔍 在Milvus中检索top{self.top_k}相似结果...")
            similar_results = self.milvus_client.search_similar_images(
                query_vector=image_embedding.tolist(),
                top_k=self.top_k,
                score_threshold=0.3
            )
            
            if not similar_results:
                logger.info("❌ 未找到相似结果，使用标准分析")
                yield from self._analyze_standard_stream(image, text_query)
                return
            
            # 3. 检查相似度
            max_similarity = max([result.get('score', 0) for result in similar_results])
            logger.info(f"📊 最高相似度: {max_similarity:.3f}, 阈值: {self.similarity_threshold}")
            
            if max_similarity >= self.similarity_threshold:
                logger.info("✅ 相似度达标，使用metadata增强分析")
                yield from self._analyze_with_metadata_enhancement_stream(
                    image, text_query, similar_results
                )
            else:
                logger.info("⚠️ 相似度未达标，使用标准分析")
                yield from self._analyze_standard_stream(image, text_query)
                
        except Exception as e:
            logger.error(f"流式优化查询处理失败: {e}")
            yield f"查询处理过程中出现错误: {str(e)}"
    
    def _analyze_standard_stream(self, 
                               image: Union[str, Image.Image, np.ndarray],
                               text_query: str):
        """
        流式标准分析（无metadata增强）
        
        Args:
            image: 输入图片
            text_query: 文字查询
            
        Yields:
            分析结果文本片段
        """
        logger.info("⚡ 开始流式标准分析")
        
        # 构建标准prompt
        standard_prompt = f"""
作为专业的古董鉴定专家，请对这件古董进行详细分析：

用户查询: {text_query if text_query else '请分析这件古董'}

请从以下方面进行专业分析：
1. 外观特征：描述古董的形状、颜色、纹饰等视觉特征
2. 材质推测：分析可能的制作材料
3. 工艺特征：观察制作工艺和技法
4. 年代推测：根据风格特征推测可能的历史时期
5. 真伪评估：基于观察到的特征进行初步真伪判断
6. 保存状况：评估古董的保存状态
7. 收藏价值：分析其潜在的收藏和市场价值

请提供专业、详细的分析报告。
"""
        
        logger.info("🤖 调用InternVL3.5进行流式标准分析...")
        
        # 直接流式输出
        for chunk in self.internvl3_5_client.chat_about_antique_stream(standard_prompt, image):
            logger.debug(f"流式输出: {chunk}")
            yield chunk
    
    def _analyze_with_metadata_enhancement_stream(self, 
                                                image: Union[str, Image.Image, np.ndarray],
                                                text_query: str,
                                                similar_results: List[Dict[str, Any]]):
        """
        流式metadata增强分析
        
        Args:
            image: 输入图片
            text_query: 文字查询
            similar_results: 相似结果列表
            
        Yields:
            分析结果文本片段
        """
        logger.info("🔥 开始流式metadata增强分析")
        
        # 构建metadata上下文
        metadata_context = self._build_metadata_context(similar_results)
        
        # 构建增强prompt
        enhanced_prompt = self._build_enhanced_prompt(text_query, metadata_context)
        
        logger.info("🤖 调用InternVL3.5进行流式增强分析...")
        
        # 直接流式输出
        for chunk in self.internvl3_5_client.chat_about_antique_stream(enhanced_prompt, image):
            logger.debug(f"流式输出: {chunk}")
            yield chunk
    
    def get_processor_status(self) -> Dict[str, Any]:
        """
        获取处理器状态
        
        Returns:
            状态信息字典
        """
        return {
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
            "clip_encoder_ready": self.clip_encoder is not None,
            "milvus_client_ready": self.milvus_client is not None,
            "internvl3_5_ready": self.internvl3_5_client is not None
        }
    
    def update_similarity_threshold(self, new_threshold: float):
        """
        更新相似度阈值
        
        Args:
            new_threshold: 新的相似度阈值
        """
        old_threshold = self.similarity_threshold
        self.similarity_threshold = new_threshold
        logger.info(f"相似度阈值已更新: {old_threshold} -> {new_threshold}")


def main():
    """测试函数"""
    try:
        # 创建优化查询处理器
        processor = OptimizedQueryProcessor(
            similarity_threshold=0.7,
            top_k=5
        )
        
        # 获取状态
        status = processor.get_processor_status()
        print("处理器状态:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print("\n优化查询处理器初始化成功！")
        
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    main()