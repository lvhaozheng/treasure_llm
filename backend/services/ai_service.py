import os
import sys
import torch
import clip
from PIL import Image
import numpy as np
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import json
from typing import Dict, List, Any

# 添加AI核心模块路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'ai_core'))
from ai_core import AICore

# 导入报告格式化器
try:
    from .report_formatter import format_appraisal_report
    REPORT_FORMATTER_AVAILABLE = True
except ImportError:
    REPORT_FORMATTER_AVAILABLE = False

class AIService:
    """AI服务类，集成LangChain、CLIP和Milvus"""
    
    def __init__(self, 
                 openai_api_key: str = "",
                 smolvlm2_model_path: str = "",
                 max_tokens: int = 512,
                 temperature: float = 0.7):
        """初始化AI服务
        
        Args:
            openai_api_key: OpenAI API 密钥
            smolvlm2_model_path: SmolVLM2 模型路径
            max_tokens: 最大生成token数
            temperature: 生成温度
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 尝试初始化 AI 核心（使用 SmolVLM2）
        try:
            self.ai_core = AICore(
                openai_api_key=openai_api_key,
                smolvlm2_model_path=smolvlm2_model_path,
                max_tokens=max_tokens,
                temperature=temperature
            )
            print(f"✅ AI核心初始化成功（SmolVLM2）")
            self.use_ai_core = True
        except Exception as e:
            print(f"❌ AI核心初始化失败: {e}")
            print("⚠️ 回退到传统组件...")
            self.use_ai_core = False
            self._init_legacy_components()
    
    def _init_legacy_components(self):
        """初始化传统组件（回退方案）"""
        self.ai_core = None
        
        # 初始化CLIP模型
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # 初始化LangChain
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # 初始化Milvus连接
        self._init_milvus()
        
        # 初始化LangChain Agent
        self._init_agent()
    
    def _init_milvus(self):
        """初始化Milvus向量数据库"""
        try:
            connections.connect("default", host="localhost", port="19530")
            
            # 创建集合
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="antique_id", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
            ]
            schema = CollectionSchema(fields, "antique_embeddings")
            
            # 检查集合是否存在
            if "antique_embeddings" not in [col.name for col in Collection.list_collections()]:
                self.collection = Collection("antique_embeddings", schema)
                # 创建索引
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                self.collection.create_index("embedding", index_params)
            else:
                self.collection = Collection("antique_embeddings")
                
        except Exception as e:
            print(f"Milvus连接失败: {e}")
            self.collection = None
    
    def _init_agent(self):
        """初始化LangChain Agent"""
        tools = [
            Tool(
                name="antique_analyzer",
                func=self._analyze_antique_tool,
                description="分析古董文物的真伪、年代、价值等信息"
            ),
            Tool(
                name="antique_search",
                func=self._search_antique_tool,
                description="搜索相似的古董文物"
            ),
            Tool(
                name="value_estimator",
                func=self._estimate_value_tool,
                description="评估古董文物的市场价值"
            )
        ]
        
        self.agent = initialize_agent(
            tools, 
            self.llm, 
            agent="conversational-react-description",
            memory=self.memory,
            verbose=True
        )
    
    def analyze_antique_stream(self, image_path: str = None, description: str = ""):
        """流式分析古董图像和描述
        
        Args:
            image_path: 图像路径
            description: 文本描述
            
        Yields:
            str: 流式输出的分析结果片段
        """
        if self.use_ai_core and self.ai_core:
            yield from self._analyze_with_ai_core_stream(image_path, description)
        else:
            # 传统组件不支持流式输出，使用模拟流式
            result = self._analyze_with_legacy(image_path)
            analysis_text = result.get('description', '分析完成')
            # 按句子分割进行流式输出
            sentences = analysis_text.split('。')
            for sentence in sentences:
                if sentence.strip():
                    yield sentence.strip() + '。'
    
    def analyze_antique(self, image_path: str = None, description: str = "") -> Dict[str, Any]:
        """分析古董文物（优先使用 AI 核心）
        
        Args:
            image_path: 图像路径
            description: 文本描述
            
        Returns:
            Dict: 分析结果
        """
        try:
            if self.use_ai_core and self.ai_core:
                # 使用新的 AI 核心（支持 SmolVLM2）
                return self._analyze_with_ai_core(image_path)
            else:
                # 使用传统方法
                return self._analyze_with_legacy(image_path)
                
        except Exception as e:
            raise Exception(f"分析失败: {str(e)}")
    
    def _analyze_with_ai_core_stream(self, image_path: str = None, description: str = ""):
        """使用 AI 核心进行流式分析"""
        try:
            # 加载图片
            image = None
            if image_path:
                image = Image.open(image_path).convert('RGB')
            
            # 使用 AI 核心进行流式分析
            if hasattr(self.ai_core, 'analyze_antique_stream'):
                yield from self.ai_core.analyze_antique_stream(
                    image=image,
                    description=description or "请对这件古董文物进行专业鉴定分析"
                )
            else:
                # 回退到非流式分析
                result = self.ai_core.analyze_antique_image(
                    image=image,
                    description=description or "请对这件古董文物进行专业鉴定分析"
                )
                # 模拟流式输出
                if isinstance(result, dict):
                    analysis_text = result.get('internvl3_5_analysis', {}).get('analysis_text', '分析完成')
                    sentences = analysis_text.split('。')
                    for sentence in sentences:
                        if sentence.strip():
                            yield sentence.strip() + '。'
                            
        except Exception as e:
            print(f"⚠️ AI核心流式分析失败: {e}")
            yield f"分析过程中出现错误: {str(e)}"
    
    def _analyze_with_ai_core(self, image_path: str) -> Dict[str, Any]:
        """使用 AI 核心进行分析"""
        try:
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            
            # 使用 AI 核心进行分析
            result = self.ai_core.analyze_antique_image(
                image=image,
                description="请对这件古董文物进行专业鉴定分析"
            )
            
            # 如果AI核心已经返回结构化数据，直接使用
            if isinstance(result, dict) and "report_type" in result:
                # AI核心已经使用了新的格式化器
                analysis_result = result
                analysis_result["backend"] = "ai_core_structured"
            elif REPORT_FORMATTER_AVAILABLE and isinstance(result, dict):
                # 使用格式化器处理原始结果
                raw_analysis = result.get('internvl3_5_analysis', {}).get('analysis_text', '')
                if raw_analysis:
                    analysis_result = format_appraisal_report(
                        ai_analysis=raw_analysis,
                        user_query="",
                        image_path=image_path,
                        model_info=result.get('model_info', {})
                    )
                    analysis_result["backend"] = "ai_core_formatted"
                    analysis_result["raw_result"] = result
                else:
                    # 回退到原始格式
                    analysis_result = self._format_legacy_result(result)
            else:
                # 回退到原始格式
                analysis_result = self._format_legacy_result(result)
            
            return analysis_result
            
        except Exception as e:
            print(f"⚠️ AI核心分析失败: {e}")
            # 回退到传统方法
            return self._analyze_with_legacy(image_path)
    
    def _format_legacy_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化传统结果为兼容格式"""
        return {
            "backend": "ai_core_internvl3_5",
            "model_info": result.get("model_info", {}),
            "internvl3_5_analysis": result.get("internvl3_5_analysis", {}),
            "similar_antiques": result.get("similar_antiques", []),
            "image_features": result.get("image_features", []),
            "analysis_timestamp": result.get("analysis_timestamp", "")
        }
    
    def _analyze_with_legacy(self, image_path: str) -> Dict[str, Any]:
        """使用传统方法进行分析"""
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        
        # CLIP编码
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_embedding = image_features.cpu().numpy()[0]
        
        # 使用LangChain Agent进行分析
        analysis_prompt = f"""
        请分析这张古董文物图片，提供以下信息：
        1. 文物类型和名称
        2. 可能的年代和朝代
        3. 材质和工艺特点
        4. 真伪评估（0-100分）
        5. 保存状况
        6. 市场价值预估
        7. 历史背景和文化价值
        
        请以JSON格式返回结果。
        """
        
        result = self.agent.run(analysis_prompt)
        
        # 解析结果
        try:
            analysis_result = json.loads(result)
        except:
            analysis_result = {
                "type": "未知",
                "name": "未知文物",
                "dynasty": "未知",
                "material": "未知",
                "authenticity_score": 50,
                "condition": "未知",
                "estimated_value": "未知",
                "description": result
            }
        
        # 添加CLIP特征
        analysis_result["backend"] = "legacy_langchain"
        analysis_result["embedding"] = image_embedding.tolist()
        
        return analysis_result
    
    def search_antiques(self, query: str) -> List[Dict[str, Any]]:
        """搜索相似古董"""
        try:
            # 文本编码
            text_input = clip.tokenize([query]).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_input)
                text_embedding = text_features.cpu().numpy()[0]
            
            # 在Milvus中搜索
            if self.collection:
                self.collection.load()
                search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
                results = self.collection.search(
                    data=[text_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=10,
                    output_fields=["antique_id", "metadata"]
                )
                
                search_results = []
                for hits in results:
                    for hit in hits:
                        metadata = json.loads(hit.entity.get("metadata"))
                        search_results.append({
                            "antique_id": hit.entity.get("antique_id"),
                            "similarity": hit.score,
                            "metadata": metadata
                        })
                
                return search_results
            else:
                return []
                
        except Exception as e:
            raise Exception(f"搜索失败: {str(e)}")
    
    def _analyze_antique_tool(self, query: str) -> str:
        """LangChain工具：分析古董"""
        return f"基于图像分析，这是一个{query}。建议进行专业鉴定。"
    
    def _search_antique_tool(self, query: str) -> str:
        """LangChain工具：搜索古董"""
        return f"找到与'{query}'相关的古董文物信息。"
    
    def _estimate_value_tool(self, query: str) -> str:
        """LangChain工具：价值评估"""
        return f"基于市场行情，{query}的预估价值为参考价格。"
    
    def add_to_vector_db(self, antique_id: int, embedding: List[float], metadata: Dict[str, Any]):
        """添加古董到向量数据库"""
        try:
            if self.collection:
                data = [
                    [antique_id],
                    [embedding],
                    [json.dumps(metadata)]
                ]
                self.collection.insert(data)
                self.collection.flush()
        except Exception as e:
            print(f"添加到向量数据库失败: {e}")
    
    def get_antique_knowledge(self, category: str) -> str:
        """获取古董知识"""
        knowledge_base = {
            "瓷器": "瓷器是中国古代重要的工艺品，主要分为青瓷、白瓷、彩瓷等。宋代五大名窑包括汝窑、官窑、哥窑、钧窑、定窑。",
            "玉器": "玉器在中国文化中具有重要地位，主要分为软玉和硬玉。和田玉、翡翠等都是珍贵的玉器材料。",
            "书画": "中国书画艺术源远流长，包括书法和绘画两大类。历代名家辈出，作品具有很高的艺术价值和收藏价值。",
            "青铜器": "青铜器是中国古代重要的礼器和实用器，主要出现在商周时期，具有重要的历史和文化价值。",
            "家具": "明清家具是中国古典家具的代表，以黄花梨、紫檀等珍贵木材制作，工艺精湛，造型优美。"
        }
        
        return knowledge_base.get(category, "该类别古董信息暂无详细资料。")
