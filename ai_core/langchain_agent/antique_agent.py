"""
古董分析智能代理
基于 LangChain 构建的智能古董分析和对话系统
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional, Union
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
import numpy as np
from PIL import Image

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.logger_config import get_ai_logger

# 添加 ai_core 目录到路径以正确导入模型客户端
ai_core_path = os.path.join(os.path.dirname(__file__), '..')
if ai_core_path not in sys.path:
    sys.path.insert(0, ai_core_path)
    
from models.client import Qwen3Client, InternVL3_5Client

# 配置LangChain代理模块日志
logger = get_ai_logger('langchain_agent')


class AntiqueAgent:
    """古董分析智能代理"""
    
    def __init__(self, openai_api_key: str = "", model_name: str = "gpt-3.5-turbo", 
                 temperature: float = 0.7, max_tokens: int = 1000,
                 use_local_models: bool = True, clip_encoder=None, vector_db=None):
        """
        初始化古董分析代理
        
        Args:
            openai_api_key: OpenAI API 密钥（可选，用于 LangChain 代理）
            model_name: 使用的模型名称
            temperature: 生成温度
            max_tokens: 最大生成令牌数
            use_local_models: 是否使用本地模型（Qwen3 和 SmolVLM2）
            clip_encoder: CLIP编码器实例（用于图像编码）
            vector_db: 向量数据库实例（用于相似性搜索）
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local_models = use_local_models
        self.clip_encoder = clip_encoder
        self.vector_db = vector_db
        
        # 设置环境变量（如果提供了 OpenAI API key）
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # 初始化组件
        self.llm = None
        self.memory = None
        self.tools = []
        self.agent = None
        
        # 本地模型客户端
        self.qwen3_client = None
        self.internvl3_5_client = None
        
        try:
            self._init_components()
            logger.info("古董分析代理初始化成功")
        except Exception as e:
            logger.error(f"古董分析代理初始化失败: {e}")
            raise
    
    def _init_components(self):
        """初始化组件"""
        # 初始化本地模型（如果启用）
        if self.use_local_models:
            self._init_local_models()
        
        # 初始化 LLM（如果提供了 OpenAI API key）
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # 初始化记忆
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # 初始化工具
            self._init_tools()
            
            # 初始化代理
            self._init_agent()
        else:
            logger.info("未提供 OpenAI API 密钥，将仅使用本地模型")
    
    def _init_local_models(self):
        """初始化本地模型"""
        try:
            # 初始化 Qwen3 客户端（用于纯文本）- 性能优化版本
            self.qwen3_client = Qwen3Client(
                device="auto",
                quantization="4bit"  # 启用4bit量化以提升推理速度
            )
            logger.info("Qwen3 客户端初始化成功")
            
            # 初始化 InternVL3_5 客户端（用于图像+文本）
            self.internvl3_5_client = InternVL3_5Client(
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            logger.info("InternVL3_5 客户端初始化成功")
            
        except Exception as e:
            logger.error(f"本地模型初始化失败: {e}")
            raise
    
    def _init_tools(self):
        """初始化工具"""
        # 古董分析工具
        analyze_tool = Tool(
            name="analyze_antique",
            description="分析古董的特征、年代、材质、真伪等信息",
            func=self._analyze_antique_tool
        )
        
        # 价值评估工具
        value_tool = Tool(
            name="estimate_value",
            description="评估古董的市场价值和收藏价值",
            func=self._estimate_value_tool
        )
        
        # 历史查询工具
        history_tool = Tool(
            name="search_history",
            description="查询古董相关的历史背景和文化信息",
            func=self._search_history_tool
        )
        
        # 相似古董搜索工具
        similar_tool = Tool(
            name="find_similar",
            description="查找与当前古董相似的其他古董",
            func=self._find_similar_tool
        )
        
        self.tools = [analyze_tool, value_tool, history_tool, similar_tool]
    
    def _init_agent(self):
        """初始化代理"""
        # 系统消息
        system_message = SystemMessage(content="""
        你是一个专业的古董鉴定专家，具有丰富的古董知识和鉴定经验。
        你的主要职责包括：
        1. 分析古董的特征、年代、材质等信息
        2. 评估古董的真伪和收藏价值
        3. 提供古董相关的历史背景和文化信息
        4. 回答用户关于古董的各种问题
        
        请始终保持专业、客观的态度，基于事实进行分析和回答。
        如果不确定某些信息，请明确说明并建议用户咨询更专业的鉴定机构。
        """)
        
        # 初始化代理
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            agent_kwargs={
                "system_message": system_message,
                "human_message": MessagesPlaceholder(variable_name="chat_history")
            }
        )
    
    def _analyze_antique_tool(self, query: str) -> str:
        """
        古董分析工具
        
        Args:
            query: 分析查询
            
        Returns:
            分析结果
        """
        try:
            # 这里可以集成图像分析结果
            # 目前返回模拟分析结果
            analysis_result = {
                "特征分析": "根据图像特征分析，该古董具有典型的古代工艺特征",
                "年代推测": "基于样式和工艺特征，推测年代为明清时期",
                "材质鉴定": "主要材质为陶瓷/青铜/玉器等",
                "真伪评估": "整体特征符合真品特征，建议进一步专业鉴定",
                "保存状况": "保存状况良好，有轻微磨损"
            }
            
            return json.dumps(analysis_result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"古董分析失败: {e}")
            return f"分析过程中出现错误: {str(e)}"
    
    def _estimate_value_tool(self, query: str) -> str:
        """
        价值评估工具
        
        Args:
            query: 评估查询
            
        Returns:
            评估结果
        """
        try:
            # 模拟价值评估结果
            value_estimate = {
                "市场价值": "根据当前市场行情，预估价值在 10,000-50,000 元之间",
                "收藏价值": "具有较高的收藏价值，建议长期持有",
                "升值潜力": "具有较好的升值潜力，建议关注市场动态",
                "影响因素": "价值受年代、品相、稀有度、市场需求等因素影响",
                "建议": "建议通过专业拍卖行或鉴定机构进行准确评估"
            }
            
            return json.dumps(value_estimate, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"价值评估失败: {e}")
            return f"评估过程中出现错误: {str(e)}"
    
    def _search_history_tool(self, query: str) -> str:
        """
        历史查询工具
        
        Args:
            query: 历史查询
            
        Returns:
            历史信息
        """
        try:
            # 模拟历史信息查询结果
            history_info = {
                "历史背景": "该类型古董在历史上具有重要的文化意义",
                "制作工艺": "采用传统手工制作工艺，体现了古代工匠的精湛技艺",
                "文化价值": "承载着丰富的历史文化内涵，是重要的文化遗产",
                "传承历史": "经过多代传承，见证了历史变迁",
                "相关典故": "与历史事件或人物有关联，具有重要的历史价值"
            }
            
            return json.dumps(history_info, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"历史查询失败: {e}")
            return f"查询过程中出现错误: {str(e)}"
    
    def _find_similar_tool(self, query: str) -> str:
        """
        相似古董搜索工具
        
        Args:
            query: 搜索查询
            
        Returns:
            相似古董信息
        """
        try:
            # 模拟相似古董搜索结果
            similar_antiques = {
                "相似古董": [
                    {
                        "名称": "相似古董1",
                        "年代": "明清时期",
                        "相似度": "85%",
                        "特征": "具有相似的工艺特征和风格"
                    },
                    {
                        "名称": "相似古董2", 
                        "年代": "清代",
                        "相似度": "78%",
                        "特征": "材质和制作工艺相似"
                    }
                ],
                "建议": "建议对比分析这些相似古董，有助于更准确的鉴定"
            }
            
            return json.dumps(similar_antiques, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"相似古董搜索失败: {e}")
            return f"搜索过程中出现错误: {str(e)}"
    
    def analyze_antique(self, image: Optional[Union[str, Image.Image, np.ndarray]] = None,
                       image_features: Optional[np.ndarray] = None, 
                       text_description: str = "") -> Dict[str, Any]:
        """
        分析古董
        
        Args:
            image: 古董图像（优先使用）
            image_features: 图像特征向量（向后兼容）
            text_description: 文本描述
            
        Returns:
            分析结果
        """
        try:
            # 如果使用本地模型
            if self.use_local_models:
                # 如果第一个参数是字符串且不是图像路径，则当作文本描述处理
                if isinstance(image, str) and not (image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))):
                    # 第一个参数是文本描述
                    return self._analyze_with_local_models(None, image)
                else:
                    # 正常的图像+文本描述
                    return self._analyze_with_local_models(image, text_description)
            
            # 使用 LangChain 代理（原有逻辑）
            if not self.agent:
                raise RuntimeError("LangChain 代理未初始化")
                
            # 构建分析查询
            query = f"请分析这个古董：{text_description}"
            if image is not None or image_features is not None:
                query += "（包含图像特征分析）"
            
            # 执行分析
            result = self.agent.run(query)
            
            return {
                "analysis": result,
                "image_features": image_features.tolist() if image_features is not None else None,
                "text_description": text_description,
                "model_used": "langchain_agent"
            }
            
        except Exception as e:
            logger.error(f"古董分析失败: {e}")
            raise
    
    def _analyze_with_local_models(self, image: Optional[Union[str, Image.Image, np.ndarray]] = None,
                                  text_description: str = "") -> Dict[str, Any]:
        """
        使用本地模型分析古董
        
        Args:
            image: 古董图像
            text_description: 文本描述
            
        Returns:
            分析结果
        """
        try:
            # 根据输入类型选择处理策略
            if image is not None:
                # 图像输入：使用 CLIP编码 + 向量库查询 + InternVL3_5模型
                if not self.internvl3_5_client:
                    raise RuntimeError("InternVL3_5 客户端未初始化")
                
                # 1. 使用CLIP编码图像（如果有clip_encoder）
                image_features = None
                similar_antiques = []
                if hasattr(self, 'clip_encoder') and self.clip_encoder:
                    try:
                        image_features = self.clip_encoder.encode_image(image)
                        logger.info("图像CLIP编码完成")
                        
                        # 2. 查询向量数据库寻找相似古董
                        if hasattr(self, 'vector_db') and self.vector_db:
                            similar_antiques = self.vector_db.search_similar_images(
                                query_vector=image_features.tolist(),
                                top_k=3,
                                score_threshold=0.6
                            )
                            logger.info(f"找到 {len(similar_antiques)} 个相似古董")
                    except Exception as e:
                        logger.warning(f"CLIP编码或向量搜索失败: {e}")
                
                # 3. 构建包含相似古董信息的分析提示
                similar_info = ""
                if similar_antiques:
                    similar_info = "\n\n参考相似古董信息：\n"
                    for i, similar in enumerate(similar_antiques[:3], 1):
                        similar_info += f"{i}. 相似度: {similar.get('score', 0):.2f}, 信息: {similar.get('metadata', {}).get('description', '无描述')}\n"
                
                analysis_prompt = f"""
请作为专业的古董鉴定专家，详细分析这件古董。请从以下几个方面进行分析：

1. 外观特征：描述古董的形状、颜色、纹饰等视觉特征
2. 材质鉴定：分析可能的制作材料
3. 工艺特征：观察制作工艺和技法
4. 年代推测：根据风格特征推测可能的历史时期
5. 真伪评估：基于观察到的特征进行初步真伪判断
6. 保存状况：评估古董的保存状态
7. 收藏价值：分析其潜在的收藏和市场价值

用户描述：{text_description if text_description else '无额外描述'}{similar_info}

请提供专业、详细的分析报告。
"""
                
                result = self.internvl3_5_client.chat_about_antique(image, analysis_prompt)
                model_used = "internvl3_5_with_clip_search"
                
                return {
                    "analysis": result,
                    "text_description": text_description,
                    "has_image": True,
                    "image_features": image_features.tolist() if image_features is not None else None,
                    "similar_antiques": similar_antiques,
                    "model_used": model_used
                }
                
            else:
                # 纯文本输入：直接使用 Qwen3模型，不进行向量检索
                if not self.qwen3_client:
                    raise RuntimeError("Qwen3 客户端未初始化")
                
                # 构建文本分析提示
                analysis_prompt = f"""
你是一位专业的古董鉴定专家，请根据用户的描述分析这件古董：

用户描述：{text_description}

请从以下方面进行分析：
1. 根据描述推测古董类型和特征
2. 可能的历史时期和文化背景
3. 制作工艺和材质分析
4. 收藏价值和市场前景
5. 鉴定建议和注意事项

请提供专业的分析意见。
"""
                
                # 使用优化的生成参数提升速度，并添加超时保护（Windows兼容）
                import threading
                import time
                
                result = None
                model_used = "qwen3_text_only"
                timeout_occurred = False
                
                def run_analysis():
                    nonlocal result, timeout_occurred
                    try:
                        if not timeout_occurred:
                            start_time = time.time()
                            result = self.qwen3_client.text_chat(
                                analysis_prompt, 
                                max_length=256  # 减少生成长度以提升速度
                            )
                            end_time = time.time()
                            logger.info(f"Qwen3文本分析耗时: {end_time - start_time:.2f}秒")
                    except Exception as e:
                        if not timeout_occurred:
                            raise e
                
                # 启动分析线程
                analysis_thread = threading.Thread(target=run_analysis)
                analysis_thread.daemon = True
                analysis_thread.start()
                
                # 等待30秒
                analysis_thread.join(timeout=30)
                
                if analysis_thread.is_alive():
                    # 超时处理
                    timeout_occurred = True
                    logger.warning("Qwen3文本分析超时，返回简化结果")
                    result = f"基于描述'{text_description}'的快速分析：这可能是一件具有历史价值的古董，建议进一步专业鉴定。"
                    model_used = "qwen3_timeout"
                
                return {
                    "analysis": result,
                    "text_description": text_description,
                    "has_image": False,
                    "model_used": model_used,
                    "optimization": "文本分析跳过向量检索，直接使用Qwen3模型"
                }
            
        except Exception as e:
            logger.error(f"本地模型分析失败: {e}")
            raise
    
    def chat(self, message: str, image: Optional[Union[str, Image.Image, np.ndarray]] = None) -> str:
        """
        与代理对话
        
        Args:
            message: 用户消息
            image: 可选的图像输入
            
        Returns:
            代理回复
        """
        try:
            # 如果使用本地模型
            if self.use_local_models:
                return self._chat_with_local_models(message, image)
            
            # 使用 LangChain 代理（原有逻辑）
            if not self.agent:
                return "抱歉，LangChain 代理未初始化，请使用本地模型或提供 OpenAI API 密钥。"
                
            response = self.agent.run(message)
            return response
            
        except Exception as e:
            logger.error(f"对话失败: {e}")
            return f"抱歉，处理您的请求时出现错误: {str(e)}"
    
    def _chat_with_local_models(self, message: str, image: Optional[Union[str, Image.Image, np.ndarray]] = None) -> str:
        """
        使用本地模型进行对话
        
        Args:
            message: 用户消息
            image: 可选的图像输入
            
        Returns:
            模型回复
        """
        try:
            # 根据输入类型选择模型
            if image is not None:
                # 有图像输入，使用 InternVL3_5
                if not self.internvl3_5_client:
                    return "抱歉，InternVL3_5 模型未初始化。"
                
                # 构建古董专家对话提示
                expert_prompt = f"""
你是一位经验丰富的古董鉴定专家，请根据用户提供的图像和问题进行专业回答。

用户问题：{message}

请提供专业、准确的回答，如果涉及鉴定，请说明这只是初步分析，建议用户寻求专业机构的进一步鉴定。
"""
                
                return self.internvl3_5_client.chat_about_antique(image, expert_prompt)
                
            else:
                # 纯文本输入，使用 Qwen3
                if not self.qwen3_client:
                    return "抱歉，Qwen3 模型未初始化。"
                
                # 构建古董专家对话提示
                expert_prompt = f"""
你是一位专业的古董鉴定专家，拥有丰富的古董知识和鉴定经验。请针对以下咨询提供专业的回答：

{message}

请提供准确、专业的回答。如果问题涉及具体古董的鉴定，建议用户提供图像或寻求专业机构的帮助。请直接回答问题，不要重复用户的咨询内容。
"""
                
                return self.qwen3_client.text_chat(expert_prompt)
                
        except Exception as e:
            logger.error(f"本地模型对话失败: {e}")
            return f"抱歉，处理您的请求时出现错误: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Returns:
            对话历史列表
        """
        try:
            history = []
            if self.memory and self.memory.chat_memory:
                for message in self.memory.chat_memory.messages:
                    history.append({
                        "role": message.type,
                        "content": message.content
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []
    
    def clear_memory(self):
        """清除记忆"""
        try:
            if self.memory:
                self.memory.clear()
            logger.info("对话记忆已清除")
            
        except Exception as e:
            logger.error(f"清除记忆失败: {e}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        获取代理信息
        
        Returns:
            代理信息字典
        """
        info = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_local_models": self.use_local_models,
            "tools_count": len(self.tools),
            "memory_type": "ConversationBufferMemory" if self.memory else None
        }
        
        # 添加本地模型信息
        if self.use_local_models:
            info["local_models"] = {
                "qwen3_available": self.qwen3_client is not None,
                "internvl3_5_available": self.internvl3_5_client is not None
            }
            
            if self.qwen3_client:
                info["qwen3_info"] = self.qwen3_client.get_model_info()
            
            if self.internvl3_5_client:
                info["internvl3_5_info"] = self.internvl3_5_client.get_model_info()
        
        return info
