"""鉴宝报告格式化器
统一处理AI核心返回的数据，生成标准化的JSON结构供前端渲染
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging

# 获取日志器
logger = logging.getLogger(__name__)

class AppraisalReportFormatter:
    """鉴宝报告格式化器"""
    
    def __init__(self):
        """初始化格式化器"""
        self.dynasty_keywords = {
            '唐': '唐代', '宋': '宋代', '元': '元代', '明': '明代', '清': '清代',
            '汉': '汉代', '魏': '魏代', '晋': '晋代', '隋': '隋代',
            '春秋': '春秋时期', '战国': '战国时期', '秦': '秦代',
            '五代': '五代十国', '辽': '辽代', '金': '金代', '民国': '民国时期'
        }
        
        self.material_keywords = {
            '瓷': '瓷器', '陶': '陶器', '玉': '玉石', '铜': '青铜',
            '铁': '铁器', '金': '黄金', '银': '白银', '木': '木器',
            '竹': '竹器', '石': '石器', '漆': '漆器', '丝': '丝织品',
            '纸': '纸质', '象牙': '象牙制品', '珐琅': '珐琅器'
        }
        
        self.category_keywords = {
            '瓷器': '瓷器', '玉器': '玉器', '青铜器': '青铜器',
            '书画': '书画', '家具': '家具', '杂项': '杂项',
            '佛像': '佛教艺术品', '印章': '印章篆刻',
            '钱币': '古钱币', '织物': '纺织品'
        }
    
    def format_report(self, 
                     ai_analysis: str, 
                     user_query: str = "",
                     image_path: str = None,
                     model_info: Dict = None) -> Dict[str, Any]:
        """格式化鉴宝报告为标准JSON结构
        
        Args:
            ai_analysis: AI分析的原始文本
            user_query: 用户查询
            image_path: 图片路径
            model_info: 模型信息
            
        Returns:
            标准化的鉴宝报告JSON
        """
        try:
            # 基础报告结构 - 修改为前端期望的扁平化格式
            basic_info = self._extract_basic_info(ai_analysis)
            authenticity = self._extract_authenticity_analysis(ai_analysis)
            value_assessment = self._extract_value_assessment(ai_analysis)
            condition = self._extract_condition_evaluation(ai_analysis)
            
            report = {
                "basic_reply": ai_analysis[:200] + "..." if len(ai_analysis) > 200 else ai_analysis,
                "appraisal_report": {
                    "item_name": basic_info.get("item_name", "未知物品"),
                    "category": basic_info.get("category", "未知"),
                    "dynasty": basic_info.get("dynasty_period", "待确定"),
                    "material": basic_info.get("material_type", "待分析"),
                    "authenticity": {
                        "score": authenticity.get("confidence_score", 0),
                        "confidence": authenticity.get("overall_assessment", "待评估"),
                        "analysis": authenticity.get("detailed_analysis", "需要进一步分析")
                    },
                    "value_estimation": {
                        "market_value": value_assessment.get("market_value", "待评估"),
                        "collection_value": value_assessment.get("collection_value", "待评估"),
                        "factors": value_assessment.get("value_factors", ["正在分析中"])
                    },
                    "condition": {
                        "overall": condition.get("overall_condition", "未检测"),
                        "details": condition.get("detailed_condition", "需要详细检查")
                    },
                    "historical_context": self._extract_historical_context_string(ai_analysis),
                    "recommendations": self._extract_recommendations(ai_analysis)
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "confidence_score": self._calculate_confidence_score(ai_analysis),
                    "model_info": model_info or {}
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"格式化报告失败: {e}")
            return self._create_error_report(str(e), ai_analysis)
    
    def _extract_basic_info(self, analysis: str) -> Dict[str, Any]:
        """提取基础信息"""
        basic_info = {
            "item_name": "待确定物品",
            "category": "未分类",
            "dynasty_period": "待确定时期",
            "material_type": "待分析材质",
            "estimated_size": "待测量",
            "overall_description": "正在分析中"
        }
        
        # 提取物品名称（通常在开头或标题中）
        name_patterns = [
            r'(?:藏品名称|物品名称|名称)[：:](.*?)(?:\n|$)',
            r'这是一件(.*?)(?:，|。|\n)',
            r'(?:^|\n)(.*?瓷|.*?器|.*?玉|.*?铜器)(?:，|。|\n)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, analysis, re.MULTILINE | re.IGNORECASE)
            if match:
                basic_info["item_name"] = match.group(1).strip()
                break
        
        # 提取朝代
        for keyword, dynasty in self.dynasty_keywords.items():
            if keyword in analysis:
                basic_info["dynasty_period"] = dynasty
                break
        
        # 提取材质
        for keyword, material in self.material_keywords.items():
            if keyword in analysis:
                basic_info["material_type"] = material
                break
        
        # 提取类别
        for keyword, category in self.category_keywords.items():
            if keyword in analysis:
                basic_info["category"] = category
                break
        
        # 提取整体描述（取前200字符作为概述）
        if len(analysis) > 50:
            # 去除特殊字符，取前200字符
            clean_text = re.sub(r'[*#\-=]+', '', analysis)
            basic_info["overall_description"] = clean_text[:200].strip() + "..."
        
        return basic_info
    
    def _extract_authenticity_analysis(self, analysis: str) -> Dict[str, Any]:
        """提取真伪分析"""
        authenticity = {
            "authenticity_score": 0,
            "confidence_level": "低",
            "key_indicators": [],
            "risk_factors": [],
            "expert_opinion": "需要进一步专业鉴定"
        }
        
        # 提取评分（寻找数字+分的模式）
        score_patterns = [
            r'(?:真伪|评分|得分)[：:]?\s*(\d+)(?:分|/100)',
            r'(\d+)(?:分|/100)(?:的真伪)',
            r'评估为\s*(\d+)(?:%|分)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, analysis, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                authenticity["authenticity_score"] = min(score, 100)
                break
        
        # 根据评分设置可信度
        score = authenticity["authenticity_score"]
        if score >= 80:
            authenticity["confidence_level"] = "高"
        elif score >= 60:
            authenticity["confidence_level"] = "中"
        else:
            authenticity["confidence_level"] = "低"
        
        # 提取关键指标
        indicator_keywords = ['工艺', '纹饰', '胎质', '釉色', '款识', '器型', '包浆', '老化']
        for keyword in indicator_keywords:
            if keyword in analysis:
                authenticity["key_indicators"].append(keyword)
        
        # 提取风险因素
        risk_keywords = ['仿制', '现代', '新仿', '疑点', '不符', '异常']
        for keyword in risk_keywords:
            if keyword in analysis:
                authenticity["risk_factors"].append(f"发现{keyword}特征")
        
        return authenticity
    
    def _extract_value_assessment(self, analysis: str) -> Dict[str, Any]:
        """提取价值评估"""
        value_assessment = {
            "market_value_range": "待专业评估",
            "collection_value": "待确定",
            "investment_potential": "需谨慎评估",
            "value_factors": [],
            "market_trends": "请咨询专业机构"
        }
        
        # 提取价值相关信息
        value_keywords = ['价值', '价格', '市场', '收藏', '投资']
        for keyword in value_keywords:
            if keyword in analysis:
                value_assessment["value_factors"].append(f"涉及{keyword}评估")
        
        # 寻找具体价格信息
        price_patterns = [
            r'(?:价值|价格).*?(\d+(?:万|千|百)?元)',
            r'(\d+(?:万|千|百)?元).*?(?:价值|价格)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, analysis, re.IGNORECASE)
            if match:
                value_assessment["market_value_range"] = f"参考价格：{match.group(1)}"
                break
        
        return value_assessment
    
    def _extract_condition_evaluation(self, analysis: str) -> Dict[str, Any]:
        """提取状况评估"""
        condition = {
            "overall_condition": "待检查",
            "preservation_score": 0,
            "damage_assessment": [],
            "restoration_history": "未知",
            "care_recommendations": []
        }
        
        # 提取保存状况关键词
        condition_keywords = {
            '完好': 90, '良好': 80, '一般': 60, '较差': 40, '破损': 20
        }
        
        for keyword, score in condition_keywords.items():
            if keyword in analysis:
                condition["overall_condition"] = keyword
                condition["preservation_score"] = score
                break
        
        # 提取损坏信息
        damage_keywords = ['裂纹', '缺失', '磨损', '褪色', '变形', '修复']
        for keyword in damage_keywords:
            if keyword in analysis:
                condition["damage_assessment"].append(f"发现{keyword}")
        
        return condition
    
    def _extract_historical_context(self, analysis: str) -> Dict[str, Any]:
        """提取历史背景"""
        historical_context = {
            "historical_period": "待确定",
            "cultural_significance": "需要研究",
            "production_background": "待考证",
            "similar_artifacts": [],
            "research_references": []
        }
        
        # 提取历史相关信息
        history_keywords = ['历史', '文化', '背景', '意义', '传统', '工艺']
        context_text = ""
        
        for keyword in history_keywords:
            if keyword in analysis:
                # 提取包含关键词的句子
                sentences = re.split(r'[。！？\n]', analysis)
                for sentence in sentences:
                    if keyword in sentence and len(sentence) > 10:
                        context_text += sentence + "。"
        
        if context_text:
            historical_context["cultural_significance"] = context_text[:300] + "..."
        
        return historical_context
    
    def _extract_historical_context_string(self, analysis: str) -> str:
        """提取历史背景信息并返回字符串格式"""
        # 提取历史相关信息
        history_keywords = ['历史', '文化', '背景', '意义', '传统', '工艺', '朝代', '时期']
        context_sentences = []
        
        # 分割文本为句子
        sentences = re.split(r'[。！？\n]', analysis)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # 过滤太短的句子
                # 检查是否包含历史相关关键词
                for keyword in history_keywords:
                    if keyword in sentence:
                        context_sentences.append(sentence)
                        break
        
        if context_sentences:
            # 合并句子，限制长度
            context_text = '。'.join(context_sentences[:3])  # 最多取前3句
            if len(context_text) > 200:
                context_text = context_text[:200] + '...'
            return context_text + '。'
        else:
            return "正在分析历史背景和文化价值，建议咨询专业文物专家获取更详细的历史信息。"
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """提取专业建议"""
        recommendations = [
            "建议寻求专业文物鉴定机构的权威鉴定",
            "如需交易，请通过正规渠道并确保来源合法",
            "妥善保存，避免阳光直射和潮湿环境"
        ]
        
        # 从分析中提取建议
        suggestion_keywords = ['建议', '推荐', '应该', '需要', '注意']
        for keyword in suggestion_keywords:
            if keyword in analysis:
                sentences = re.split(r'[。！？\n]', analysis)
                for sentence in sentences:
                    if keyword in sentence and len(sentence) > 10:
                        recommendations.append(sentence.strip())
        
        return list(set(recommendations))  # 去重
    
    def _get_standard_disclaimer(self) -> Dict[str, str]:
        """获取标准免责声明"""
        return {
            "limitation_statement": "本报告基于AI分析，结论存在一定局限性，仅供参考",
            "no_guarantee_statement": "本报告不对文物的真伪、价值承担法律责任",
            "professional_advice": "如需权威鉴定，请咨询专业文物鉴定机构",
            "legal_compliance": "请确保文物来源合法，遵守相关法律法规"
        }
    
    def _calculate_confidence_score(self, analysis: str) -> int:
        """计算分析可信度评分"""
        score = 50  # 基础分数
        
        # 根据分析内容的详细程度调整分数
        if len(analysis) > 500:
            score += 20
        elif len(analysis) > 200:
            score += 10
        
        # 根据专业术语的使用情况调整分数
        professional_terms = ['工艺', '胎质', '釉色', '纹饰', '款识', '器型', '包浆']
        term_count = sum(1 for term in professional_terms if term in analysis)
        score += min(term_count * 5, 20)
        
        # 根据具体数据的存在调整分数
        if re.search(r'\d+(?:年|世纪|朝代)', analysis):
            score += 10
        
        return min(score, 100)
    
    def _create_error_report(self, error_msg: str, raw_analysis: str = "") -> Dict[str, Any]:
        """创建错误报告"""
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error_message": error_msg,
            "appraisal_report": {
                "basic_info": {
                    "item_name": "分析失败",
                    "category": "错误",
                    "dynasty_period": "无法确定",
                    "material_type": "无法分析",
                    "overall_description": f"分析过程中出现错误：{error_msg}"
                }
            },
            "raw_analysis": raw_analysis,
            "confidence_score": 0
        }

# 创建全局格式化器实例
report_formatter = AppraisalReportFormatter()

def format_appraisal_report(ai_analysis: str, 
                           user_query: str = "",
                           image_path: str = None,
                           model_info: Dict = None) -> Dict[str, Any]:
    """格式化鉴宝报告的便捷函数"""
    return report_formatter.format_report(ai_analysis, user_query, image_path, model_info)