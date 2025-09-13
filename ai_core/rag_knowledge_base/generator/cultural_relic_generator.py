#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文物生成器 - 简化版
生成文物图片和对应的文字描述
"""

import os
import json
import yaml
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

try:
    from dotenv import load_dotenv
    # 查找项目根目录的.env文件
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent  # 回到项目根目录
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    # 如果没有安装python-dotenv，尝试手动加载.env文件
    def load_env_file():
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent  # 回到项目根目录
        env_file = project_root / '.env'
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    load_env_file()

from data_structures import CulturalRelicInfo, RelicCategory, Dynasty
from image_generator import ImageGenerator


class CulturalRelicGenerator:
    """文物生成器主类"""
    
    def __init__(self, config_path: str = "config_cultural_relic.yaml"):
        """初始化生成器"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # 设置输出目录（使用相对路径）
        self.images_dir = Path("../images")
        self.metadata_dir = Path("../metadata")
        
        # 创建目录
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化图片生成器
        self.image_generator = ImageGenerator(config=self.config)
        
        print(f"✅ 文物生成器初始化完成")
        print(f"📁 图片保存目录: {self.images_dir}")
        print(f"📄 描述保存目录: {self.metadata_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # 替换环境变量
            config_content = self._replace_env_variables(config_content)
            
            # 解析YAML
            config = yaml.safe_load(config_content)
            return config
        except Exception as e:
            print(f"⚠️  配置文件加载失败: {e}")
            return self._get_default_config()
    
    def _replace_env_variables(self, content: str) -> str:
        """替换配置文件中的环境变量"""
        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))  # 如果环境变量不存在，保持原样
        
        # 匹配 ${VAR_NAME} 格式的环境变量
        return re.sub(r'\$\{([^}]+)\}', replace_var, content)

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'model': {
                'doubao_image': {
                    'model_name': 'doubao-xl',
                    'api_key': 'your_actual_doubao_api_key_here',
                    'base_url': 'https://ark.cn-beijing.volces.com/api/v3'
                }
            }
        }
    
    def generate_relic_info(self) -> CulturalRelicInfo:
        """生成随机文物信息"""
        # 文物名称模板 - 按朝代和类别组织
        relic_data = {
            Dynasty.SHANG: {
                RelicCategory.BRONZE: ["四羊方尊", "司母戊鼎", "商代青铜爵", "饕餮纹鼎", "青铜觚"],
                RelicCategory.JADE: ["商代玉璧", "玉戈", "玉琮", "龙形玉佩", "玉刀"]
            },
            Dynasty.TANG: {
                RelicCategory.POTTERY: ["唐三彩马", "唐三彩骆驼", "彩绘陶俑", "胡人俑", "仕女俑"],
                RelicCategory.PORCELAIN: ["唐代白瓷", "长沙窑瓷器", "邢窑白瓷", "越窑青瓷", "唐代花瓷"]
            },
            Dynasty.SONG: {
                RelicCategory.PORCELAIN: ["汝窑天青釉盘", "官窑青瓷", "哥窑开片瓷", "定窑白瓷", "钧窑紫斑瓷"],
                RelicCategory.JADE: ["宋代玉如意", "白玉观音", "青玉花鸟佩", "玉带钩", "玉簪"]
            },
            Dynasty.MING: {
                RelicCategory.PORCELAIN: ["青花瓷瓶", "成化斗彩", "永乐甜白瓷", "宣德青花", "嘉靖五彩瓷"],
                RelicCategory.LACQUERWARE: ["明代雕漆盒", "描金漆器", "剔红漆盘", "黑漆嵌螺钿", "朱漆木雕"]
            },
            Dynasty.QING: {
                RelicCategory.PORCELAIN: ["康熙青花", "雍正粉彩", "乾隆珐琅彩", "景德镇官窑", "清代斗彩"],
                RelicCategory.JADE: ["清代白玉", "翡翠如意", "和田玉佩", "玉山子", "碧玉花瓶"]
            }
        }
        
        # 随机选择朝代
        dynasty = random.choice(list(Dynasty))
        
        # 获取该朝代可用的类别
        available_categories = list(relic_data.get(dynasty, {}).keys())
        if not available_categories:
            # 如果没有特定朝代的数据，使用通用数据
            category = random.choice(list(RelicCategory))
            name = f"{dynasty.value}时期文物"
        else:
            category = random.choice(available_categories)
            names = relic_data[dynasty][category]
            name = random.choice(names)
        
        return CulturalRelicInfo(
            name=name,
            category=category,
            dynasty=dynasty
        )
    
    def generate_detailed_description(self, relic: CulturalRelicInfo) -> Dict[str, str]:
        """生成详细的文物描述"""
        # 详细的描述数据库（限制每个字段最多50字符）
        detailed_info = {
            "四羊方尊": {
                "appearance": "通高58.3厘米，重近35公斤，方形器身，四角各有一羊首"[:50],
                "historical_background": "商朝晚期青铜礼器，1938年出土于湖南宁乡，是商朝青铜器的杰作"[:50],
                "cultural_significance": "代表了商代青铜文明的最高成就，是中国古代青铜艺术的瑰宝"[:50]
            },
            "司母戊鼎": {
                "appearance": "通高133厘米，口长110厘米，口宽79厘米，重达832.84公斤"[:50],
                "historical_background": "商王武丁时期铸造，用于祭祀武丁的母亲戊，1939年出土于河南安阳殷墟"[:50],
                "cultural_significance": "体现了商代王室的威严和青铜铸造技术的巅峰，是中华文明的重要象征"[:50]
            },
            "唐三彩马": {
                "appearance": "马身施黄、绿、白三色釉，造型生动，马首高昂，四肢健壮，鞍具精美"[:50],
                "historical_background": "唐代盛行的陶瓷工艺品，多作为陪葬品，反映了唐代丝绸之路的繁荣"[:50],
                "cultural_significance": "展现了唐代开放包容的文化特色和高超的陶瓷烧制技艺，是唐代文化"[:50]
            },
            "汝窑天青釉盘": {
                "appearance": "釉色如天青色，温润如玉，开片细密，造型简洁优雅，胎质细腻"[:50],
                "historical_background": "北宋时期汝窑烧制，专供宫廷使用，传世品极少，是宋代五大名窑之首"[:50],
                "cultural_significance": "代表了宋代瓷器烧制技术的最高水平，体现了宋人追求自然、含蓄"[:50]
            },
            "青花瓷瓶": {
                "appearance": "瓶身修长，青花纹饰清晰，色泽浓艳，图案精美，胎质洁白，釉面光润"[:50],
                "historical_background": "明代景德镇窑烧制，青花瓷技术在明代达到成熟，成为中国瓷器的代表"[:50],
                "cultural_significance": "体现了明代瓷器工艺的精湛技术，是中外文化交流的重要载体"[:50]
            }
        }
        
        # 获取具体文物信息，如果没有则使用通用模板
        specific_info = detailed_info.get(relic.name)
        
        if specific_info:
            return {
                "name": relic.name[:50],
                "category": relic.category.value[:50],
                "dynasty": relic.dynasty.value[:50],
                "appearance": specific_info["appearance"][:50],
                "historical_background": specific_info["historical_background"][:50],
                "cultural_significance": specific_info["cultural_significance"][:50],
                "preservation_status": "保存完好，细节清晰可见"[:50],
                "collection_value": "具有极高的收藏价值和研究意义"[:50]
            }
        else:
            # 通用模板（限制每个字段最多50字符）
            appearance_templates = {
                RelicCategory.BRONZE: f"青铜材质，{relic.dynasty.value}时期典型风格，纹饰精美，工艺精湛"[:50],
                RelicCategory.POTTERY: f"陶土烧制，{relic.dynasty.value}时期特色鲜明，色彩丰富，造型生动"[:50],
                RelicCategory.PORCELAIN: f"瓷质细腻，{relic.dynasty.value}时期工艺特点明显，釉色莹润"[:50],
                RelicCategory.JADE: f"玉质温润，{relic.dynasty.value}时期雕工精细，造型典雅，寓意深远"[:50],
                RelicCategory.LACQUERWARE: f"漆质光亮，{relic.dynasty.value}时期装饰风格，色彩鲜艳"[:50]
            }
            
            historical_templates = {
                Dynasty.SHANG: f"{relic.name}是商朝时期的重要{relic.category.value}文物，反映了当时高超的手工艺技术"[:50],
                Dynasty.TANG: f"{relic.name}是唐朝盛世的文化瑰宝，体现了开放包容的时代精神"[:50],
                Dynasty.SONG: f"{relic.name}体现了宋代文人雅士的审美追求，展现了精致典雅的艺术风格"[:50],
                Dynasty.MING: f"{relic.name}是明朝时期的珍贵{relic.category.value}，代表了当时最高水平的工艺技术"[:50],
                Dynasty.QING: f"{relic.name}是清朝时期的精美器物，融合了满汉文化特色"[:50]
            }
            
            return {
                "name": relic.name[:50],
                "category": relic.category.value[:50],
                "dynasty": relic.dynasty.value[:50],
                "appearance": appearance_templates.get(relic.category, f"{relic.dynasty.value}时期{relic.category.value}，造型独特，工艺精美")[:50],
                "historical_background": historical_templates.get(relic.dynasty, f"{relic.name}具有重要的历史价值")[:50],
                "cultural_significance": f"{relic.name}是{relic.dynasty.value}时期的代表性{relic.category.value}文物，具有重要的历史、艺术和文化价值。"[:50],
                "preservation_status": "保存完好，细节清晰可见"[:50],
                "collection_value": "具有极高的收藏价值和研究意义"[:50]
            }
    
    def generate_image_prompt(self, relic: CulturalRelicInfo, description: Dict[str, str]) -> str:
        """生成图像提示词"""
        prompt = f"一件{relic.dynasty.value}时期的{relic.name}，{relic.category.value}类文物。\n\n"
        prompt += f"外观特征：{description['appearance']}\n\n"
        prompt += f"风格要求：{relic.dynasty.value}风格，工艺精湛，{relic.category.value}材质\n\n"
        prompt += "画面要求：\n"
        prompt += "- 文物居中展示，背景简洁\n"
        prompt += "- 光线柔和，突出文物细节\n"
        prompt += "- 高清晰度，展现工艺精美\n"
        prompt += "- 符合博物馆展品摄影标准\n"
        prompt += f"- 体现{relic.dynasty.value}时期的历史韵味\n\n"
        prompt += "注意：图像应真实反映文物特征，避免过度艺术化处理。"
        
        return prompt
    
    def save_metadata(self, relic_id: str, description: Dict[str, str]) -> str:
        """保存文物描述到metadata目录"""
        metadata_file = self.metadata_dir / f"{relic_id}.json"
        
        metadata = {
            "id": relic_id,
            "generated_at": datetime.now().isoformat(),
            "description": description
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return str(metadata_file)
    
    def generate_single_relic(self) -> Dict[str, Any]:
        """生成单个文物（图片+描述）"""
        print("\n🎨 开始生成文物...")
        
        # 1. 生成文物信息
        relic = self.generate_relic_info()
        print(f"📝 生成文物信息: {relic.name} ({relic.dynasty.value} - {relic.category.value})")
        
        # 2. 生成详细描述
        description = self.generate_detailed_description(relic)
        print(f"📄 生成文物描述完成")
        
        # 3. 生成图像提示词
        image_prompt = self.generate_image_prompt(relic, description)
        
        # 4. 生成图片
        print(f"🖼️  开始生成图片...")
        image_result = self.image_generator.generate_image_with_doubao(image_prompt)
        
        # 5. 创建文物ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        relic_id = f"{relic.dynasty.value}_{relic.category.value}_{timestamp}"
        
        result = {
            "id": relic_id,
            "relic_info": relic,
            "description": description,
            "image_prompt": image_prompt,
            "image_generation": image_result,
            "success": False,
            "image_path": "",
            "metadata_path": ""
        }
        
        # 6. 下载图片（如果生成成功）
        if image_result.get("success") and image_result.get("image_url"):
            print(f"💾 开始下载图片...")
            download_result = self.image_generator.download_image_from_url(
                image_result["image_url"], 
                relic_id
            )
            
            if download_result.get("success"):
                result["image_path"] = download_result["local_path"]
                print(f"✅ 图片保存成功: {download_result['local_path']}")
            else:
                print(f"❌ 图片下载失败: {download_result.get('error', '未知错误')}")
        else:
            print(f"❌ 图片生成失败: {image_result.get('error', '未知错误')}")
        
        # 7. 保存描述文件
        print(f"💾 保存文物描述...")
        metadata_path = self.save_metadata(relic_id, description)
        result["metadata_path"] = metadata_path
        print(f"✅ 描述保存成功: {metadata_path}")
        
        # 8. 更新成功状态
        result["success"] = bool(result["image_path"] and result["metadata_path"])
        
        return result
    
    def generate_multiple_relics(self, count: int = 5) -> List[Dict[str, Any]]:
        """生成多个文物"""
        print(f"\n🚀 开始批量生成 {count} 个文物...")
        print("=" * 50)
        
        results = []
        success_count = 0
        
        for i in range(count):
            print(f"\n📦 生成第 {i+1}/{count} 个文物")
            try:
                result = self.generate_single_relic()
                results.append(result)
                
                if result["success"]:
                    success_count += 1
                    print(f"✅ 第 {i+1} 个文物生成成功")
                else:
                    print(f"⚠️  第 {i+1} 个文物部分失败")
                    
            except Exception as e:
                print(f"❌ 第 {i+1} 个文物生成失败: {e}")
                results.append({
                    "success": False,
                    "error": str(e)
                })
        
        print(f"\n📊 批量生成完成")
        print(f"✅ 成功: {success_count}/{count}")
        print(f"❌ 失败: {count - success_count}/{count}")
        
        return results


def main():
    """主函数"""
    print("文物生成器 - 简化版")
    print("=" * 30)
    
    try:
        # 初始化生成器
        generator = CulturalRelicGenerator()
        
        # 用户选择
        print("\n请选择操作：")
        print("1. 生成单个文物")
        print("2. 批量生成文物")
        
        choice = input("\n请输入选择 (1-2): ").strip()
        
        if choice == "1":
            result = generator.generate_single_relic()
            print(f"\n🎉 生成完成！")
            if result["success"]:
                print(f"📁 图片路径: {result['image_path']}")
                print(f"📄 描述路径: {result['metadata_path']}")
        
        elif choice == "2":
            count = input("请输入生成数量 (默认5): ").strip()
            count = int(count) if count.isdigit() else 5
            
            results = generator.generate_multiple_relics(count)
            
            print(f"\n🎉 批量生成完成！")
            success_results = [r for r in results if r.get("success")]
            print(f"📊 成功生成 {len(success_results)} 个文物")
        
        else:
            print("❌ 无效选择")
    
    except KeyboardInterrupt:
        print("\n👋 用户取消操作")
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")


if __name__ == "__main__":
    main()