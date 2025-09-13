"""
AI 鉴宝师后端服务（简化版）
专注于核心的AI鉴宝功能，图片本地存储
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io

# 初始化基本日志（防止循环导入）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend_app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 在日志初始化后再导入自定义模块
try:
    # 添加项目根目录到系统路径
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.logger_config import get_backend_logger, get_system_logger
    
    # 更新为更好的日志器
    logger = get_backend_logger('main_app')
    system_logger = get_system_logger()
    logger.info("日志系统初始化成功")
except Exception as e:
    logger.warning(f"日志系统初始化失败，使用默认日志: {e}")
    system_logger = logger

# 导入AI核心模块
try:
    from ai_core import AICore
    logger.info("AI核心模块导入成功")
except Exception as e:
    logger.error(f"AI核心模块导入失败: {e}")
    AICore = None

# 导入报告格式化器
try:
    from services.report_formatter import format_appraisal_report
    REPORT_FORMATTER_AVAILABLE = True
    logger.info("报告格式化器导入成功")
except ImportError as e:
    REPORT_FORMATTER_AVAILABLE = False
    logger.warning(f"报告格式化器导入失败: {e}")

app = Flask(__name__)
CORS(app)

# 配置
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
# 禁用响应缓冲以支持流式输出
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

# 初始化AI核心
ai_core = None

def handle_stream_response(input_type, question=None, image_path=None):
    """处理流式响应"""
    def generate_stream():
        try:
            if input_type == 'text':
                logger.info(f"开始流式文本分析: {question[:100]}..." if len(question) > 100 else f"开始流式文本分析: {question}")
                
                # 发送开始信号
                yield f"data: [START]\n\n"
                
                # 使用AI核心进行流式分析
                if ai_core is None:
                    yield f"data: {{\"content\": \"AI服务未初始化，正在处理您的问题...\"}}\n\n"
                    yield f"data: {{\"content\": \"\\n\\n这是一个基础响应，如需详细分析请稍后重试.\"}}\n\n"
                else:
                    try:
                        analysis_content = ""  # 初始化变量
                        for text_chunk in ai_core.analyze_antique_text_stream(question):
                            if text_chunk and text_chunk.strip():
                                analysis_content += text_chunk
                                # 立即发送内容块
                                import json
                                content_data = {"type": "content", "text": text_chunk}
                                yield f"data: {json.dumps(content_data, ensure_ascii=False)}\n\n"
                                import sys
                                sys.stdout.flush()
                        
                        # 生成结构化报告
                        structured_report = generate_structured_report(analysis_content, question, 'text')
                        import json
                        yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(structured_report, ensure_ascii=False)}}}\n\n"
                        
                    except Exception as e:
                        error_report = {
                            "basic_reply": f"分析过程中出现错误: {str(e)}",
                            "appraisal_report": {
                                "item_name": "分析失败",
                                "category": "错误",
                                "error": str(e)
                            }
                        }
                        import json
                        yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(error_report, ensure_ascii=False)}}}\n\n"
                
                # 发送结束信号
                yield f"data: [DONE]\n\n"
                
            elif input_type == 'image':
                logger.info(f"开始流式图像分析: {question[:100]}..." if len(question) > 100 else f"开始流式图像分析: {question}")
                
                # 发送开始信号
                yield f"data: [START]\n\n"
                
                # 使用AI核心进行流式图像分析
                if ai_core is None:
                    yield f"data: {{\"content\": \"AI服务未初始化，正在处理您的图片...\"}}\n\n"
                    yield f"data: {{\"content\": \"图片已上传成功，这是一个基础响应。\"}}\n\n"
                else:
                    try:
                        for text_chunk in ai_core.analyze_antique_image_stream(image_path, question):
                            if text_chunk and text_chunk.strip():
                                import json
                                content_data = {"type": "content", "text": text_chunk}
                                yield f"data: {json.dumps(content_data, ensure_ascii=False)}\n\n"
                                import sys
                                sys.stdout.flush()
                    except Exception as e:
                        yield f"data: {{\"content\": \"分析过程中出现错误: {str(e)}\"}}\n\n"
                
                # 发送结束信号
                yield f"data: [DONE]\n\n"
                
        except Exception as e:
            logger.error(f"流式响应处理失败: {e}")
            yield f"data: {{\"error\": \"处理失败: {str(e)}\"}}\n\n"
    
    return Response(generate_stream(), mimetype='text/event-stream; charset=utf-8', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',  # 禁用nginx缓冲
        'Transfer-Encoding': 'chunked',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
    })

def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_ai_core():
    """初始化AI核心（支持 llama-cpp-python）"""
    global ai_core
    
    logger.info("开始初始化AI核心...")
    
    try:
        # 检查AICore类是否可用
        if AICore is None:
            logger.error("AICore类未成功导入")
            return False
            
        # 检查环境变量
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY 环境变量未设置，LangChain功能将不可用")
        
        # 检查Milvus连接参数
        milvus_host = os.getenv('MILVUS_HOST', 'localhost')
        milvus_port = os.getenv('MILVUS_PORT', '19530')
        logger.info(f"使用Milvus连接: {milvus_host}:{milvus_port}")
        
        # 检查SmolVLM2模型路径
        smolvlm2_model_path = os.getenv('SMOLVLM2_MODEL_PATH', '')
        max_tokens = int(os.getenv('MAX_TOKENS', '512'))
        temperature = float(os.getenv('TEMPERATURE', '0.7'))
        
        if smolvlm2_model_path:
            logger.info(f"使用本地SmolVLM2模型: {smolvlm2_model_path}")
        else:
            logger.info("使用默认SmolVLM2配置")
        
        logger.info(f"Max Tokens: {max_tokens}, Temperature: {temperature}")
        
        # 初始化AI核心
        logger.info("正在创建AICore实例...")
        ai_core = AICore(
            openai_api_key=openai_api_key or "",
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            internvl3_5_model_path=smolvlm2_model_path,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        logger.info("✅ AI核心初始化成功")
        return True
        
    except ImportError as e:
        logger.error(f"AI核心模块导入失败: {e}")
        return False
    except Exception as e:
        logger.error(f"AI核心初始化失败: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False

def save_analysis_result(image_path, analysis_result):
    """保存分析结果到本地文件"""
    try:
        # 生成结果文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"analysis_{timestamp}.json"
        result_path = os.path.join(app.config['DATA_FOLDER'], result_filename)
        
        # 保存结果
        result_data = {
            'timestamp': timestamp,
            'image_path': image_path,
            'analysis': analysis_result
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析结果已保存: {result_path}")
        return result_filename
    except Exception as e:
        logger.error(f"保存分析结果失败: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    logger.debug("健康检查请求")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ai_core_ready': ai_core is not None
    })

@app.route('/api/debug/status', methods=['GET'])
def debug_status():
    """调试状态接口"""
    logger.info("调试状态检查")
    
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'ai_core': {
                'initialized': ai_core is not None,
                'type': str(type(ai_core)) if ai_core else None
            },
            'environment': {
                'openai_key_set': bool(os.getenv('OPENAI_API_KEY')),
                'milvus_host': os.getenv('MILVUS_HOST', 'localhost'),
                'milvus_port': os.getenv('MILVUS_PORT', '19530'),
                'smolvlm2_model_path': os.getenv('SMOLVLM2_MODEL_PATH', ''),
                'max_tokens': os.getenv('MAX_TOKENS', '512'),
                'temperature': os.getenv('TEMPERATURE', '0.7')
            },
            'folders': {
                'upload_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
                'data_exists': os.path.exists(app.config['DATA_FOLDER'])
            }
        }
        
        # 如果AI核心已初始化，获取更多信息
        if ai_core:
            try:
                ai_status = ai_core.get_system_status()
                status['ai_core']['system_status'] = ai_status
            except Exception as e:
                status['ai_core']['status_error'] = str(e)
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"获取调试状态失败: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/v1/appraisal', methods=['POST'])
def appraisal_antique():
    """鉴赏古董（符合项目文档要求的接口）"""
    logger.info("收到古董鉴赏请求")
    
    try:
        # 检查请求类型：JSON（文本）或 multipart/form-data（图片+文本）
        content_type = request.content_type
        
        if content_type and 'application/json' in content_type:
            # 处理纯文本问题
            data = request.get_json()
            
            # 支持多种字段名：question, text
            question = data.get('question') or data.get('text')
            if not question:
                logger.warning("JSON请求格式错误，缺少question或text字段")
                return jsonify({'error': '请提供问题文本'}), 400
            
            # 检查是否请求流式响应
            is_stream = data.get('stream', False)
            if is_stream:
                return handle_stream_response('text', question=question)
            
            logger.info(f"收到文本问题: {question[:100]}..." if len(question) > 100 else f"收到文本问题: {question}")
            
            # 简化的文本分析响应（避免AI模型超时）
            start_time = datetime.now()
            
            try:
                # 检查AI核心是否就绪
                if ai_core is None:
                    logger.warning("AI服务未初始化，使用默认响应")
                    analysis_result = f"您的问题已收到：{question}\n\n正在处理中，这是一个基础响应。如需详细分析，请稍后重试。"
                else:
                    analysis_result = ai_core.analyze_antique_text(question)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"文本分析完成，耗时: {duration:.2f}秒")
                
            except Exception as analysis_error:
                logger.warning(f"AI分析失败，使用默认响应: {analysis_error}")
                analysis_result = f"您的问题已收到：{question}\n\n正在处理中，这是一个基础响应。如需详细分析，请稍后重试。"
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
            
            # 保存分析结果
            try:
                result_filename = save_analysis_result(f"text_question_{datetime.now().strftime('%Y%m%d_%H%M%S')}", analysis_result)
                logger.info(f"分析结果已保存: {result_filename}")
            except Exception as save_error:
                logger.warning(f"保存分析结果失败: {save_error}")
                result_filename = None
            
            # 返回Markdown格式的鉴赏报告
            response_data = {
                'success': True,
                'input_type': 'text',
                'question': question,
                'report': analysis_result,  # Markdown格式的鉴赏报告
                'result_file': result_filename,
                'analysis_duration': duration
            }
            
            logger.info("文本问题分析成功完成")
            return jsonify(response_data)
            
        else:
            # 处理图片+文本的混合输入
            if 'image' not in request.files:
                logger.warning("请求中缺少图片文件")
                return jsonify({'error': '没有上传图片'}), 400
            
            file = request.files['image']
            if file.filename == '':
                logger.warning("文件名为空")
                return jsonify({'error': '没有选择文件'}), 400
            
            if not allowed_file(file.filename):
                logger.warning(f"不支持的文件类型: {file.filename}")
                return jsonify({'error': '不支持的文件类型'}), 400
            
            # 保存上传的图片
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            logger.info(f"保存图片到: {filepath}")
            file.save(filepath)
            
            # 获取问题文本和流式参数
            question = request.form.get('question', '')
            is_stream = request.form.get('stream', '').lower() in ['true', '1', 'yes']
            logger.info(f"用户问题: {question[:100]}..." if len(question) > 100 else f"用户问题: {question}")
            
            # 如果请求流式响应，直接返回流式处理
            if is_stream:
                return handle_stream_response('image', question=question, image_path=filepath)
            
            # 简化的图片分析响应（避免AI模型超时）
            start_time = datetime.now()
            
            try:
                # 检查AI核心是否就绪
                if ai_core is None:
                    logger.warning("AI服务未初始化，使用默认响应")
                    analysis_result = f"图片已接收并保存为 {filename}。正在分析中，请稍后查看详细报告。"
                else:
                    analysis_result = ai_core.analyze_antique_image(
                        image=filepath,
                        description=question
                    )
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"图像+问题分析完成，耗时: {duration:.2f}秒")
                
            except Exception as analysis_error:
                logger.warning(f"AI分析失败，使用默认响应: {analysis_error}")
                analysis_result = f"图片已接收并保存为 {filename}。正在分析中，请稍后查看详细报告。"
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
            
            # 保存分析结果
            try:
                result_filename = save_analysis_result(filepath, analysis_result)
                logger.info(f"分析结果已保存: {result_filename}")
            except Exception as save_error:
                logger.warning(f"保存分析结果失败: {save_error}")
                result_filename = None
            
            # 返回Markdown格式的鉴赏报告
            response_data = {
                'success': True,
                'input_type': 'multimodal',
                'image_filename': filename,
                'question': question,
                'report': analysis_result,  # Markdown格式的鉴赏报告
                'result_file': result_filename,
                'analysis_duration': duration
            }
            
            logger.info(f"图像+问题分析成功完成: {filename}")
            return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"鉴赏接口失败: {e}")
        import traceback
        logger.error(f"接口错误详情: {traceback.format_exc()}")
        return jsonify({
            'error': f'鉴赏失败: {str(e)}',
            'type': type(e).__name__
        }), 500

@app.route('/api/v1/appraisal/stream', methods=['POST'])
def appraisal_antique_stream():
    """流式鉴赏古董（Server-Sent Events）"""
    logger.info("收到流式古董鉴赏请求")
    
    # 在生成器外部获取请求数据，避免Flask上下文问题
    try:
        content_type = request.content_type
        
        if content_type and 'application/json' in content_type:
            # 处理纯文本问题
            data = request.get_json()
            if not data or 'question' not in data:
                return jsonify({"error": "请提供问题文本"}), 400
            
            question = data['question']
            input_type = 'text'
            image_path = None
            logger.info(f"收到流式文本问题: {question[:100]}..." if len(question) > 100 else f"收到流式文本问题: {question}")
        else:
            # 处理图片+文本的混合输入
            if 'image' not in request.files:
                return jsonify({"error": "没有上传图片"}), 400
            
            file = request.files['image']
            if file.filename == '' or not allowed_file(file.filename):
                return jsonify({"error": "无效的图片文件"}), 400
            
            # 保存上传的图片
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            question = request.form.get('question', '')
            input_type = 'image'
            image_path = filepath
            logger.info(f"收到流式图片问题: {question[:100]}..." if len(question) > 100 else f"收到流式图片问题: {question}")
    
    except Exception as e:
        logger.error(f"请求解析失败: {e}")
        return jsonify({"error": f"请求解析失败: {str(e)}"}), 400
    
    # 调用流式生成器
    def generate_stream(input_type, question, image_path=None):
        try:
            if input_type == 'text':
                # 发送开始信号
                yield f"data: [START]\n\n"
                
                # 使用AI核心进行流式分析
                if ai_core is None:
                    # 生成结构化的基础响应
                    basic_response = {
                        "basic_reply": "AI服务正在初始化中，请稍后重试获取详细分析。",
                        "appraisal_report": {
                            "item_name": "待分析物品",
                            "category": "未知",
                            "dynasty": "待确定",
                            "material": "待分析",
                            "authenticity": {
                                "score": 0,
                                "confidence": "低",
                                "analysis": "需要进一步分析"
                            },
                            "value_estimation": {
                                "market_value": "待评估",
                                "collection_value": "待评估",
                                "factors": ["服务初始化中"]
                            },
                            "condition": {
                                "overall": "未检测",
                                "details": "需要图像分析"
                            },
                            "historical_context": "服务初始化中，请稍后重试",
                            "recommendations": ["等待AI服务完全加载后重新分析"]
                        }
                    }
                    
                    import json
                    yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(basic_response, ensure_ascii=False)}}}\n\n"
                else:
                    # 使用AI核心的流式文本分析
                    try:
                        # 实时流式处理 - 立即发送每个数据块
                        analysis_content = ""
                        import json
                        for text_chunk in ai_core.analyze_antique_text_stream(question):
                            if text_chunk and text_chunk.strip():
                                analysis_content += text_chunk
                                # 立即发送内容块
                                import json
                                content_data = {"type": "content", "text": text_chunk}
                                yield f"data: {json.dumps(content_data, ensure_ascii=False)}\n\n"
                                import sys
                                sys.stdout.flush()  # 强制刷新缓冲区
                        
                        # 生成结构化报告
                        structured_report = generate_structured_report(analysis_content, question, 'text')
                        import json
                        yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(structured_report, ensure_ascii=False)}}}\n\n"
                        
                    except Exception as e:
                        error_report = {
                            "basic_reply": f"分析过程中出现错误: {str(e)}",
                            "appraisal_report": {
                                "item_name": "分析失败",
                                "category": "错误",
                                "error": str(e)
                            }
                        }
                        import json
                        yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(error_report, ensure_ascii=False)}}}\n\n"
                
                # 发送结束信号
                yield f"data: {{\"type\": \"end\", \"success\": true}}\n\n"
                
            else:  # input_type == 'image'
                # 发送开始信号
                yield f"data: [START]\n\n"
                
                # 使用AI核心进行流式图像分析
                if ai_core is None:
                    # 生成结构化的基础响应
                    basic_response = {
                        "basic_reply": f"图片已上传成功：{filename}\n\n{question if question else '正在分析图片内容...'}\n\nAI服务正在初始化中，这是一个基础响应。",
                        "appraisal_report": {
                            "item_name": "图片中的物品",
                            "category": "待识别",
                            "dynasty": "待确定",
                            "material": "待分析",
                            "authenticity": {
                                "score": 0,
                                "confidence": "低",
                                "analysis": "需要图像识别分析"
                            },
                            "value_estimation": {
                                "market_value": "待评估",
                                "collection_value": "待评估",
                                "factors": ["图像分析中"]
                            },
                            "condition": {
                                "overall": "从图片观察",
                                "details": "需要AI视觉分析"
                            },
                            "historical_context": "正在进行图像识别和历史匹配",
                            "recommendations": ["等待AI图像分析完成"]
                        }
                    }
                    
                    import json
                    yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(basic_response, ensure_ascii=False)}}}\n\n"
                else:
                    # 使用AI核心的流式图像分析
                    try:
                        # 实时流式处理 - 立即发送每个数据块
                        analysis_content = ""
                        import json
                        for text_chunk in ai_core.analyze_antique_image_stream(image_path, question):
                            if text_chunk and text_chunk.strip():
                                analysis_content += text_chunk
                                # 立即发送内容块
                                content_data = {"type": "content", "text": text_chunk}
                                yield f"data: {json.dumps(content_data, ensure_ascii=False)}\n\n"
                                import sys
                                sys.stdout.flush()  # 强制刷新缓冲区
                        
                        # 生成结构化报告
                        structured_report = generate_structured_report(analysis_content, question, 'image', image_path)
                        import json
                        yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(structured_report, ensure_ascii=False)}}}\n\n"
                        
                    except Exception as e:
                        error_report = {
                            "basic_reply": f"图像分析过程中出现错误: {str(e)}",
                            "appraisal_report": {
                                "item_name": "分析失败",
                                "category": "错误",
                                "error": str(e)
                            }
                        }
                        import json
                        yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(error_report, ensure_ascii=False)}}}\n\n"
                
                # 发送结束信号
                yield f"data: {{\"type\": \"end\", \"success\": true}}\n\n"
                
        except Exception as e:
            logger.error(f"流式古董鉴赏处理失败: {e}")
            yield f"data: {{\"type\": \"error\", \"message\": \"处理失败: {str(e)}\"}}\n\n"
    
    return Response(generate_stream(input_type, question, image_path), 
                   mimetype='text/event-stream; charset=utf-8', 
                   headers={
                       'Cache-Control': 'no-cache',
                       'Connection': 'keep-alive',
                       'X-Accel-Buffering': 'no',  # 禁用nginx缓冲
                       'Transfer-Encoding': 'chunked',
                       'Access-Control-Allow-Origin': '*',
                       'Access-Control-Allow-Headers': 'Cache-Control'
                   })

def generate_structured_report(analysis_content, question, input_type, image_path=None):
    """生成结构化的鉴宝报告"""
    try:
        # 如果有新的格式化器，使用它
        if REPORT_FORMATTER_AVAILABLE:
            return format_appraisal_report(
                ai_analysis=analysis_content or f"您的{input_type}已收到并正在分析中。",
                user_query=question or "",
                image_path=image_path,
                model_info={"backend": "app_structured"}
            )
        
        # 回退到原始格式
        # 基础报告结构 - 修正为前端期望的格式
        structured_report = {
            "basic_reply": analysis_content if analysis_content else f"您的{input_type}已收到并正在分析中。",
            "appraisal_report": {
                "item_name": "待分析物品",
                "category": "未知",
                "dynasty": "待确定",
                "material": "待分析",
                "authenticity": {
                    "score": 0,
                    "confidence": "待评估",
                    "analysis": "需要进一步分析"
                },
                "value_estimation": {
                    "market_value": "待评估",
                    "collection_value": "待评估",
                    "factors": ["正在分析中"]
                },
                "condition": {
                    "overall": "未检测",
                    "details": "需要详细检查"
                },
                "historical_context": "正在分析历史背景和文化价值",
                "recommendations": [
                    "等待AI分析完成",
                    "建议专业机构进一步鉴定"
                ]
            }
        }
        
        # 如果有分析内容，尝试提取结构化信息
        if analysis_content and len(analysis_content) > 50:
            # 简单的关键词提取和分类
            content_lower = analysis_content.lower()
            
            # 朝代识别
            dynasties = ['唐', '宋', '元', '明', '清', '汉', '魏', '晋', '隋']
            for dynasty in dynasties:
                if dynasty in analysis_content:
                    structured_report["appraisal_report"]["dynasty"] = f"{dynasty}代"
                    break
            
            # 材质识别
            materials = ['瓷', '陶', '玉', '铜', '铁', '金', '银', '木', '竹', '石']
            for material in materials:
                if material in analysis_content:
                    structured_report["appraisal_report"]["material"] = material
                    break
            
            # 类别识别
            categories = ['瓷器', '玉器', '青铜器', '书画', '家具', '杂项']
            for category in categories:
                if category in analysis_content:
                    structured_report["appraisal_report"]["category"] = category
                    break
        
        return structured_report
        
    except Exception as e:
        logger.error(f"生成结构化报告失败: {e}")
        return {
            "basic_reply": analysis_content if analysis_content else "分析过程中出现错误",
            "appraisal_report": {
                "error": str(e)
            }
        }

    def generate_stream(input_type, question, image_path=None):
        try:
            if input_type == 'text':
                # 发送开始信号
                yield f"data: [START]\n\n"
                
                # 使用AI核心进行流式分析
                if ai_core is None:
                    # 生成结构化的基础响应
                    basic_response = {
                        "basic_reply": "AI服务正在初始化中，请稍后重试获取详细分析。",
                        "appraisal_report": {
                            "item_name": "待分析物品",
                            "category": "未知",
                            "dynasty": "待确定",
                            "material": "待分析",
                            "authenticity": {
                                "score": 0,
                                "confidence": "低",
                                "analysis": "需要进一步分析"
                            },
                            "value_estimation": {
                                "market_value": "待评估",
                                "collection_value": "待评估",
                                "factors": ["服务初始化中"]
                            },
                            "condition": {
                                "overall": "未检测",
                                "details": "需要图像分析"
                            },
                            "historical_context": "服务初始化中，请稍后重试",
                            "recommendations": ["等待AI服务完全加载后重新分析"]
                        }
                    }
                    
                    import json
                    yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(basic_response, ensure_ascii=False)}}}\n\n"
                else:
                    # 使用AI核心的流式文本分析
                    try:
                        # 收集所有分析内容
                        analysis_content = ""
                        for text_chunk in ai_core.analyze_antique_text_stream(question):
                            if text_chunk and text_chunk.strip():
                                analysis_content += text_chunk
                                # 立即发送内容块
                                content_data = {"type": "content", "text": text_chunk}
                                yield f"data: {json.dumps(content_data, ensure_ascii=False)}\n\n"
                                import sys
                                sys.stdout.flush()
                        
                        # 生成结构化报告
                        structured_report = generate_structured_report(analysis_content, question, 'text')
                        import json
                        yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(structured_report, ensure_ascii=False)}}}\n\n"
                        
                    except Exception as e:
                        error_report = {
                            "basic_reply": f"分析过程中出现错误: {str(e)}",
                            "appraisal_report": {
                                "item_name": "分析失败",
                                "category": "错误",
                                "error": str(e)
                            }
                        }
                        import json
                        yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(error_report, ensure_ascii=False)}}}\n\n"
                
                # 发送结束信号
                yield f"data: {{\"type\": \"end\", \"success\": true}}\n\n"
                
            else:  # input_type == 'image'
                # 发送开始信号
                yield f"data: [START]\n\n"
                
                # 使用AI核心进行流式图像分析
                if ai_core is None:
                    # 生成结构化的基础响应
                    basic_response = {
                        "basic_reply": f"图片已上传成功：{filename}\n\n{question if question else '正在分析图片内容...'}\n\nAI服务正在初始化中，这是一个基础响应。",
                        "appraisal_report": {
                            "item_name": "图片中的物品",
                            "category": "待识别",
                            "dynasty": "待确定",
                            "material": "待分析",
                            "authenticity": {
                                "score": 0,
                                "confidence": "低",
                                "analysis": "需要图像识别分析"
                            },
                            "value_estimation": {
                                "market_value": "待评估",
                                "collection_value": "待评估",
                                "factors": ["图像分析中"]
                            },
                            "condition": {
                                "overall": "从图片观察",
                                "details": "需要AI视觉分析"
                            },
                            "historical_context": "正在进行图像识别和历史匹配",
                            "recommendations": ["等待AI图像分析完成"]
                        }
                    }
                    
                    import json
                    yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(basic_response, ensure_ascii=False)}}}\n\n"
                else:
                    # 使用AI核心的流式图像分析
                    try:
                        # 收集所有分析内容
                        analysis_content = ""
                        for text_chunk in ai_core.analyze_antique_image_stream(image_path, question):
                            if text_chunk and text_chunk.strip():
                                analysis_content += text_chunk
                                # 立即发送内容块
                                content_data = {"type": "content", "text": text_chunk}
                                yield f"data: {json.dumps(content_data, ensure_ascii=False)}\n\n"
                                import sys
                                sys.stdout.flush()  # 强制刷新缓冲区
                        
                        # 生成结构化报告
                        structured_report = generate_structured_report(analysis_content, question, 'image', image_path)
                        import json
                        yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(structured_report, ensure_ascii=False)}}}\n\n"
                        
                    except Exception as e:
                        error_report = {
                            "basic_reply": f"图像分析过程中出现错误: {str(e)}",
                            "appraisal_report": {
                                "item_name": "分析失败",
                                "category": "错误",
                                "error": str(e)
                            }
                        }
                        import json
                        yield f"data: {{\"type\": \"structured_report\", \"data\": {json.dumps(error_report, ensure_ascii=False)}}}\n\n"
                
                # 发送结束信号
                yield f"data: {{\"type\": \"end\", \"success\": true}}\n\n"
                
        except Exception as e:
            logger.error(f"流式古董鉴赏处理失败: {e}")
            yield f"data: {{\"type\": \"error\", \"message\": \"处理失败: {str(e)}\"}}\n\n"
    
    return Response(generate_stream(input_type, question, image_path), mimetype='text/event-stream; charset=utf-8', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',  # 禁用nginx缓冲
        'Transfer-Encoding': 'chunked',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_antique():
    """分析古董（支持图像和文本输入）- 保留向后兼容性"""
    logger.info("收到古董分析请求（兼容接口）")
    
    try:
        # 检查请求类型：JSON（文本）或 multipart/form-data（图片）
        content_type = request.content_type
        
        if content_type and 'application/json' in content_type:
            # 处理文本描述分析
            data = request.get_json()
            if not data or data.get('type') != 'text' or not data.get('description'):
                logger.warning("文本分析请求格式错误")
                return jsonify({'error': '请提供有效的文本描述'}), 400
            
            description = data['description']
            logger.info(f"收到文本分析请求: {description[:100]}..." if len(description) > 100 else f"收到文本分析请求: {description}")
            
            # 检查AI核心是否就绪
            if ai_core is None:
                logger.error("AI服务未初始化")
                return jsonify({'error': 'AI服务未就绪', 'details': '请检查服务器日志'}), 503
            
            # 分析文本描述
            logger.info("开始分析文本描述")
            start_time = datetime.now()
            
            try:
                analysis_result = ai_core.analyze_antique_text(description)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"文本分析完成，耗时: {duration:.2f}秒")
                
            except Exception as analysis_error:
                logger.error(f"文本分析过程失败: {analysis_error}")
                import traceback
                logger.error(f"分析错误详情: {traceback.format_exc()}")
                return jsonify({
                    'error': '分析过程失败', 
                    'details': str(analysis_error)
                }), 500
            
            # 保存分析结果
            try:
                result_filename = save_analysis_result(f"text_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}", analysis_result)
                logger.info(f"分析结果已保存: {result_filename}")
            except Exception as save_error:
                logger.warning(f"保存分析结果失败: {save_error}")
                result_filename = None
            
            # 返回结果
            response_data = {
                'success': True,
                'input_type': 'text',
                'analysis': analysis_result,
                'result_file': result_filename,
                'analysis_duration': duration if 'duration' in locals() else None
            }
            
            logger.info("文本分析成功完成")
            return jsonify(response_data)
            
        else:
            # 处理图片上传分析（原有逻辑）
            if 'image' not in request.files:
                logger.warning("请求中缺少图片文件")
                return jsonify({'error': '没有上传图片'}), 400
            
            file = request.files['image']
            if file.filename == '':
                logger.warning("文件名为空")
                return jsonify({'error': '没有选择文件'}), 400
            
            if not allowed_file(file.filename):
                logger.warning(f"不支持的文件类型: {file.filename}")
                return jsonify({'error': '不支持的文件类型'}), 400
            
            # 保存上传的图片
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            logger.info(f"保存图片到: {filepath}")
            file.save(filepath)
            
            # 获取描述信息
            description = request.form.get('description', '')
            logger.info(f"用户描述: {description[:100]}..." if len(description) > 100 else f"用户描述: {description}")
            
            # 检查AI核心是否就绪
            if ai_core is None:
                logger.error("AI服务未初始化")
                return jsonify({'error': 'AI服务未就绪', 'details': '请检查服务器日志'}), 503
            
            # 分析图像
            logger.info(f"开始分析图像: {filename}")
            start_time = datetime.now()
            
            try:
                analysis_result = ai_core.analyze_antique_image(
                    image=filepath,
                    description=description
                )
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"图像分析完成，耗时: {duration:.2f}秒")
                
            except Exception as analysis_error:
                logger.error(f"图像分析过程失败: {analysis_error}")
                import traceback
                logger.error(f"分析错误详情: {traceback.format_exc()}")
                return jsonify({
                    'error': '分析过程失败', 
                    'details': str(analysis_error)
                }), 500
            
            # 保存分析结果
            try:
                result_filename = save_analysis_result(filepath, analysis_result)
                logger.info(f"分析结果已保存: {result_filename}")
            except Exception as save_error:
                logger.warning(f"保存分析结果失败: {save_error}")
                result_filename = None
            
            # 返回结果
            response_data = {
                'success': True,
                'input_type': 'image',
                'image_filename': filename,
                'analysis': analysis_result,
                'result_file': result_filename,
                'analysis_duration': duration if 'duration' in locals() else None
            }
            
            logger.info(f"图像分析成功完成: {filename}")
            return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"分析接口失败: {e}")
        import traceback
        logger.error(f"接口错误详情: {traceback.format_exc()}")
        return jsonify({
            'error': f'分析失败: {str(e)}',
            'type': type(e).__name__
        }), 500

@app.route('/api/search', methods=['POST'])
def search_antiques():
    """通过文本搜索古董"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': '缺少查询参数'}), 400
        
        query_text = data['query']
        top_k = data.get('top_k', 10)
        
        # 检查AI核心是否就绪
        if ai_core is None:
            return jsonify({'error': 'AI服务未就绪'}), 503
        
        # 搜索古董
        logger.info(f"开始文本搜索: {query_text}")
        search_result = ai_core.search_antiques_by_text(
            query_text=query_text,
            top_k=top_k
        )
        
        response_data = {
            'success': True,
            'query': query_text,
            'results': search_result
        }
        
        logger.info(f"文本搜索完成: {query_text}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"文本搜索失败: {e}")
        return jsonify({'error': f'搜索失败: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """与AI对话"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': '缺少消息参数'}), 400
        
        message = data['message']
        
        # 检查AI核心是否就绪
        if ai_core is None:
            return jsonify({'error': 'AI服务未就绪'}), 503
        
        # 与AI对话
        logger.info(f"开始AI对话: {message[:50]}...")
        response = ai_core.chat_with_agent(message)
        
        response_data = {
            'success': True,
            'message': message,
            'response': response
        }
        
        logger.info("AI对话完成")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"AI对话失败: {e}")
        return jsonify({'error': f'对话失败: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_antique():
    """上传古董到向量数据库"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型'}), 400
        
        # 保存上传的图片
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 获取古董信息
        antique_id = int(request.form.get('antique_id', timestamp))
        description = request.form.get('description', '')
        metadata = {
            'name': request.form.get('name', ''),
            'category': request.form.get('category', ''),
            'dynasty': request.form.get('dynasty', ''),
            'material': request.form.get('material', ''),
            'upload_time': timestamp
        }
        
        # 检查AI核心是否就绪
        if ai_core is None:
            return jsonify({'error': 'AI服务未就绪'}), 503
        
        # 添加到向量数据库
        logger.info(f"开始上传古董: {filename}")
        success = ai_core.add_antique_to_database(
            antique_id=antique_id,
            image=filepath,
            text_description=description,
            metadata=metadata
        )
        
        if success:
            response_data = {
                'success': True,
                'antique_id': antique_id,
                'image_filename': filename,
                'message': '古董上传成功'
            }
            logger.info(f"古董上传成功: {antique_id}")
        else:
            response_data = {
                'success': False,
                'error': '古董上传失败'
            }
            logger.error(f"古董上传失败: {antique_id}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"古董上传失败: {e}")
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/api/images/<filename>')
def get_image(filename):
    """获取上传的图片"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/results/<filename>')
def get_analysis_result(filename):
    """获取分析结果"""
    try:
        result_path = os.path.join(app.config['DATA_FOLDER'], filename)
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            return jsonify(result_data)
        else:
            return jsonify({'error': '结果文件不存在'}), 404
    except Exception as e:
        logger.error(f"获取分析结果失败: {e}")
        return jsonify({'error': f'获取结果失败: {str(e)}'}), 500

@app.route('/api/model/switch', methods=['POST'])
def switch_multimodal_model():
    """切换多模态模型"""
    try:
        data = request.get_json()
        if not data or 'model_type' not in data:
            return jsonify({'error': '缺少模型类型参数'}), 400
        
        model_type = data['model_type'].lower()
        
        # 检查AI核心是否就绪
        if ai_core is None:
            return jsonify({'error': 'AI服务未就绪'}), 503
        
        # 切换模型
        logger.info(f"开始切换到{model_type}模型")
        success = ai_core.switch_multimodal_model(model_type)
        
        if success:
            current_model_info = ai_core.get_current_model_info()
            response_data = {
                'success': True,
                'message': f'成功切换到{model_type}模型',
                'current_model': current_model_info
            }
            logger.info(f"模型切换成功: {model_type}")
            return jsonify(response_data)
        else:
            return jsonify({
                'error': f'切换到{model_type}模型失败',
                'details': '请检查模型配置和日志'
            }), 500
        
    except Exception as e:
        logger.error(f"模型切换失败: {e}")
        return jsonify({'error': f'切换失败: {str(e)}'}), 500

@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    """获取当前模型状态"""
    try:
        # 检查AI核心是否就绪
        if ai_core is None:
            return jsonify({'error': 'AI服务未就绪'}), 503
        
        # 获取模型信息
        current_model_info = ai_core.get_current_model_info()
        system_status = ai_core.get_system_status()
        
        response_data = {
            'success': True,
            'current_model': current_model_info,
            'available_models': ['omnivision'],
            'system_status': system_status
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"获取模型状态失败: {e}")
        return jsonify({'error': f'获取状态失败: {str(e)}'}), 500

@app.route('/api/chat/multimodal', methods=['POST'])
def chat_with_multimodal():
    """使用当前多模态模型对话"""
    try:
        # 检查文件上传
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型'}), 400
        
        # 保存上传的图片
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 获取用户消息
        message = request.form.get('message', '')
        if not message:
            return jsonify({'error': '缺少消息内容'}), 400
        
        # 检查AI核心是否就绪
        if ai_core is None:
            return jsonify({'error': 'AI服务未就绪'}), 503
        
        # 与多模态模型对话
        logger.info(f"开始多模态对话: {message[:50]}...")
        response = ai_core.chat_with_multimodal_model(filepath, message)
        
        response_data = {
            'success': True,
            'message': message,
            'response': response,
            'model_type': ai_core.multimodal_model_type,
            'image_filename': filename
        }
        
        logger.info("多模态对话完成")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"多模态对话失败: {e}")
        return jsonify({'error': f'对话失败: {str(e)}'}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取系统状态"""
    try:
        status = {
            'ai_core_ready': ai_core is not None,
            'timestamp': datetime.now().isoformat(),
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'data_folder': app.config['DATA_FOLDER']
        }
        
        if ai_core:
            try:
                system_status = ai_core.get_system_status()
                status['ai_system'] = system_status
            except Exception as e:
                status['ai_system_error'] = str(e)
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        return jsonify({'error': f'获取状态失败: {str(e)}'}), 500

@app.route('/api/clear', methods=['POST'])
def clear_data():
    """清理数据（可选功能）"""
    try:
        # 清理分析结果
        data_folder = app.config['DATA_FOLDER']
        for filename in os.listdir(data_folder):
            if filename.endswith('.json'):
                os.remove(os.path.join(data_folder, filename))
        
        # 清理上传的图片
        upload_folder = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(upload_folder):
            if filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                os.remove(os.path.join(upload_folder, filename))
        
        logger.info("数据清理完成")
        return jsonify({'success': True, 'message': '数据清理完成'})
        
    except Exception as e:
        logger.error(f"数据清理失败: {e}")
        return jsonify({'error': f'清理失败: {str(e)}'}), 500

@app.route('/api/logs', methods=['POST'])
def receive_frontend_logs():
    """接收前端日志"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '缺少日志数据'}), 400
        
        # 记录前端日志
        from utils.logger_config import get_global_logger_config
        frontend_logger = get_global_logger_config().get_frontend_logger()
        log_level = data.get('level', 'INFO').upper()
        log_message = data.get('message', '')
        
        # 构造日志信息
        log_info = {
            'frontend_timestamp': data.get('timestamp'),
            'url': data.get('url'),
            'userAgent': data.get('userAgent'),
            'extra': data.get('extra', {})
        }
        
        # 根据级别记录日志
        if log_level == 'DEBUG':
            frontend_logger.debug(f'前端: {log_message}', extra={'frontend_data': log_info})
        elif log_level == 'INFO':
            frontend_logger.info(f'前端: {log_message}', extra={'frontend_data': log_info})
        elif log_level == 'WARNING':
            frontend_logger.warning(f'前端: {log_message}', extra={'frontend_data': log_info})
        elif log_level == 'ERROR':
            frontend_logger.error(f'前端: {log_message}', extra={'frontend_data': log_info})
        else:
            frontend_logger.info(f'前端: {log_message}', extra={'frontend_data': log_info})
        
        return jsonify({'success': True}), 200
        
    except Exception as e:
        logger.error(f'接收前端日志失败: {e}')
        return jsonify({'error': '日志接收失败'}), 500

if __name__ == '__main__':
    # 初始化AI核心
    if init_ai_core():
        logger.info("启动AI鉴宝师后端服务")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("AI核心初始化失败，服务无法启动")
        exit(1)
