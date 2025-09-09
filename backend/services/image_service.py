import os
import sys
import uuid
from PIL import Image
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# 添加项目根目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.logger_config import get_backend_logger

# 配置图像服务模块日志
logger = get_backend_logger('image_service')

class ImageService:
    """图片处理服务"""
    
    def __init__(self):
        """初始化图片服务"""
        self.upload_folder = 'uploads'
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        
        # 创建上传目录
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)
    
    def allowed_file(self, filename):
        """检查文件扩展名是否允许"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def save_image(self, file):
        """保存上传的图片"""
        if file and self.allowed_file(file.filename):
            # 生成唯一文件名
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
            
            # 保存文件
            file_path = os.path.join(self.upload_folder, unique_filename)
            file.save(file_path)
            
            return file_path
        else:
            raise ValueError("不支持的文件格式")
    
    def process_image(self, image_path):
        """处理图片（调整大小、增强等）"""
        try:
            # 打开图片
            image = Image.open(image_path)
            
            # 调整大小（保持宽高比）
            max_size = (800, 800)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 保存处理后的图片
            processed_path = image_path.replace('.', '_processed.')
            image.save(processed_path, quality=85, optimize=True)
            
            return processed_path
            
        except Exception as e:
            raise Exception(f"图片处理失败: {str(e)}")
    
    def enhance_image(self, image_path):
        """增强图片质量"""
        try:
            # 使用OpenCV读取图片
            image = cv2.imread(image_path)
            
            # 转换为LAB颜色空间
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # 分离通道
            l, a, b = cv2.split(lab)
            
            # 对L通道进行CLAHE（对比度限制自适应直方图均衡）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # 合并通道
            enhanced_lab = cv2.merge([l, a, b])
            
            # 转换回BGR
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 保存增强后的图片
            enhanced_path = image_path.replace('.', '_enhanced.')
            cv2.imwrite(enhanced_path, enhanced_image)
            
            return enhanced_path
            
        except Exception as e:
            raise Exception(f"图片增强失败: {str(e)}")
    
    def extract_features(self, image_path):
        """提取图片特征"""
        try:
            # 使用OpenCV读取图片
            image = cv2.imread(image_path)
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 提取SIFT特征
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            # 提取颜色直方图
            color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            color_hist = cv2.normalize(color_hist, color_hist).flatten()
            
            # 提取纹理特征（GLCM）
            # 这里简化处理，实际可以使用更复杂的纹理特征提取方法
            
            features = {
                'sift_keypoints': len(keypoints),
                'sift_descriptors': descriptors.shape if descriptors is not None else (0, 0),
                'color_histogram': color_hist.tolist(),
                'image_size': image.shape,
                'aspect_ratio': image.shape[1] / image.shape[0]
            }
            
            return features
            
        except Exception as e:
            raise Exception(f"特征提取失败: {str(e)}")
    
    def validate_image(self, image_path):
        """验证图片质量"""
        try:
            # 检查文件大小
            file_size = os.path.getsize(image_path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                return False, "图片文件过大"
            
            # 检查图片尺寸
            image = Image.open(image_path)
            width, height = image.size
            
            if width < 100 or height < 100:
                return False, "图片尺寸过小"
            
            if width > 5000 or height > 5000:
                return False, "图片尺寸过大"
            
            # 检查图片是否损坏
            try:
                image.verify()
            except:
                return False, "图片文件损坏"
            
            return True, "图片验证通过"
            
        except Exception as e:
            return False, f"图片验证失败: {str(e)}"
    
    def delete_image(self, image_path):
        """删除图片文件"""
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                return True
            return False
        except Exception as e:
            print(f"删除图片失败: {e}")
            return False
