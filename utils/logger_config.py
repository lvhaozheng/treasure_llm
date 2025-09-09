"""
统一日志配置工具
为整个AI鉴宝师项目提供标准化的日志记录配置
"""

import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class LoggerConfig:
    """统一日志配置类"""
    
    # 日志级别映射
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self, base_log_dir: str = "logs"):
        """
        初始化日志配置
        
        Args:
            base_log_dir: 基础日志目录
        """
        self.base_log_dir = Path(base_log_dir)
        self.ensure_log_directory()
    
    def ensure_log_directory(self):
        """确保日志目录存在"""
        self.base_log_dir.mkdir(exist_ok=True)
        # 精简后不再需要子目录，所有日志文件直接放在logs根目录
    
    def get_logger(self, 
                   name: str,
                   log_file: str,
                   level: str = 'INFO',
                   max_bytes: int = 10 * 1024 * 1024,  # 10MB
                   backup_count: int = 5,
                   console_output: bool = True,
                   json_format: bool = False) -> logging.Logger:
        """
        获取配置好的logger
        
        Args:
            name: logger名称
            log_file: 日志文件路径（相对于base_log_dir）
            level: 日志级别
            max_bytes: 单个日志文件最大大小
            backup_count: 备份文件数量
            console_output: 是否输出到控制台
            json_format: 是否使用JSON格式
        
        Returns:
            配置好的logger
        """
        # 获取或创建logger
        logger = logging.getLogger(name)
        
        # 如果logger已经配置过，直接返回
        if logger.handlers:
            return logger
        
        logger.setLevel(self.LOG_LEVELS.get(level.upper(), logging.INFO))
        
        # 创建文件处理器（轮转日志）
        log_path = self.base_log_dir / log_file
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # 创建格式化器
        if json_format:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器（如果需要）
        if console_output:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def get_ai_core_logger(self, module_name: str) -> logging.Logger:
        """获取AI核心模块logger - 所有AI相关日志合并到一个文件"""
        return self.get_logger(
            name=f"ai_core.{module_name}",
            log_file="ai_core.log",
            level='DEBUG',
            console_output=True
        )
    
    def get_backend_logger(self, module_name: str) -> logging.Logger:
        """获取后端模块logger - 所有后端相关日志合并到一个文件"""
        return self.get_logger(
            name=f"backend.{module_name}",
            log_file="backend.log",
            level='INFO',
            console_output=True
        )
    
    def get_frontend_logger(self) -> logging.Logger:
        """获取前端logger - 前端日志独立存储"""
        return self.get_logger(
            name="frontend.app",
            log_file="frontend.log",
            level='INFO',
            console_output=False
        )
    
    def get_system_logger(self) -> logging.Logger:
        """获取系统级logger"""
        return self.get_logger(
            name="system.main",
            log_file="system.log",
            level='INFO',
            console_output=True
        )
    
    def cleanup_old_logs(self, days: int = 30):
        """清理超过指定天数的日志文件"""
        try:
            cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
            
            for log_file in self.base_log_dir.rglob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    print(f"已删除旧日志文件: {log_file}")
        
        except Exception as e:
            print(f"清理日志文件失败: {e}")


class JsonFormatter(logging.Formatter):
    """JSON格式的日志格式化器"""
    
    def format(self, record):
        import json
        
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceLogger:
    """性能监控日志类"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_execution_time(self, func_name: str, execution_time: float, *args, **kwargs):
        """记录函数执行时间"""
        self.logger.info(
            f"性能监控 - {func_name} 执行时间: {execution_time:.4f}s",
            extra={'extra_fields': {
                'function': func_name,
                'execution_time': execution_time,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            }}
        )
    
    def log_memory_usage(self, func_name: str, memory_usage: float):
        """记录内存使用情况"""
        self.logger.info(
            f"内存监控 - {func_name} 内存使用: {memory_usage:.2f}MB",
            extra={'extra_fields': {
                'function': func_name,
                'memory_mb': memory_usage
            }}
        )


def setup_project_logging(base_dir: str = None) -> LoggerConfig:
    """
    为整个项目设置日志系统
    
    Args:
        base_dir: 项目根目录，如果为None则自动检测
    
    Returns:
        LoggerConfig实例
    """
    if base_dir is None:
        # 自动检测项目根目录
        base_dir = Path(__file__).parent.parent
    
    log_dir = Path(base_dir) / "logs"
    return LoggerConfig(str(log_dir))


# 单例模式的全局logger配置
_global_logger_config = None

def get_global_logger_config() -> LoggerConfig:
    """获取全局logger配置实例"""
    global _global_logger_config
    if _global_logger_config is None:
        _global_logger_config = setup_project_logging()
    return _global_logger_config


# 便捷函数
def get_ai_logger(module_name: str) -> logging.Logger:
    """便捷函数：获取AI核心模块logger"""
    return get_global_logger_config().get_ai_core_logger(module_name)

def get_backend_logger(module_name: str) -> logging.Logger:
    """便捷函数：获取后端模块logger"""
    return get_global_logger_config().get_backend_logger(module_name)

def get_system_logger() -> logging.Logger:
    """便捷函数：获取系统logger"""
    return get_global_logger_config().get_system_logger()