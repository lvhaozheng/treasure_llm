#!/usr/bin/env python3
"""
日志管理工具
用于管理、查看和分析项目日志
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import gzip
import shutil
import glob
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.logger_config import LoggerConfig, get_global_logger_config


class LogManager:
    """日志管理器"""
    
    def __init__(self):
        self.logger_config = get_global_logger_config()
        self.base_log_dir = self.logger_config.base_log_dir
        
    def list_log_files(self, module: str = None) -> List[Path]:
        """列出日志文件"""
        if module:
            pattern = f"{module}/*.log*"
        else:
            pattern = "**/*.log*"
        
        log_files = list(self.base_log_dir.glob(pattern))
        return sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def show_recent_logs(self, module: str = None, lines: int = 50):
        """显示最近的日志"""
        log_files = self.list_log_files(module)
        
        if not log_files:
            print("未找到日志文件")
            return
        
        print(f"显示最新 {lines} 行日志:")
        print("=" * 80)
        
        for log_file in log_files[:3]:  # 显示最新的3个文件
            print(f"\n文件: {log_file}")
            print("-" * 40)
            
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_lines = f.readlines()
                    recent_lines = log_lines[-lines:] if len(log_lines) > lines else log_lines
                    
                    for line in recent_lines:
                        print(line.rstrip())
                        
            except Exception as e:
                print(f"读取日志文件失败: {e}")
    
    def search_logs(self, keyword: str, module: str = None, days: int = 7):
        """搜索日志内容"""
        log_files = self.list_log_files(module)
        cutoff_time = datetime.now() - timedelta(days=days)
        
        matches = []
        
        for log_file in log_files:
            # 检查文件修改时间
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_time:
                continue
                
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line_no, line in enumerate(f, 1):
                        if keyword.lower() in line.lower():
                            matches.append({
                                'file': str(log_file),
                                'line': line_no,
                                'content': line.strip()
                            })
            except Exception as e:
                print(f"搜索文件 {log_file} 失败: {e}")
        
        if matches:
            print(f"找到 {len(matches)} 条匹配记录:")
            print("=" * 80)
            
            for match in matches[:100]:  # 限制显示前100条
                print(f"{match['file']}:{match['line']} - {match['content']}")
        else:
            print("未找到匹配的日志记录")
    
    def analyze_errors(self, days: int = 7):
        """分析错误日志"""
        log_files = self.list_log_files()
        cutoff_time = datetime.now() - timedelta(days=days)
        
        error_stats = {}
        warning_stats = {}
        
        for log_file in log_files:
            # 检查文件修改时间
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_time:
                continue
                
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ' - ERROR - ' in line:
                            # 提取错误类型
                            parts = line.split(' - ERROR - ')
                            if len(parts) > 1:
                                error_msg = parts[1].split(':')[0]
                                error_stats[error_msg] = error_stats.get(error_msg, 0) + 1
                        
                        elif ' - WARNING - ' in line:
                            # 提取警告类型
                            parts = line.split(' - WARNING - ')
                            if len(parts) > 1:
                                warning_msg = parts[1].split(':')[0]
                                warning_stats[warning_msg] = warning_stats.get(warning_msg, 0) + 1
                                
            except Exception as e:
                print(f"分析文件 {log_file} 失败: {e}")
        
        # 输出分析结果
        print(f"最近 {days} 天的错误和警告统计:")
        print("=" * 80)
        
        if error_stats:
            print("\n错误统计:")
            for error, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error}: {count} 次")
        
        if warning_stats:
            print("\n警告统计:")
            for warning, count in sorted(warning_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {warning}: {count} 次")
        
        if not error_stats and not warning_stats:
            print("未发现错误或警告记录")
    
    def cleanup_old_logs(self, days: int = 30, compress: bool = True):
        """清理旧日志"""
        cutoff_time = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        compressed_count = 0
        
        print(f"清理超过 {days} 天的日志文件...")
        
        for log_file in self.list_log_files():
            file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            
            if file_time < cutoff_time:
                if compress and not str(log_file).endswith('.gz'):
                    # 压缩文件
                    try:
                        gz_file = Path(str(log_file) + '.gz')
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(gz_file, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        log_file.unlink()  # 删除原文件
                        compressed_count += 1
                        print(f"压缩: {log_file} -> {gz_file}")
                        
                    except Exception as e:
                        print(f"压缩失败 {log_file}: {e}")
                else:
                    # 直接删除
                    try:
                        log_file.unlink()
                        cleaned_count += 1
                        print(f"删除: {log_file}")
                    except Exception as e:
                        print(f"删除失败 {log_file}: {e}")
        
        print(f"清理完成: 删除 {cleaned_count} 个文件, 压缩 {compressed_count} 个文件")
    
    def show_log_stats(self):
        """显示日志统计信息"""
        log_files = self.list_log_files()
        
        stats = {
            'total_files': len(log_files),
            'total_size': 0,
            'modules': {},
            'recent_activity': []
        }
        
        for log_file in log_files:
            # 统计大小
            stats['total_size'] += log_file.stat().st_size
            
            # 统计模块
            module = log_file.parent.name
            if module not in stats['modules']:
                stats['modules'][module] = {'count': 0, 'size': 0}
            
            stats['modules'][module]['count'] += 1
            stats['modules'][module]['size'] += log_file.stat().st_size
            
            # 最近活动
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=24):
                stats['recent_activity'].append({
                    'file': str(log_file),
                    'time': mtime.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # 显示统计信息
        print("日志统计信息:")
        print("=" * 80)
        print(f"总文件数: {stats['total_files']}")
        print(f"总大小: {stats['total_size'] / 1024 / 1024:.2f} MB")
        
        print(f"\n模块分布:")
        for module, info in stats['modules'].items():
            size_mb = info['size'] / 1024 / 1024
            print(f"  {module}: {info['count']} 文件, {size_mb:.2f} MB")
        
        if stats['recent_activity']:
            print(f"\n最近24小时活跃文件:")
            for activity in sorted(stats['recent_activity'], 
                                 key=lambda x: x['time'], reverse=True)[:10]:
                print(f"  {activity['time']} - {activity['file']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='日志管理工具')
    parser.add_argument('command', choices=['list', 'tail', 'search', 'analyze', 'cleanup', 'stats'],
                       help='执行的命令')
    parser.add_argument('--module', '-m', help='指定模块 (ai_core, backend, frontend, system)')
    parser.add_argument('--lines', '-n', type=int, default=50, help='显示行数')
    parser.add_argument('--days', '-d', type=int, default=7, help='天数')
    parser.add_argument('--keyword', '-k', help='搜索关键词')
    parser.add_argument('--compress', action='store_true', help='清理时压缩而不是删除')
    
    args = parser.parse_args()
    
    log_manager = LogManager()
    
    try:
        if args.command == 'list':
            files = log_manager.list_log_files(args.module)
            print(f"日志文件列表 ({len(files)} 个文件):")
            for f in files:
                size = f.stat().st_size / 1024  # KB
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                print(f"  {f} ({size:.1f}KB, {mtime.strftime('%Y-%m-%d %H:%M')})")
        
        elif args.command == 'tail':
            log_manager.show_recent_logs(args.module, args.lines)
        
        elif args.command == 'search':
            if not args.keyword:
                print("搜索需要指定关键词 --keyword")
                return
            log_manager.search_logs(args.keyword, args.module, args.days)
        
        elif args.command == 'analyze':
            log_manager.analyze_errors(args.days)
        
        elif args.command == 'cleanup':
            log_manager.cleanup_old_logs(args.days, args.compress)
        
        elif args.command == 'stats':
            log_manager.show_log_stats()
    
    except KeyboardInterrupt:
        print("\n操作被用户中断")
    except Exception as e:
        print(f"执行失败: {e}")


if __name__ == '__main__':
    main()
