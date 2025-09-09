#!/usr/bin/env python3
"""
AI鉴宝师项目启动脚本 (跨平台Python版)
自动启动所有必要的服务：Docker依赖、后端、前端
支持Windows、Linux、macOS
"""

import os
import sys
import subprocess
import time
import signal
import threading
import argparse
import requests
import socket
import platform
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

# 尝试导入colorama用于Windows颜色支持
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

# 尝试导入dotenv
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

class Colors:
    """跨平台控制台颜色"""
    if HAS_COLORAMA:
        HEADER = Fore.MAGENTA
        OKBLUE = Fore.BLUE
        OKCYAN = Fore.CYAN
        OKGREEN = Fore.GREEN
        WARNING = Fore.YELLOW
        FAIL = Fore.RED
        ENDC = Style.RESET_ALL
        BOLD = Style.BRIGHT
    else:
        # 对于非Windows系统或未安装colorama的情况
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'

class ProjectStarter:
    """AI鉴宝师项目启动器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.backend_dir = self.project_root / "backend"
        self.frontend_dir = self.project_root / "frontend"
        self.docker_compose_file = "docker-compose.yml"
        self.conda_env = "treasure_llm"
        self.processes: List[Tuple[str, subprocess.Popen]] = []
        self.docker_started = False
        self.is_windows = platform.system() == "Windows"
        self._load_env_file()
        self._register_signal_handlers()
    
    def _load_env_file(self):
        """加载环境变量文件"""
        env_file = self.project_root / ".env"
        if env_file.exists() and HAS_DOTENV:
            load_dotenv(env_file)
            self.print_status("✓ 环境变量加载成功", "SUCCESS")
        elif env_file.exists():
            self.print_status("⚠ .env文件存在但python-dotenv未安装", "WARNING")
        else:
            self.print_status("⚠ .env文件不存在，使用默认配置", "WARNING")
    
    def _register_signal_handlers(self):
        """注册信号处理器"""
        def signal_handler(signum, frame):
            self.print_status("收到停止信号，正在清理资源...", "WARNING")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        if not self.is_windows:
            signal.signal(signal.SIGTERM, signal_handler)
    
    def print_status(self, message: str, status: str = "INFO"):
        """打印状态信息"""
        color_map = {
            "INFO": Colors.OKBLUE,
            "SUCCESS": Colors.OKGREEN,
            "WARNING": Colors.WARNING,
            "ERROR": Colors.FAIL,
            "DOCKER": Colors.HEADER,
            "FRONTEND": Colors.OKCYAN,
            "BACKEND": Colors.OKGREEN
        }
        color = color_map.get(status, Colors.OKBLUE)
        print(f"{color}[{status}]{Colors.ENDC} {message}")
    
    def command_exists(self, command: str) -> bool:
        """检查命令是否存在"""
        try:
            subprocess.run([command, "--version"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def check_port(self, port: int) -> bool:
        """检查端口是否被占用"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result == 0
        except Exception:
            return False
    
    def wait_for_service(self, url: str, name: str, max_attempts: int = 30) -> bool:
        """等待服务就绪"""
        self.print_status(f"等待 {name} 服务就绪...", "INFO")
        
        for i in range(max_attempts):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    self.print_status(f"{name} 服务就绪", "SUCCESS")
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)
        
        self.print_status(f"{name} 服务启动超时", "WARNING")
        return False
    
    def wait_for_port(self, host: str, port: int, service_name: str, timeout: int = 60) -> bool:
        """等待端口服务就绪"""
        self.print_status(f"等待 {service_name} 服务就绪...", "INFO")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        # 对于 Milvus，额外检查服务是否真正就绪
                        if service_name == "Milvus" and port == 19530:
                            time.sleep(5)  # 给 Milvus 额外时间初始化
                            try:
                                # 尝试简单的连接测试
                                test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                test_sock.settimeout(3)
                                test_result = test_sock.connect_ex((host, port))
                                test_sock.close()
                                if test_result == 0:
                                    self.print_status(f"{service_name} 服务已就绪", "SUCCESS")
                                    return True
                            except Exception:
                                pass
                        else:
                            self.print_status(f"{service_name} 服务已就绪", "SUCCESS")
                            return True
            except Exception:
                pass
            
            time.sleep(2)
        
        self.print_status(f"等待 {service_name} 服务超时", "ERROR")
        return False
    
    def check_requirements(self) -> bool:
        """检查系统要求"""
        self.print_status("检查系统要求...", "INFO")
        
        # 检查Python版本
        if sys.version_info < (3, 9):
            self.print_status(f"需要Python 3.9+，当前版本: {sys.version}", "ERROR")
            return False
        self.print_status("Python版本检查通过", "SUCCESS")
        
        # 检查Docker
        if not self.check_docker():
            return False
        
        # 检查Docker Compose
        if not self.command_exists("docker-compose"):
            self.print_status("Docker Compose未安装", "ERROR")
            return False
        self.print_status("Docker Compose检查通过", "SUCCESS")
        
        # 检查Node.js
        if not self.command_exists("node"):
            self.print_status("Node.js未安装，请先安装Node.js 16+", "ERROR")
            return False
        self.print_status("Node.js检查通过", "SUCCESS")
        
        return True
    
    def check_docker(self) -> bool:
        """检查Docker是否可用"""
        try:
            # 检查Docker客户端
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.print_status("Docker未安装或不可用", "ERROR")
                return False
            
            # 检查Docker守护进程是否运行
            result = subprocess.run(['docker', 'info'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.print_status("Docker检查通过", "SUCCESS")
                return True
            else:
                self.print_status("Docker Desktop未运行，请启动Docker Desktop后重试", "ERROR")
                self.print_status("提示: 请确保Docker Desktop正在运行", "INFO")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.print_status("Docker未安装或不可用", "ERROR")
            return False
    
    def start_docker_dependencies(self) -> bool:
        """启动Docker依赖服务"""
        self.print_status("启动Docker依赖服务...", "DOCKER")
        
        try:
            os.chdir(self.project_root)
            result = subprocess.run(
                ["docker-compose", "up", "-d"],
                capture_output=True, text=True, check=True
            )
            self.docker_started = True
            self.print_status("Docker依赖服务启动成功", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            self.print_status(f"Docker服务启动失败: {e.stderr}", "ERROR")
            return False
    
    def start_backend(self) -> bool:
        """启动后端服务"""
        self.print_status("启动后端服务...", "BACKEND")
        
        try:
            os.chdir(self.backend_dir)
            
            # 确保日志目录存在
            logs_dir = self.backend_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # 检查是否使用生产模式
            use_production = os.getenv('USE_PRODUCTION_SERVER', 'true').lower() == 'true'
            
            if use_production:
                # Windows环境使用Waitress，Linux/macOS使用Gunicorn
                if self.is_windows:
                    self.print_status("使用Waitress生产服务器（Windows）", "SUCCESS")
                    
                    # 检查虚拟环境
                    if self.command_exists("conda"):
                        # 检查conda环境是否存在
                        result = subprocess.run(
                            ["conda", "info", "--envs"],
                            capture_output=True, text=True
                        )
                        if self.conda_env in result.stdout:
                            self.print_status(f"使用Conda环境: {self.conda_env}", "SUCCESS")
                            cmd = f"conda activate {self.conda_env} && python waitress_server.py"
                            process = subprocess.Popen(cmd, shell=True)
                        else:
                            self.print_status(f"Conda环境 {self.conda_env} 不存在，使用系统Python", "WARNING")
                            process = subprocess.Popen([sys.executable, "waitress_server.py"])
                    else:
                        self.print_status("使用系统Python", "INFO")
                        process = subprocess.Popen([sys.executable, "waitress_server.py"])
                        
                elif self.command_exists("gunicorn"):
                    self.print_status("使用Gunicorn生产服务器（Linux/macOS）", "SUCCESS")
                    
                    # 检查虚拟环境
                    if self.command_exists("conda"):
                        # 检查conda环境是否存在
                        result = subprocess.run(
                            ["conda", "info", "--envs"],
                            capture_output=True, text=True
                        )
                        if self.conda_env in result.stdout:
                            self.print_status(f"使用Conda环境: {self.conda_env}", "SUCCESS")
                            cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env} && gunicorn -c gunicorn.conf.py wsgi:application"
                            process = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
                        else:
                            self.print_status(f"Conda环境 {self.conda_env} 不存在，使用系统Python", "WARNING")
                            process = subprocess.Popen(["gunicorn", "-c", "gunicorn.conf.py", "wsgi:application"])
                    else:
                        self.print_status("使用系统Python", "INFO")
                        process = subprocess.Popen(["gunicorn", "-c", "gunicorn.conf.py", "wsgi:application"])
                else:
                    self.print_status("生产服务器不可用，回退到开发服务器", "WARNING")
                    use_production = False
            else:
                # 回退到Flask开发服务器
                self.print_status("使用Flask开发服务器（不推荐用于生产环境）", "WARNING")
                
                # 检查虚拟环境
                if self.command_exists("conda"):
                    # 检查conda环境是否存在
                    result = subprocess.run(
                        ["conda", "info", "--envs"],
                        capture_output=True, text=True
                    )
                    if self.conda_env in result.stdout:
                        self.print_status(f"使用Conda环境: {self.conda_env}", "SUCCESS")
                        if self.is_windows:
                            cmd = f"conda activate {self.conda_env} && python app.py"
                            process = subprocess.Popen(cmd, shell=True)
                        else:
                            cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env} && python app.py"
                            process = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
                    else:
                        self.print_status(f"Conda环境 {self.conda_env} 不存在，使用系统Python", "WARNING")
                        process = subprocess.Popen([sys.executable, "app.py"])
                else:
                    self.print_status("使用系统Python", "INFO")
                    process = subprocess.Popen([sys.executable, "app.py"])
            
            self.processes.append(("backend", process))
            
            # 等待后端启动
            time.sleep(8)  # Gunicorn需要更多时间启动
            if self.wait_for_service("http://localhost:5000/health", "后端", 30):
                return True
            else:
                self.print_status("后端服务启动超时，但继续启动前端", "WARNING")
                return True
                
        except Exception as e:
            self.print_status(f"后端服务启动失败: {e}", "ERROR")
            return False
    
    def start_frontend(self) -> bool:
        """启动前端服务"""
        self.print_status("启动前端服务...", "FRONTEND")
        
        try:
            os.chdir(self.frontend_dir)
            
            # 检查依赖
            if not (self.frontend_dir / "node_modules").exists():
                self.print_status("安装前端依赖...", "INFO")
                if self.is_windows:
                    result = subprocess.run(
                        ["npm", "install"],
                        capture_output=True, text=True, check=True, shell=True
                    )
                else:
                    result = subprocess.run(
                        ["npm", "install"],
                        capture_output=True, text=True, check=True
                    )
                self.print_status("前端依赖安装完成", "SUCCESS")
            
            # 启动前端
            if self.is_windows:
                # Windows系统使用shell=True
                process = subprocess.Popen(["npm", "start"], shell=True)
            else:
                # Unix系统直接执行
                process = subprocess.Popen(["npm", "start"])
            
            self.processes.append(("frontend", process))
            
            self.print_status("前端服务启动成功", "SUCCESS")
            
            # 等待前端启动
            time.sleep(15)
            self.wait_for_service("http://localhost:3000", "前端", 30)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.print_status(f"前端服务启动失败: {e.stderr}", "ERROR")
            return False
        except Exception as e:
            self.print_status(f"前端服务启动失败: {e}", "ERROR")
            return False
    
    def show_status(self):
        """显示服务状态"""
        print()
        print("=" * 60)
        print(f"{Colors.BOLD}                    服务启动完成{Colors.ENDC}")
        print("=" * 60)
        print()
        self.print_status("前端地址: http://localhost:3000", "SUCCESS")
        self.print_status("后端地址: http://localhost:5000", "SUCCESS")
        self.print_status("后端健康检查: http://localhost:5000/health", "SUCCESS")
        print()
        self.print_status("按 Ctrl+C 停止所有服务...", "INFO")
        print()
    
    def cleanup(self):
        """清理资源"""
        self.print_status("正在清理资源...", "INFO")
        
        # 停止本地服务
        for name, process in self.processes:
            if process.poll() is None:
                self.print_status(f"停止{name}服务...", "INFO")
                try:
                    if self.is_windows:
                        subprocess.run(["taskkill", "/f", "/t", "/pid", str(process.pid)], 
                                     capture_output=True)
                    else:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                except Exception as e:
                    self.print_status(f"停止{name}服务时出错: {e}", "WARNING")
        
        # 停止Docker服务
        if self.docker_started:
            self.print_status("停止Docker依赖服务...", "DOCKER")
            try:
                os.chdir(self.project_root)
                subprocess.run(
                    ["docker-compose", "down"],
                    capture_output=True, check=True
                )
            except subprocess.CalledProcessError as e:
                self.print_status(f"停止Docker服务时出错: {e.stderr}", "WARNING")
        
        self.print_status("清理完成", "SUCCESS")
    
    def start_services(self):
        """启动所有服务"""
        print()
        print("=" * 60)
        print(f"{Colors.BOLD}           AI鉴宝师项目启动器 v3.0{Colors.ENDC}")
        print("=" * 60)
        print()
        
        try:
            # 检查系统要求
            if not self.check_requirements():
                return False
            
            # 启动Docker依赖
            if not self.start_docker_dependencies():
                return False
            
            # 等待Docker服务就绪
            self.print_status("等待Docker服务就绪...", "INFO")
            time.sleep(5)
            
            # 检查关键服务端口
            self.wait_for_port("localhost", 6379, "Redis", 30)   # Redis
            self.wait_for_port("localhost", 19530, "Milvus", 60) # Milvus需要更长时间
            
            # 启动后端
            if not self.start_backend():
                return False
            
            # 启动前端
            if not self.start_frontend():
                return False
            
            # 显示状态
            self.show_status()
            
            # 等待用户中断
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
            return True
            
        except Exception as e:
            self.print_status(f"启动过程中出现错误: {e}", "ERROR")
            return False
        finally:
            self.cleanup()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI鉴宝师项目启动器")
    parser.add_argument("--no-docker", action="store_true", 
                       help="跳过Docker依赖启动")
    parser.add_argument("--backend-only", action="store_true", 
                       help="只启动后端服务")
    parser.add_argument("--frontend-only", action="store_true", 
                       help="只启动前端服务")
    
    args = parser.parse_args()
    
    starter = ProjectStarter()
    
    try:
        if args.backend_only:
            starter.start_backend()
        elif args.frontend_only:
            starter.start_frontend()
        else:
            success = starter.start_services()
            if not success:
                sys.exit(1)
    except KeyboardInterrupt:
        starter.print_status("用户中断启动", "INFO")
    except Exception as e:
        starter.print_status(f"启动失败: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()