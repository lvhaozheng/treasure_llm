#!/bin/bash

# ============================================================
# AI鉴宝师项目启动脚本 (Linux/Mac版)
# 自动启动所有必要的服务：Docker依赖、后端、前端
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 全局变量
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
CONDA_ENV="treasure_llm"
FRONTEND_PID=""
BACKEND_PID=""
DOCKER_STARTED=0

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_docker() {
    echo -e "${PURPLE}[DOCKER]${NC} $1"
}

log_backend() {
    echo -e "${GREEN}[BACKEND]${NC} $1"
}

log_frontend() {
    echo -e "${CYAN}[FRONTEND]${NC} $1"
}

# 清理函数
cleanup() {
    echo
    log_info "正在清理资源..."
    
    # 停止本地服务
    if [ ! -z "$FRONTEND_PID" ]; then
        log_frontend "停止前端服务..."
        kill $FRONTEND_PID 2>/dev/null || true
        wait $FRONTEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$BACKEND_PID" ]; then
        log_backend "停止后端服务..."
        kill $BACKEND_PID 2>/dev/null || true
        wait $BACKEND_PID 2>/dev/null || true
    fi
    
    # 停止Docker服务
    if [ "$DOCKER_STARTED" -eq 1 ]; then
        log_docker "停止Docker依赖服务..."
        cd "$PROJECT_ROOT"
        docker-compose down >/dev/null 2>&1 || true
    fi
    
    log_success "清理完成"
    exit 0
}

# 注册信号处理
trap cleanup SIGINT SIGTERM

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查端口是否被占用
check_port() {
    local port=$1
    if command_exists lsof; then
        lsof -i :$port >/dev/null 2>&1
    elif command_exists netstat; then
        netstat -ln | grep ":$port " >/dev/null 2>&1
    else
        return 1
    fi
}

# 等待服务就绪
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    
    log_info "等待 $name 服务就绪..."
    
    for i in $(seq 1 $max_attempts); do
        if curl -s "$url" >/dev/null 2>&1; then
            log_success "$name 服务就绪"
            return 0
        fi
        sleep 2
    done
    
    log_warning "$name 服务启动超时"
    return 1
}

echo
echo "============================================================"
echo "           AI鉴宝师项目启动器 v3.0"
echo "============================================================"
echo

# 检查系统要求
log_info "检查系统要求..."

# 检查Docker
if ! command_exists docker; then
    log_error "Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker守护进程
if ! docker info >/dev/null 2>&1; then
    log_error "Docker守护进程未运行，请启动Docker后重试"
    log_info "提示: 请确保Docker服务正在运行"
    exit 1
fi
log_success "Docker检查通过"

# 检查Docker Compose
if ! command_exists docker-compose; then
    log_error "Docker Compose未安装"
    exit 1
fi
log_success "Docker Compose检查通过"

# 检查Node.js
if ! command_exists node; then
    log_error "Node.js未安装，请先安装Node.js 16+"
    exit 1
fi
log_success "Node.js检查通过"

# 检查Python
if ! command_exists python && ! command_exists python3; then
    log_error "Python未安装，请先安装Python 3.9+"
    exit 1
fi
log_success "Python检查通过"

# 检查curl
if ! command_exists curl; then
    log_warning "curl未安装，无法进行健康检查"
fi

# 启动Docker依赖服务
echo
log_docker "启动Docker依赖服务..."
cd "$PROJECT_ROOT"

if ! docker-compose up -d; then
    log_error "Docker服务启动失败"
    exit 1
fi
DOCKER_STARTED=1
log_success "Docker依赖服务启动成功"

# 等待服务就绪
log_info "等待服务就绪..."
sleep 10

# 启动后端服务
echo
log_backend "启动后端服务..."
cd "$BACKEND_DIR"

# 检查虚拟环境
if command_exists conda && conda info --envs | grep -q "$CONDA_ENV"; then
    log_success "使用Conda环境: $CONDA_ENV"
    # 启动后端（后台运行）
    bash -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python app.py" &
    BACKEND_PID=$!
else
    log_warning "Conda环境 $CONDA_ENV 不存在，使用系统Python"
    # 使用系统Python
    if command_exists python3; then
        python3 app.py &
    else
        python app.py &
    fi
    BACKEND_PID=$!
fi

# 等待后端启动
if command_exists curl; then
    wait_for_service "http://localhost:5000/health" "后端" 30
else
    log_info "等待后端服务启动..."
    sleep 10
fi

# 启动前端服务
echo
log_frontend "启动前端服务..."
cd "$FRONTEND_DIR"

# 检查依赖
if [ ! -d "node_modules" ]; then
    log_info "安装前端依赖..."
    if ! npm install; then
        log_error "前端依赖安装失败"
        cleanup
        exit 1
    fi
fi

# 启动前端（后台运行）
npm start &
FRONTEND_PID=$!
log_success "前端服务启动成功"

# 等待前端启动
if command_exists curl; then
    wait_for_service "http://localhost:3000" "前端" 30
else
    log_info "等待前端服务启动..."
    sleep 15
fi

# 显示服务状态
echo
echo "============================================================"
echo "                    服务启动完成"
echo "============================================================"
echo
log_success "前端地址: http://localhost:3000"
log_success "后端地址: http://localhost:5000"
log_success "后端健康检查: http://localhost:5000/health"
echo
log_info "按 Ctrl+C 停止所有服务..."
echo

# 等待用户中断
while true; do
    sleep 1
done