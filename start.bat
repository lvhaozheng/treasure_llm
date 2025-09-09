@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

REM ============================================================
REM AI鉴宝师项目启动脚本 (Windows版)
REM 自动启动所有必要的服务：Docker依赖、后端、前端
REM ============================================================

title AI鉴宝师项目启动器

REM 设置项目根目录
set "PROJECT_ROOT=%~dp0"
set "BACKEND_DIR=%PROJECT_ROOT%backend"
set "FRONTEND_DIR=%PROJECT_ROOT%frontend"
set "CONDA_ENV=treasure_llm"

REM 进程ID变量
set "FRONTEND_PID="
set "BACKEND_PID="
set "DOCKER_STARTED=0"

REM 颜色标记
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"
set "DOCKER=[DOCKER]"
set "BACKEND=[BACKEND]"
set "FRONTEND=[FRONTEND]"

echo.
echo ============================================================
echo           AI鉴宝师项目启动器 v3.0
echo ============================================================
echo.

REM 检查系统要求
echo %INFO% 检查系统要求...

REM 检查Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Docker未安装或不可用，请先安装Docker Desktop
    pause
    exit /b 1
)

REM 检查Docker守护进程
docker info >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Docker Desktop未运行，请启动Docker Desktop后重试
    echo %INFO% 提示: 请确保Docker Desktop正在运行
    pause
    exit /b 1
)
echo %SUCCESS% Docker检查通过

REM 检查Docker Compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Docker Compose未安装
    pause
    exit /b 1
)
echo %SUCCESS% Docker Compose检查通过

REM 检查Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Node.js未安装，请先安装Node.js 16+
    pause
    exit /b 1
)
echo %SUCCESS% Node.js检查通过

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Python未安装，请先安装Python 3.9+
    pause
    exit /b 1
)
echo %SUCCESS% Python检查通过

REM 设置清理函数
set "CLEANUP_CALLED=0"

:cleanup
if "%CLEANUP_CALLED%"=="1" goto :eof
set "CLEANUP_CALLED=1"
echo.
echo %INFO% 正在清理资源...

REM 停止本地服务
if defined FRONTEND_PID (
    echo %FRONTEND% 停止前端服务...
    taskkill /f /pid %FRONTEND_PID% >nul 2>&1
)

if defined BACKEND_PID (
    echo %BACKEND% 停止后端服务...
    taskkill /f /pid %BACKEND_PID% >nul 2>&1
)

REM 停止Docker服务
if "%DOCKER_STARTED%"=="1" (
    echo %DOCKER% 停止Docker依赖服务...
    cd /d "%PROJECT_ROOT%"
    docker-compose down >nul 2>&1
)

echo %SUCCESS% 清理完成
goto :eof

REM 注册CTRL+C处理
if "%1"=="cleanup" goto cleanup

REM 启动Docker依赖服务
echo.
echo %DOCKER% 启动Docker依赖服务...
cd /d "%PROJECT_ROOT%"
docker-compose up -d
if errorlevel 1 (
    echo %ERROR% Docker服务启动失败
    call :cleanup
    pause
    exit /b 1
)
set "DOCKER_STARTED=1"
echo %SUCCESS% Docker依赖服务启动成功

REM 等待服务就绪
echo %INFO% 等待服务就绪...
timeout /t 10 /nobreak >nul

REM 启动后端服务
echo.
echo %BACKEND% 启动后端服务...
cd /d "%BACKEND_DIR%"

REM 检查虚拟环境
conda info --envs | findstr "%CONDA_ENV%" >nul 2>&1
if errorlevel 1 (
    echo %WARNING% Conda环境 %CONDA_ENV% 不存在，使用系统Python
    start "AI鉴宝师-后端" /min cmd /c "python app.py"
) else (
    echo %SUCCESS% 使用Conda环境: %CONDA_ENV%
    start "AI鉴宝师-后端" /min cmd /c "conda activate %CONDA_ENV% && python app.py"
)

REM 等待后端启动
echo %INFO% 等待后端服务启动...
timeout /t 5 /nobreak >nul

REM 检查后端健康状态
for /l %%i in (1,1,30) do (
    curl -s http://localhost:5000/health >nul 2>&1
    if not errorlevel 1 (
        echo %SUCCESS% 后端服务启动成功
        goto backend_ready
    )
    timeout /t 2 /nobreak >nul
)
echo %WARNING% 后端服务启动超时，但继续启动前端

:backend_ready

REM 启动前端服务
echo.
echo %FRONTEND% 启动前端服务...
cd /d "%FRONTEND_DIR%"

REM 检查依赖
if not exist "node_modules" (
    echo %INFO% 安装前端依赖...
    npm install
    if errorlevel 1 (
        echo %ERROR% 前端依赖安装失败
        call :cleanup
        pause
        exit /b 1
    )
)

start "AI鉴宝师-前端" cmd /c "npm start"
echo %SUCCESS% 前端服务启动成功

REM 显示服务状态
echo.
echo ============================================================
echo                    服务启动完成
echo ============================================================
echo.
echo %SUCCESS% 前端地址: http://localhost:3000
echo %SUCCESS% 后端地址: http://localhost:5000
echo %SUCCESS% 后端健康检查: http://localhost:5000/health
echo.
echo %INFO% 按任意键停止所有服务...
echo.

REM 等待用户输入
pause >nul

REM 清理资源
call :cleanup

echo.
echo %SUCCESS% 所有服务已停止
pause
exit /b 0