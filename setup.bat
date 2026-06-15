@echo off
chcp 65001 >nul
title LLM 翻译评测 — 一键安装

echo ============================================
echo   LLM 翻译评测框架 — 环境安装
echo ============================================
echo.

cd /d "%~dp0"

:: 检查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/2] 创建虚拟环境 llm_eval_env ...
if not exist llm_eval_env\Scripts\python.exe (
    python -m venv llm_eval_env
    echo       虚拟环境创建完成。
) else (
    echo       虚拟环境已存在，跳过。
)

echo.
echo [2/2] 安装依赖包 (openai sacrebleu pandas tqdm datasets) ...
llm_eval_env\Scripts\python.exe -m pip install --quiet openai sacrebleu pandas tqdm datasets
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络连接。
    pause
    exit /b 1
)
echo       依赖安装完成。

echo.
echo ============================================
echo   安装完毕！
echo.
echo   使用方法:
echo   1. 将 llama-cli.exe 放入 bin\ 目录
echo   2. 将 .gguf 模型文件放入 models\ 目录
echo   3. 双击 run.bat 开始评测
echo ============================================
pause
