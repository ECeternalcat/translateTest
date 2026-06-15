@echo off
chcp 65001 >nul
title LLM 翻译评测

cd /d "%~dp0"

:: 检查虚拟环境
if not exist llm_eval_env\Scripts\python.exe (
    echo [提示] 虚拟环境不存在，正在自动安装...
    call setup.bat
    if errorlevel 1 exit /b 1
)

echo ============================================
echo   LLM 翻译评测 — 启动中...
echo ============================================
echo.

llm_eval_env\Scripts\python.exe run_benchmark.py

echo.
echo ============================================
echo   评测结束，按任意键退出。
echo ============================================
pause
