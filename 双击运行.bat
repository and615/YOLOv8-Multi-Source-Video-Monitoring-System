@echo off
chcp 65001 >nul 2>&1
echo ============================================
echo   YOLOv8 视频监控安全检测系统
echo   支持 Nvidia P104/V100 显卡加速
echo ============================================
echo.
echo 正在初始化 YOLOv8 监控安全检测系统...
.\python_env\python.exe monitor.py
pause