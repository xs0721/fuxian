@echo off
REM Windows 批处理脚本：上传修改后的文件到服务器
REM 使用方法：双击运行，然后输入密码

echo ========================================
echo 上传 run_experiment.py 到服务器
echo ========================================
echo.

echo 服务器: region-46.seetacloud.com
echo 端口: 52914
echo 用户: root
echo.

echo 正在上传...
scp -P 52914 "e:\大模型水印\综述\复现\run_experiment.py" root@region-46.seetacloud.com:/root/复现/TEST/run_experiment.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo 上传成功！
    echo ========================================
    echo.
    echo 现在在服务器上执行:
    echo   cd /root/复现/TEST
    echo   nohup python run_experiment.py ^> run_experiment.log 2^>^&1 ^&
    echo   tail -f run_experiment.log
    echo.
) else (
    echo.
    echo ========================================
    echo 上传失败！
    echo ========================================
    echo.
    echo 请检查:
    echo   1. 网络连接
    echo   2. SSH密码是否正确
    echo   3. 服务器是否可访问
    echo.
)

pause
