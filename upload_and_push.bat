@echo off
chcp 65001 >nul
echo ========================================
echo 上传并执行 GitHub 推送脚本
echo ========================================
echo.

set SERVER_HOST=region-46.seetacloud.com
set SERVER_PORT=52914
set SERVER_USER=root
set SCRIPT_FILE=e:\大模型水印\综述\复现\push_to_github.sh

echo 步骤 1: 上传脚本到服务器...
scp -P %SERVER_PORT% "%SCRIPT_FILE%" %SERVER_USER%@%SERVER_HOST%:/root/TEST/

if %ERRORLEVEL% neq 0 (
    echo ❌ 上传失败
    pause
    exit /b 1
)

echo ✓ 脚本已上传
echo.

echo 步骤 2: 在服务器上执行脚本...
echo.
echo ========================================
ssh -p %SERVER_PORT% %SERVER_USER%@%SERVER_HOST% "cd /root/TEST && chmod +x push_to_github.sh && bash push_to_github.sh"
echo ========================================
echo.

echo 完成！
pause
