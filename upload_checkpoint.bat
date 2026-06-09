@echo off
chcp 65001 >nul
echo ============================================
echo 上传检查点文件到服务器
echo ============================================
echo.
echo 服务器: root@region-46.seetacloud.com:52914
echo 密码: XOXoleaqZFux
echo.

set "LOCAL_FILE=..\文章\第四章引用\引用的代码\multi-bit-text-watermark-master\ckpt\WatermarkDecoder-v_head.pt"
set "REMOTE_DIR=/root/复现/文章/第四章引用/引用的代码/multi-bit-text-watermark-master/ckpt"

echo [1/2] 在服务器上创建目录...
ssh -p 52914 root@region-46.seetacloud.com "mkdir -p %REMOTE_DIR%"
echo.

echo [2/2] 上传检查点文件 (4.5 KB)...
scp -P 52914 "%LOCAL_FILE%" "root@region-46.seetacloud.com:%REMOTE_DIR%/"
echo.

if %ERRORLEVEL% EQU 0 (
    echo ============================================
    echo ✅ 上传成功！
    echo ============================================
    echo.
    echo 现在可以在服务器上运行测试：
    echo   cd /root/复现/TEST
    echo   python test9_multibit_watermark.py
) else (
    echo ============================================
    echo ❌ 上传失败
    echo ============================================
    echo 请检查网络连接和密码
)
echo.
pause
