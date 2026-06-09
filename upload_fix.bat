@echo off
echo ============================================
echo 上传修复后的文件到服务器
echo ============================================
echo.
echo 服务器: root@region-46.seetacloud.com:52914
echo 密码: XOXoleaqZFux
echo.
echo 正在上传 test9_multibit_watermark.py...
echo.

scp -P 52914 test9_multibit_watermark.py root@region-46.seetacloud.com:/root/复现/TEST/

echo.
echo ============================================
echo 上传完成！现在可以在服务器上重新运行测试
echo ============================================
pause
