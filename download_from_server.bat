@echo off
chcp 65001 >nul
echo ========================================
echo 从服务器下载所有项目文件
echo ========================================
echo.

REM 服务器配置
set SERVER_HOST=region-46.seetacloud.com
set SERVER_PORT=52914
set SERVER_USER=root

REM 本地备份目录
set BACKUP_DIR=E:\大模型水印\综述\复现\server_backup_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set BACKUP_DIR=%BACKUP_DIR: =0%

echo 目标目录: %BACKUP_DIR%
echo.
echo 正在连接服务器: %SERVER_USER%@%SERVER_HOST%:%SERVER_PORT%
echo.

REM 创建备份目录
mkdir "%BACKUP_DIR%" 2>nul

echo 下载 /root/复现/TEST/ 目录下的所有文件...
echo.

REM 使用 scp 递归下载
scp -r -P %SERVER_PORT% %SERVER_USER%@%SERVER_HOST%:/root/复现/TEST/* "%BACKUP_DIR%\"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ✅ 下载完成！
    echo.
    echo 文件保存在: %BACKUP_DIR%
    echo.
    explorer "%BACKUP_DIR%"
) else (
    echo.
    echo ❌ 下载失败
    echo.
    echo 可能的原因:
    echo 1. SSH 连接失败（需要输入密码）
    echo 2. 服务器路径不存在
    echo 3. 权限问题
)

echo.
pause
