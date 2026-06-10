@echo off
chcp 65001 >nul
echo ========================================
echo 从服务器下载所有项目文件
echo ========================================
echo.

set SERVER_HOST=region-46.seetacloud.com
set SERVER_PORT=52914
set SERVER_USER=root
set SERVER_PATH=/root/复现/TEST/

REM 创建带时间戳的备份目录
set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set BACKUP_DIR=e:\大模型水印\综述\复现\server_backup_%TIMESTAMP%

echo 本地目标目录: %BACKUP_DIR%
echo 服务器路径: %SERVER_USER%@%SERVER_HOST%:%SERVER_PORT%:%SERVER_PATH%
echo.
echo 开始下载...
echo.

REM 创建目录
mkdir "%BACKUP_DIR%" 2>nul

REM 使用 scp 递归下载所有文件
scp -r -P %SERVER_PORT% %SERVER_USER%@%SERVER_HOST%:%SERVER_PATH%* "%BACKUP_DIR%\"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================
    echo ✅ 下载完成！
    echo ========================================
    echo.
    echo 文件保存在: %BACKUP_DIR%
    echo.
    echo 现在可以在本地推送到 GitHub:
    echo   cd "%BACKUP_DIR%"
    echo   git push -u origin master --force
    echo.

    REM 自动打开文件夹
    explorer "%BACKUP_DIR%"
) else (
    echo.
    echo ========================================
    echo ❌ 下载失败
    echo ========================================
    echo.
    echo 可能需要输入密码: XOXoleaqZFux
)

echo.
pause
