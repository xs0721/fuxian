@echo off
REM 一键运行独立测试
REM 用法: run_test 1    (运行测试1)
REM       run_test all  (运行全部 1-7)

setlocal
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set NO_PROXY=*
set PYTHONNOUSERSITE=1

set PYTHON=C:\Users\20747\.conda\envs\watermark\python.exe

if "%1"=="" (
    echo 请指定测试编号: run_test 1-7 或 run_test all
    exit /b
)

if "%1"=="all" (
    for /L %%i in (1,1,7) do (
        echo ================================================
        echo 运行测试 %%i...
        echo ================================================
        %PYTHON% test%%i_*.py
        if errorlevel 1 echo [测试 %%i 出错] & pause
    )
    exit /b
)

if "%1"=="7" (
    %PYTHON% test7_b4_proxy_erasure.py
) else (
    %PYTHON% test%1_*.py
)
endlocal
