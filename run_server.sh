#!/bin/bash
# 服务器运行脚本 - 在服务器上执行项目

set -e

echo "================================"
echo "在V100 32G服务器上运行项目"
echo "================================"

# 显示GPU信息
echo "GPU信息："
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 设置环境变量（优化显存使用）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "开始运行测试..."
echo "日志将保存到: server_run.log"
echo ""

# 后台运行并输出日志
nohup python -u run_test.py > server_run.log 2>&1 &

# 获取进程ID
PID=$!
echo "进程已启动，PID: $PID"
echo "PID: $PID" > server_run.pid

echo ""
echo "================================"
echo "运行命令："
echo "  查看实时日志: tail -f server_run.log"
echo "  查看进程状态: ps -p $PID"
echo "  停止运行: kill $PID"
echo "  查看GPU使用: watch -n 1 nvidia-smi"
echo "================================"

# 显示前几行日志
sleep 2
echo ""
echo "前几行日志："
head -n 20 server_run.log
