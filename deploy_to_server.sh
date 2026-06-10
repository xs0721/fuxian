#!/bin/bash
# 服务器部署脚本 - 在服务器上运行此脚本

set -e  # 遇到错误立即退出

echo "================================"
echo "开始部署项目到服务器"
echo "================================"

# 1. 克隆代码（如果还没有克隆）
if [ ! -d "reproduce" ]; then
    echo "[1/4] 克隆代码仓库..."
    git clone https://gitee.com/qiu-jianbo555/reproduce.git
    cd reproduce
else
    echo "[1/4] 代码仓库已存在，拉取最新代码..."
    cd reproduce
    git pull origin master
fi

# 2. 检查Python版本
echo "[2/4] 检查Python环境..."
python --version
which python

# 3. 安装依赖
echo "[3/4] 安装Python依赖包..."
# 如果有requirements.txt就安装
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "未找到requirements.txt，跳过依赖安装"
    echo "常见依赖: torch, transformers, datasets, numpy, pandas"
fi

# 4. 检查GPU
echo "[4/4] 检查GPU状态..."
nvidia-smi

echo "================================"
echo "部署完成！"
echo "================================"
echo ""
echo "下一步："
echo "  运行测试: bash run_server.sh"
echo "  或直接运行: python run_test.py"
