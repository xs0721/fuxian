#!/bin/bash
################################################################################
# 自动运行所有测试 - 按顺序执行
# 适用于AutoDL服务器，可以关闭窗口继续运行
################################################################################

# set -e  # 遇到错误时继续运行，不停止

echo "=================================="
echo "🚀 开始自动测试流程"
echo "=================================="
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📂 项目目录: $SCRIPT_DIR"
echo ""

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# 开启学术加速（如果可用）
if [ -f /etc/network_turbo ]; then
    echo "📡 开启学术加速..."
    source /etc/network_turbo
fi

################################################################################
# 安装依赖
################################################################################
echo ""
echo "=================================="
echo "📦 步骤0: 检查并安装依赖"
echo "=================================="
pip install tabulate -q -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "✅ 依赖安装完成"

################################################################################
# 测试1: 主要基准测试（最重要）
################################################################################
echo ""
echo "=================================="
echo "📊 步骤1: 主要基准测试 (run_experiment.py)"
echo "预计时间: 1-2小时"
echo "=================================="

if [ -f "watermark_benchmark_results.csv" ]; then
    echo "⚠️  发现已存在的结果文件，备份..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mkdir -p backup_${TIMESTAMP}
    mv watermark_benchmark_results.csv backup_${TIMESTAMP}/ 2>/dev/null || true
    mv table_*.csv backup_${TIMESTAMP}/ 2>/dev/null || true
    mv benchmark_comparison_plot.png backup_${TIMESTAMP}/ 2>/dev/null || true
fi

python run_experiment.py 2>&1 | tee run_experiment.log
echo "✅ 步骤1完成: 基准测试"

################################################################################
# 测试2: 长度敏感性测试
################################################################################
echo ""
echo "=================================="
echo "📏 步骤2: 长度敏感性测试 (length_sensitivity_test.py)"
echo "预计时间: 30-60分钟"
echo "=================================="

if [ -f "length_sensitivity_test.py" ]; then
    python length_sensitivity_test.py 2>&1 | tee length_sensitivity.log
    echo "✅ 步骤2完成: 长度敏感性测试"
else
    echo "⚠️  未找到 length_sensitivity_test.py，跳过"
fi

################################################################################
# 测试3: 快速验证测试（可选）
################################################################################
echo ""
echo "=================================="
echo "🔍 步骤3: 快速验证测试（可选）"
echo "预计时间: 20-30分钟"
echo "=================================="

QUICK_TESTS=(
    "test_official_watermarks_quick.py"
    "test_k_semstamp_quick.py"
    "test_icw_quick.py"
    "test_xsir_quick.py"
    "test_ewd_detection.py"
)

for test_file in "${QUICK_TESTS[@]}"; do
    if [ -f "$test_file" ]; then
        echo ""
        echo "▶️  运行: $test_file"
        python "$test_file" 2>&1 | tee "${test_file%.py}.log" || echo "⚠️  $test_file 运行失败，继续..."
        echo "✅ $test_file 完成"
    else
        echo "⚠️  未找到 $test_file，跳过"
    fi
done

echo "✅ 步骤3完成: 快速验证"

################################################################################
# 测试4: 其他测试脚本（如果存在）
################################################################################
echo ""
echo "=================================="
echo "🧪 步骤4: 其他测试脚本"
echo "预计时间: 10-20分钟"
echo "=================================="

OTHER_TESTS=(
    "run_test.py"
)

for test_file in "${OTHER_TESTS[@]}"; do
    if [ -f "$test_file" ]; then
        echo ""
        echo "▶️  运行: $test_file"
        python "$test_file" 2>&1 | tee "${test_file%.py}.log" || echo "⚠️  $test_file 运行失败，继续..."
        echo "✅ $test_file 完成"
    else
        echo "⚠️  未找到 $test_file，跳过"
    fi
done

echo "✅ 步骤4完成: 其他测试"

################################################################################
# 汇总结果
################################################################################
echo ""
echo "=================================="
echo "📋 测试完成汇总"
echo "=================================="

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "⏱️  总运行时间: ${HOURS}小时 ${MINUTES}分钟"
echo ""
echo "📁 生成的文件："
ls -lh *.csv *.png *.log 2>/dev/null || echo "  (未找到输出文件)"
echo ""
echo "🎉 所有测试完成！"
echo ""
echo "=================================="
echo "下一步："
echo "1. 查看结果: cat watermark_benchmark_results.csv"
echo "2. 下载文件: 用VS Code或scp下载所有.csv和.png文件"
echo "3. 查看日志: cat *.log"
echo "=================================="
