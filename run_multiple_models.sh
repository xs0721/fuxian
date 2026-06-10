#!/bin/bash
# AutoDL服务器上运行多模型测试的脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "多模型水印测试自动化脚本"
echo "=========================================="

# 配置
CACHE_DIR="/root/autodl-tmp/hf_cache"
export HF_ENDPOINT=https://hf-mirror.com

# 创建缓存目录
mkdir -p $CACHE_DIR

# 模型列表
MODELS=("facebook/opt-125m" "facebook/opt-6.7b")
SAMPLES=(200 200)

# 依次运行每个模型
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    SAMPLE="${SAMPLES[$i]}"

    echo ""
    echo "=========================================="
    echo "[$((i+1))/${#MODELS[@]}] 开始测试: $MODEL"
    echo "样本数: $SAMPLE"
    echo "=========================================="

    # 修改配置
    sed -i "s|MODEL_NAME = .*|MODEL_NAME = \"$MODEL\"|" run_experiment.py
    sed -i "s|TEST_SAMPLE_SIZE = .*|TEST_SAMPLE_SIZE = $SAMPLE  # 自动修改|" run_experiment.py
    sed -i "s|CACHE_DIR = .*|CACHE_DIR = \"$CACHE_DIR\"|" run_experiment.py

    # 运行测试
    python run_experiment.py

    # 重命名结果文件
    MODEL_NAME=$(echo $MODEL | tr '/' '_')
    if [ -f "watermark_benchmark_results.csv" ]; then
        cp watermark_benchmark_results.csv "results_${MODEL_NAME}_${SAMPLE}samples.csv"
        echo "结果已保存: results_${MODEL_NAME}_${SAMPLE}samples.csv"
    fi

    echo "[$((i+1))/${#MODELS[@]}] 完成: $MODEL"
done

echo ""
echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
echo "生成的结果文件："
ls -lh results_*.csv

echo ""
echo "请下载这些CSV文件到本地"
echo "然后在控制台关机节省费用！"
