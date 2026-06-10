#!/bin/bash
set -e

echo "========================================" 
echo "修复 CSV 文件损坏问题"
echo "========================================"

# 检查并备份损坏的文件
if [ -f "watermark_benchmark_results.csv" ]; then
    SIZE=$(stat -c%s watermark_benchmark_results.csv 2>/dev/null || echo "0")
    echo "当前文件大小: ${SIZE} 字节"
    
    if [ "$SIZE" -lt 1000 ]; then
        echo "⚠️  文件损坏，备份中..."
        mv watermark_benchmark_results.csv watermark_benchmark_results.csv.broken.$(date +%Y%m%d_%H%M%S)
    fi
fi

# 重新生成CSV
echo "开始重新生成 CSV 文件..."
echo "========================================"
python run_experiment.py

# 验证
if [ -f "watermark_benchmark_results.csv" ]; then
    NEW_SIZE=$(stat -c%s watermark_benchmark_results.csv)
    NEW_LINES=$(wc -l < watermark_benchmark_results.csv)
    echo ""
    echo "✅ 成功！文件大小: ${NEW_SIZE} 字节，行数: ${NEW_LINES}"
    echo "前5行预览:"
    head -5 watermark_benchmark_results.csv
else
    echo "✗ 生成失败"
    exit 1
fi
