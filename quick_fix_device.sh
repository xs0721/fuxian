#!/bin/bash
# 快速修复 device_map 配置的脚本

FILE="/root/复现/TEST/test9_multibit_watermark.py"

echo "============================================"
echo "修复 device_map 配置"
echo "============================================"
echo ""

# 备份原文件
cp "$FILE" "${FILE}.backup"
echo "✓ 已备份原文件到 ${FILE}.backup"
echo ""

# 替换 device_map="cpu" 为 device_map="auto"
sed -i 's/device_map="cpu"/device_map="auto"/g' "$FILE"

# 统计修改次数
count=$(grep -c 'device_map="auto"' "$FILE")

echo "✅ 修改完成！"
echo "   共修改 device_map 配置"
echo "   现在模型将自动分配到 GPU"
echo ""
echo "运行测试:"
echo "  python test9_multibit_watermark.py"
echo ""
echo "============================================"
