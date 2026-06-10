#!/bin/bash
# 修复test_common.py中的设备不匹配问题

cd /root/复现/TEST/

echo "========================================"
echo "修复 test_common.py 设备不匹配错误"
echo "========================================"

# 备份
cp test_common.py test_common.py.backup_device_fix_$(date +%Y%m%d_%H%M%S)
echo "✓ 已备份原文件"

# 修复detect_sweet函数中的设备问题
python << 'PYTHON_FIX'
with open('test_common.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复1: 在detect_sweet函数开始处，确保tokens在CPU上
old_line1 = '    tokens = tokenizer.encode(text, return_tensors="pt")[0]'
new_line1 = '    tokens = tokenizer.encode(text, return_tensors="pt")[0].cpu()'

# 修复2: 检查greenlist时确保都在CPU上
old_line2 = '        if tokens[i] in greenlist:'
new_line2 = '        if tokens[i].item() in greenlist.tolist():'

content = content.replace(old_line1, new_line1)
content = content.replace(old_line2, new_line2)

with open('test_common.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 已修复detect_sweet设备不匹配问题")
PYTHON_FIX

echo ""
echo "✅ 修复完成！"
echo ""
echo "现在可以重新运行test7:"
echo "  python test7_b4_proxy_erasure.py"
