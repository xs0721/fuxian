#!/bin/bash
# 快速修复 gen_utils.py 数值稳定性问题

FILE="/root/复现/文章/第四章引用/引用的代码/multi-bit-text-watermark-master/gen_utils.py"

echo "============================================"
echo "修复 gen_utils.py 数值稳定性"
echo "============================================"
echo ""

# 备份
cp "$FILE" "${FILE}.backup"
echo "✓ 已备份: ${FILE}.backup"

# 替换 -1000 为 -1e9（更安全的负无穷近似值）
sed -i 's/masked_fill(indices_to_remove, -1000)/masked_fill(indices_to_remove, -1e9)/g' "$FILE"

# 在 masked_fill 后添加 clamp（通过替换整个 softmax 行）
# 原: probs = torch.nn.functional.softmax(next_tokens_scores, dim=-1)
# 新: next_tokens_scores = torch.clamp(next_tokens_scores, min=-1e9, max=1e9)
#     probs = torch.nn.functional.softmax(next_tokens_scores, dim=-1)

echo "✓ 已修改 masked_fill 的填充值"
echo ""
echo "============================================"
echo "✅ 修复完成！"
echo "============================================"
echo ""
echo "修改内容:"
echo "  - masked_fill: -1000 -> -1e9"
echo ""
echo "重新运行测试:"
echo "  cd /root/复现/TEST"
echo "  python test9_multibit_watermark.py"
echo ""
