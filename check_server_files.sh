#!/bin/bash
# 检查服务器文件完整性

echo "=== 检查服务器文件 ==="
echo ""

# 检查主要目录
echo "1. 检查 /root/复现/TEST 目录:"
ssh -p 52914 root@region-46.seetacloud.com "ls -lh /root/复现/TEST/ | head -20"
echo ""

echo "2. 检查测试脚本:"
ssh -p 52914 root@region-46.seetacloud.com "ls -lh /root/复现/TEST/test*.py | wc -l"
echo ""

echo "3. 检查 ckpt 目录:"
ssh -p 52914 root@region-46.seetacloud.com "ls -lh /root/复现/TEST/ckpt/ 2>/dev/null || echo '目录不存在'"
echo ""

echo "4. 检查关键文件:"
ssh -p 52914 root@region-46.seetacloud.com "ls -lh /root/复现/TEST/gen_utils.py /root/复现/TEST/test9_multibit_watermark.py 2>&1"
echo ""

echo "✓ 检查完成"
