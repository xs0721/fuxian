#!/bin/bash
# 上传检查点文件到服务器

SERVER="root@region-46.seetacloud.com"
PORT="52914"
LOCAL_FILE="../文章/第四章引用/引用的代码/multi-bit-text-watermark-master/ckpt/WatermarkDecoder-v_head.pt"
REMOTE_DIR="/root/复现/文章/第四章引用/引用的代码/multi-bit-text-watermark-master/ckpt"

echo "============================================"
echo "上传检查点文件到服务器"
echo "============================================"
echo ""
echo "服务器: $SERVER:$PORT"
echo "密码: XOXoleaqZFux"
echo ""

# 检查本地文件是否存在
if [ ! -f "$LOCAL_FILE" ]; then
    echo "❌ 本地文件不存在: $LOCAL_FILE"
    exit 1
fi

echo "[1/2] 在服务器上创建目录..."
ssh -p $PORT $SERVER "mkdir -p $REMOTE_DIR"
echo ""

echo "[2/2] 上传检查点文件 (4.5 KB)..."
scp -P $PORT "$LOCAL_FILE" "$SERVER:$REMOTE_DIR/"
echo ""

if [ $? -eq 0 ]; then
    echo "============================================"
    echo "✅ 上传成功！"
    echo "============================================"
    echo ""
    echo "现在可以在服务器上运行测试："
    echo "  cd /root/复现/TEST"
    echo "  python test9_multibit_watermark.py"
else
    echo "============================================"
    echo "❌ 上传失败"
    echo "============================================"
    echo "请检查网络连接和密码"
fi
