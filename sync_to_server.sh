#!/bin/bash
# 同步文件到 Linux 服务器
# 使用方法: bash sync_to_server.sh

SERVER="root@region-46.seetacloud.com"
PORT="52914"
REMOTE_DIR="/root/复现/TEST"

echo "=== 同步文件到服务器 ==="
echo "服务器: $SERVER:$PORT"
echo "目标目录: $REMOTE_DIR"
echo ""

# 确保远程目录存在
echo "1. 创建远程目录..."
ssh -p $PORT $SERVER "mkdir -p $REMOTE_DIR"

# 同步测试脚本
echo "2. 同步测试脚本..."
scp -P $PORT test9_multibit_watermark.py $SERVER:$REMOTE_DIR/

# 同步依赖脚本（如果需要）
echo "3. 同步依赖文件..."
if [ -f "prepare_multibit_models.py" ]; then
    scp -P $PORT prepare_multibit_models.py $SERVER:$REMOTE_DIR/
fi

# 同步检查点文件
echo "4. 同步检查点文件..."
CKPT_FILE="../文章/第四章引用/引用的代码/multi-bit-text-watermark-master/ckpt/WatermarkDecoder-v_head.pt"
if [ -f "$CKPT_FILE" ]; then
    ssh -p $PORT $SERVER "mkdir -p /root/复现/文章/第四章引用/引用的代码/multi-bit-text-watermark-master/ckpt"
    scp -P $PORT "$CKPT_FILE" $SERVER:/root/复现/文章/第四章引用/引用的代码/multi-bit-text-watermark-master/ckpt/
    echo "  ✅ 检查点文件已上传"
else
    echo "  ⚠️  本地未找到检查点文件，将在服务器上自动下载"
fi

# 同步 multi-bit 项目的关键文件
echo "5. 同步 multi-bit 项目文件..."
MULTIBIT_DIR="../文章/第四章引用/引用的代码/multi-bit-text-watermark-master"
if [ -d "$MULTIBIT_DIR" ]; then
    ssh -p $PORT $SERVER "mkdir -p /root/复现/文章/第四章引用/引用的代码/multi-bit-text-watermark-master"
    scp -P $PORT "$MULTIBIT_DIR"/*.py $SERVER:/root/复现/文章/第四章引用/引用的代码/multi-bit-text-watermark-master/ 2>/dev/null || true
    scp -P $PORT "$MULTIBIT_DIR"/watermark_test_text.txt $SERVER:/root/复现/文章/第四章引用/引用的代码/multi-bit-text-watermark-master/ 2>/dev/null || true
fi

echo ""
echo "=== 同步完成 ==="
echo "现在可以连接到服务器运行测试："
echo "  ssh -p $PORT $SERVER"
echo "  cd $REMOTE_DIR"
echo "  python test9_multibit_watermark.py"
