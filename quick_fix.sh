#!/bin/bash
# 快速修复脚本 - 直接在服务器上运行
# 使用方法: bash quick_fix.sh

echo "============================================"
echo "快速修复：下载缺失的检查点文件"
echo "============================================"
echo ""

# 设置路径
CKPT_DIR="/root/复现/文章/第四章引用/引用的代码/multi-bit-text-watermark-master/ckpt"
TARGET_FILE="$CKPT_DIR/WatermarkDecoder-v_head.pt"

# 检查文件是否已存在
if [ -f "$TARGET_FILE" ]; then
    echo "✅ 检查点文件已存在: $TARGET_FILE"
    ls -lh "$TARGET_FILE"
    exit 0
fi

# 创建目录
echo "[1/2] 创建目录..."
mkdir -p "$CKPT_DIR"
echo "     ✓ 目录已就绪: $CKPT_DIR"
echo ""

# 下载文件
echo "[2/2] 下载检查点文件..."
cd "$CKPT_DIR"

# 方法1: 使用 Python + huggingface_hub
python3 << 'PYTHON_EOF'
import os
import sys

try:
    from huggingface_hub import hf_hub_download
    import shutil

    print("     使用 huggingface_hub 下载...")
    downloaded = hf_hub_download(
        repo_id="xiaojunxu/WatermarkDecoder-Qwen2.5-1.5b",
        filename="v_head.pt",
        local_dir=".",
        local_dir_use_symlinks=False
    )

    # 重命名
    if os.path.exists("v_head.pt") and not os.path.exists("WatermarkDecoder-v_head.pt"):
        shutil.move("v_head.pt", "WatermarkDecoder-v_head.pt")

    if os.path.exists("WatermarkDecoder-v_head.pt"):
        size_kb = os.path.getsize("WatermarkDecoder-v_head.pt") / 1024
        print(f"     ✓ 下载成功 ({size_kb:.1f} KB)")
        sys.exit(0)
    else:
        print("     ❌ 文件未找到")
        sys.exit(1)

except ImportError:
    print("     ⚠️  huggingface_hub 未安装")
    print("     正在安装...")
    os.system("pip install -q huggingface_hub")
    print("     请重新运行此脚本")
    sys.exit(1)
except Exception as e:
    print(f"     ❌ 下载失败: {e}")
    sys.exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✅ 修复完成！"
    echo "============================================"
    echo ""
    echo "现在可以运行测试："
    echo "  cd /root/复现/TEST"
    echo "  python test9_multibit_watermark.py"
else
    echo ""
    echo "============================================"
    echo "❌ 自动下载失败"
    echo "============================================"
    echo ""
    echo "手动下载方法："
    echo "1. 访问: https://huggingface.co/xiaojunxu/WatermarkDecoder-Qwen2.5-1.5b/tree/main"
    echo "2. 下载 v_head.pt"
    echo "3. 上传到: $TARGET_FILE"
fi
