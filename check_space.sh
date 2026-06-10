#!/bin/bash
# 服务器空间清理和诊断脚本

echo "============================================"
echo "空间诊断和清理"
echo "============================================"
echo ""

# 1. 检查磁盘空间
echo "[1] 磁盘空间使用情况:"
df -h | grep -E "Filesystem|/root|/$"
echo ""

# 2. 检查内存使用
echo "[2] 内存使用情况:"
free -h
echo ""

# 3. 查找大文件
echo "[3] /root 下最大的10个文件/目录:"
du -sh /root/* 2>/dev/null | sort -rh | head -10
echo ""

# 4. 清理 HuggingFace 缓存中的重复文件
echo "[4] 清理建议:"
echo ""
HF_CACHE="/root/autodl-tmp/hf_cache"
if [ -d "$HF_CACHE" ]; then
    echo "   HuggingFace 缓存: $(du -sh $HF_CACHE 2>/dev/null | cut -f1)"
    echo "   可以删除旧的模型缓存"
fi

WM_CACHE="/root/.cache/watermark_fp16"
if [ -d "$WM_CACHE" ]; then
    echo "   水印模型缓存: $(du -sh $WM_CACHE 2>/dev/null | cut -f1)"
fi

HF_HOME="/root/.cache/huggingface"
if [ -d "$HF_HOME" ]; then
    echo "   ~/.cache/huggingface: $(du -sh $HF_HOME 2>/dev/null | cut -f1)"
fi

echo ""
echo "[5] 安全清理选项:"
echo ""
echo "   # 清理 pip 缓存"
echo "   pip cache purge"
echo ""
echo "   # 清理 HuggingFace 下载缓存（保留模型）"
echo "   rm -rf /root/.cache/huggingface/hub/*/blobs/*.tmp"
echo ""
echo "   # 清理临时文件"
echo "   rm -rf /tmp/* 2>/dev/null"
echo ""
echo "   # 如果不再需要，删除重复的 HF 缓存"
echo "   # rm -rf /root/autodl-tmp/hf_cache"
echo ""

echo "============================================"
