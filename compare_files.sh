#!/bin/bash
# 对比服务器和本地文件

echo "=========================================="
echo "对比服务器和本地文件"
echo "=========================================="
echo ""

# 服务器文件列表
echo "正在获取服务器文件列表..."
ssh -p 52914 root@region-46.seetacloud.com "cd /root/复现/TEST && find . -maxdepth 1 -name '*.py' -o -name '*.sh' | sort" > /tmp/server_files.txt

echo "服务器上的主要文件 (test*.py):"
ssh -p 52914 root@region-46.seetacloud.com "cd /root/复现/TEST && ls test*.py | wc -l"
echo ""

# 本地文件列表
echo "请输入你下载到的本地目录路径："
read LOCAL_DIR

if [ -d "$LOCAL_DIR" ]; then
    echo "本地文件数量 (test*.py):"
    cd "$LOCAL_DIR"
    ls test*.py 2>/dev/null | wc -l
    echo ""

    echo "缺失的文件:"
    diff <(ssh -p 52914 root@region-46.seetacloud.com "cd /root/复现/TEST && ls test*.py | sort") <(ls test*.py 2>/dev/null | sort)
else
    echo "本地目录不存在"
fi

echo ""
echo "=========================================="
echo "完成"
echo "=========================================="
