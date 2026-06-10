#!/bin/bash
# 清理大文件并推送到 Gitee

echo "=========================================="
echo "清理大文件并推送到 Gitee"
echo "=========================================="
echo ""

cd /root/复现/TEST/

# 1. 查找大文件（超过 50MB）
echo "步骤 1: 查找大文件（>50MB）..."
find . -type f -size +50M ! -path "./.git/*" -exec ls -lh {} \; | awk '{print $9, $5}'
echo ""

# 2. 创建/更新 .gitignore 排除大文件
echo "步骤 2: 创建 .gitignore 排除大文件..."
cat > .gitignore << 'EOF'
# 模型文件和缓存
*.bin
*.pt
*.pth
*.safetensors
*.ckpt
*.h5
*.pb
*.onnx

# 大型数据文件
*.tar
*.tar.gz
*.zip
*.7z
*.tar.xz

# 数据集
*.csv
*.json
*.jsonl
*.parquet

# 日志文件
*.log

# 备份文件
*.bak
*.backup
*.old
backup_*/

# Node.js
node-*
node_modules/

# Python
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp

# 其他
E:/
.DS_Store
EOF

echo "✓ .gitignore 已创建"
echo ""

# 3. 清理 Git 缓存
echo "步骤 3: 清理 Git 缓存..."
git rm -r --cached .
echo ""

# 4. 重新添加文件（会应用 .gitignore）
echo "步骤 4: 重新添加文件..."
git add .
echo ""

# 5. 提交
echo "步骤 5: 提交更改..."
git commit -m "Remove large files and update .gitignore - $(date +%Y-%m-%d)"
echo ""

# 6. 显示仓库大小
echo "步骤 6: 检查仓库大小..."
du -sh .git
echo ""

# 7. 推送到 Gitee
echo "步骤 7: 推送到 Gitee..."
git push -u gitee master --force

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推送成功！"
    echo "查看仓库: https://gitee.com/qiu-jianbo555/reproduce"
else
    echo ""
    echo "❌ 推送仍然失败"
    echo ""
    echo "可能需要进一步清理。运行以下命令查看最大的文件:"
    echo "git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | sort -n -k 2 | tail -20"
fi

echo ""
echo "=========================================="
echo "完成"
echo "=========================================="
