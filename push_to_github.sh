#!/bin/bash
# 推送到 GitHub 仓库: https://github.com/xs0721/TEST.git

echo "=========================================="
echo "推送代码到 GitHub"
echo "=========================================="
echo ""

cd /root/TEST

# 1. 检查并初始化 Git
if [ ! -d ".git" ]; then
    echo "初始化 Git 仓库..."
    git init
    echo "✓ Git 仓库已初始化"
else
    echo "✓ Git 仓库已存在"
fi
echo ""

# 2. 配置 Git 用户信息
echo "配置 Git 用户信息..."
git config user.name "xs0721"
git config user.email "xs0721@users.noreply.github.com"  # GitHub 默认邮箱
echo "✓ 用户配置完成"
echo ""

# 3. 添加所有文件
echo "添加文件..."
git add -A
echo "✓ 文件已添加"
echo ""

# 4. 提交
echo "提交更改..."
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
git commit -m "Server update: $TIMESTAMP"
echo "✓ 提交完成"
echo ""

# 5. 配置远程仓库
echo "配置远程仓库..."
REMOTE_URL=$(git remote get-url origin 2>/dev/null)

if [ -z "$REMOTE_URL" ]; then
    git remote add origin https://github.com/xs0721/TEST.git
    echo "✓ 已添加远程仓库: https://github.com/xs0721/TEST.git"
else
    echo "当前远程仓库: $REMOTE_URL"
    git remote set-url origin https://github.com/xs0721/TEST.git
    echo "✓ 已更新远程仓库"
fi
echo ""

# 6. 查看当前分支
CURRENT_BRANCH=$(git branch --show-current)
if [ -z "$CURRENT_BRANCH" ]; then
    # 如果没有分支，创建 main 分支
    git branch -M main
    CURRENT_BRANCH="main"
fi
echo "当前分支: $CURRENT_BRANCH"
echo ""

# 7. 推送
echo "=========================================="
echo "开始推送到 GitHub..."
echo "=========================================="
echo ""
echo "注意: 如果是首次推送，可能需要输入 GitHub 用户名和 Personal Access Token"
echo ""

git push -u origin $CURRENT_BRANCH

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️ 推送失败，尝试强制推送..."
    git push -u origin $CURRENT_BRANCH --force
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推送成功！"
    echo "查看你的仓库: https://github.com/xs0721/TEST"
else
    echo ""
    echo "❌ 推送失败"
    echo ""
    echo "可能原因:"
    echo "1. 需要 GitHub Personal Access Token (不是密码)"
    echo "2. 网络问题"
    echo ""
    echo "获取 Token 的步骤:"
    echo "1. 访问: https://github.com/settings/tokens"
    echo "2. 点击 'Generate new token (classic)'"
    echo "3. 勾选 'repo' 权限"
    echo "4. 生成后复制 token"
    echo "5. 再次运行此脚本，用户名输入: xs0721，密码输入: token"
fi

echo ""
echo "=========================================="
echo "完成"
echo "=========================================="
