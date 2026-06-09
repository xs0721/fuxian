#!/bin/bash
# 服务器端 Git 推送脚本

echo "=========================================="
echo "将服务器代码推送到远程仓库"
echo "=========================================="
echo ""

# 1. 检查当前 git 状态
echo "步骤 1: 检查 Git 仓库状态"
cd /root/TEST

if [ ! -d ".git" ]; then
    echo "当前目录不是 Git 仓库，初始化..."
    git init
    echo "✓ Git 仓库已初始化"
else
    echo "✓ 已存在 Git 仓库"
fi
echo ""

# 2. 显示当前状态
echo "步骤 2: 当前文件状态"
git status --short | head -20
echo ""

# 3. 配置 Git 用户信息（如果未配置）
echo "步骤 3: 配置 Git 用户信息"
if [ -z "$(git config user.name)" ]; then
    echo "请输入你的 Git 用户名: "
    read GIT_USERNAME
    git config user.name "$GIT_USERNAME"
fi

if [ -z "$(git config user.email)" ]; then
    echo "请输入你的 Git 邮箱: "
    read GIT_EMAIL
    git config user.email "$GIT_EMAIL"
fi

echo "✓ Git 用户: $(git config user.name) <$(git config user.email)>"
echo ""

# 4. 添加所有文件
echo "步骤 4: 添加所有文件到暂存区"
git add -A
echo "✓ 文件已添加"
echo ""

# 5. 提交
echo "步骤 5: 提交更改"
echo "请输入提交信息 (默认: 'Update from server'): "
read COMMIT_MSG
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Update from server"
fi

git commit -m "$COMMIT_MSG"
echo ""

# 6. 配置远程仓库
echo "步骤 6: 配置远程仓库"
REMOTE_URL=$(git remote get-url origin 2>/dev/null)

if [ -z "$REMOTE_URL" ]; then
    echo "未找到远程仓库，请选择:"
    echo "1) GitHub"
    echo "2) Gitee (码云)"
    read -p "选择 (1/2): " CHOICE

    echo "请输入仓库 URL (例如: https://github.com/username/repo.git): "
    read REPO_URL

    git remote add origin "$REPO_URL"
    echo "✓ 已添加远程仓库: $REPO_URL"
else
    echo "✓ 当前远程仓库: $REMOTE_URL"
    echo "是否更改? (y/n): "
    read CHANGE_REMOTE
    if [ "$CHANGE_REMOTE" = "y" ]; then
        echo "请输入新的仓库 URL: "
        read REPO_URL
        git remote set-url origin "$REPO_URL"
        echo "✓ 已更新远程仓库: $REPO_URL"
    fi
fi
echo ""

# 7. 推送到远程
echo "步骤 7: 推送到远程仓库"
echo "当前分支: $(git branch --show-current)"
echo "准备推送..."
echo ""

# 尝试推送
git push -u origin $(git branch --show-current) 2>&1 | tee /tmp/git_push.log

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推送成功！"
else
    echo ""
    echo "⚠️ 推送失败，可能需要:"
    echo "1. 设置 Git 凭据"
    echo "2. 如果是首次推送，使用: git push -u origin master --force"
    echo ""
    echo "错误信息已保存到: /tmp/git_push.log"
fi

echo ""
echo "=========================================="
echo "完成"
echo "=========================================="
