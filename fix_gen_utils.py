#!/usr/bin/env python3
"""
修复 gen_utils.py 中的数值稳定性问题
使用方法: python fix_gen_utils.py
"""

FILE_PATH = "/root/复现/文章/第四章引用/引用的代码/multi-bit-text-watermark-master/gen_utils.py"

print("=" * 60)
print("修复 gen_utils.py 数值稳定性问题")
print("=" * 60)
print()

# 读取文件
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到并修复第92行
modified = False
for i, line in enumerate(lines):
    # 查找 masked_fill 行
    if 'masked_fill(indices_to_remove, -1000)' in line:
        # 替换为更安全的值，并添加数值稳定性检查
        indent = len(line) - len(line.lstrip())
        lines[i] = ' ' * indent + 'next_tokens_scores = next_tokens_scores.masked_fill(indices_to_remove, -float("inf"))\n'

        # 在下一行添加数值检查
        lines.insert(i + 1, ' ' * indent + '# 数值稳定性检查\n')
        lines.insert(i + 2, ' ' * indent + 'next_tokens_scores = torch.clamp(next_tokens_scores, min=-1e9, max=1e9)\n')
        modified = True
        print(f"✓ 修复第 {i+1} 行的 masked_fill 值")
        break

if not modified:
    # 尝试另一种匹配方式
    for i, line in enumerate(lines):
        if 'next_tokens_scores.masked_fill' in line and '-1000' in line:
            indent = len(line) - len(line.lstrip())
            lines[i] = ' ' * indent + 'next_tokens_scores = next_tokens_scores.masked_fill(indices_to_remove, -1e9)\n'
            lines.insert(i + 1, ' ' * indent + '# Clamp for numerical stability\n')
            lines.insert(i + 2, ' ' * indent + 'next_tokens_scores = torch.clamp(next_tokens_scores, min=-1e9, max=1e9)\n')
            modified = True
            print(f"✓ 修复第 {i+1} 行的 masked_fill 值")
            break

if modified:
    # 写回文件
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print()
    print("=" * 60)
    print("✅ 修复完成！")
    print("=" * 60)
    print()
    print("修改内容:")
    print("  - masked_fill 值改为更安全的 -1e9")
    print("  - 添加了数值范围限制（clamp）")
    print()
    print("现在可以重新运行测试:")
    print("  python test9_multibit_watermark.py")
else:
    print()
    print("⚠️  未找到需要修改的代码")
    print("   文件可能已经被修改或版本不同")

print()
