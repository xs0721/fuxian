#!/usr/bin/env python3
"""
快速修复脚本 - 直接在服务器上修改 device_map
运行方法: python quick_fix_device.py
"""
import re

FILE_PATH = "/root/复现/TEST/test9_multibit_watermark.py"

print("=" * 60)
print("修复 device_map 配置")
print("=" * 60)
print()

# 读取文件
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换所有的 device_map="cpu" 为 device_map="auto"
original_content = content
content = content.replace('device_map="cpu"', 'device_map="auto"')

# 检查是否有修改
changes = content.count('device_map="auto"') - original_content.count('device_map="auto"')

if changes > 0:
    # 写回文件
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ 成功修改 {changes} 处 device_map 配置")
    print(f"   cpu -> auto")
    print()
    print("现在可以运行测试:")
    print("  python test9_multibit_watermark.py")
else:
    print("⚠️  未找到需要修改的配置")
    print("   文件可能已经是最新版本")

print()
print("=" * 60)
