import re
import glob

# 找到所有测试脚本（排除 test_common.py）
test_files = sorted([f for f in glob.glob('test[0-9]*.py') if 'common' not in f])

print(f"找到 {len(test_files)} 个测试脚本")

for file in test_files:
    print(f"\n处理: {file}")
    
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取测试编号
    match = re.search(r'test(\d+)_', file)
    if not match:
        print(f"  ⏭️  跳过（无法提取编号）")
        continue
    
    test_num = match.group(1)
    print(f"  测试编号: {test_num}")
    
    # 查找是否有绘图代码（包含 plt.figure 或 sns.boxplot）
    if 'plt.figure' not in content and 'sns.boxplot' not in content:
        print(f"  ⏭️  跳过（无绘图代码）")
        continue
    
    # TODO: 每个测试的绘图逻辑不同，需要逐个手动替换
    print(f"  ✅ 需要修改")

print("\n请告诉我从哪个测试开始修改，我逐个帮你替换")
