import re
import glob
import os

# 找到所有测试脚本（排除 test_common.py 和 fixed 版本）
test_files = sorted([f for f in glob.glob('test[0-9]*.py') 
                     if 'common' not in f and 'fixed' not in f])

print(f"找到 {len(test_files)} 个测试脚本\n")

for file in test_files:
    print(f"处理: {file}")
    
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取测试编号
    match = re.search(r'test(\d+)_', file)
    if not match:
        print(f"  ⏭️  跳过（无法提取编号）\n")
        continue
    
    test_num = match.group(1)
    
    # 备份原文件
    backup_file = f"{file}.bak"
    if not os.path.exists(backup_file):
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # 查找绘图代码块的模式
    # 通常以创建 DataFrame 开始，以 plt.close() 或 print("=== 测试X完成 ===") 结束
    
    # 模式1: 查找 pd.DataFrame(xxx_results) ... plt.close()
    pattern1 = r'(\w+_df = pd\.DataFrame\(\w+_results\))\s*\n\s*if not \w+_df\.empty:.*?plt\.close\(\)'
    
    # 模式2: 查找包含 sns.boxplot 的整个绘图块
    pattern2 = r'(if not \w+_df\.empty:)\s*\n(.*?)(plt\.close\(\)|print\("=== 测试\d+完成 ==="\))'
    
    modified = False
    
    # 尝试查找并提取关键信息
    if 'sns.boxplot' in content or 'plt.figure' in content:
        # 找到 DataFrame 名称
        df_match = re.search(r'(\w+)_df = pd\.DataFrame\((\w+)_results\)', content)
        if df_match:
            df_name = df_match.group(1) + '_df'
            
            # 找到测试名称（通常在 set_title 中）
            title_match = re.search(r"set_title\(['\"]Test \d+:?\s*([^'\"]+)['\"]", content)
            test_name = title_match.group(1) if title_match else "Attack Test"
            
            # 找到输出文件名
            output_match = re.search(r'savefig\(["\']([^"\']+\.png)["\']', content)
            output_file = output_match.group(1) if output_match else f"attack_{test_num}_result.png"
            
            # 找到指标名称
            metric = "Z_Score"  # 默认
            if 'TPR' in content:
                metric = "TPR"
            elif 'Perplexity' in content:
                metric = "Perplexity"
            
            # 构造新的绘图调用
            new_plot_code = f'''
{df_name} = pd.DataFrame({df_match.group(2)}_results)
if not {df_name}.empty:
    # 使用通用绘图函数
    plot_attack_results(
        df={df_name},
        test_name="{test_name}",
        test_number={test_num},
        output_filename="{output_file}",
        metric="{metric}",
        threshold=4.0
    )

print("=== 测试{test_num}完成 ===\\n")'''
            
            print(f"  📊 找到绘图代码")
            print(f"     - DataFrame: {df_name}")
            print(f"     - 测试名称: {test_name}")
            print(f"     - 输出文件: {output_file}")
            print(f"     - 指标: {metric}")
            print(f"  ⚠️  需要手动检查和替换")
    
    print()

print("\n由于每个测试的绘图代码结构不同，建议逐个手动替换。")
print("我可以帮你逐个修改，请告诉我先从哪个测试开始？")
