#!/usr/bin/env python3
"""
快速修复 C4 加载问题
在服务器上直接运行: python fix_c4_quick.py
"""
import os
import re

print("=" * 60)
print("修复 C4 数据集加载")
print("=" * 60)

file_path = '/root/复现/TEST/run_experiment.py'

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 定义新的数据集加载代码
new_loading_code = '''    for ds_name, ds_info in DATASET_CONFIGS.items():
        print(f"\\n>>> 正在连接并处理数据集: {ds_name} <<<")
        try:
            # 特殊处理：C4数据集直接从本地文件加载
            if ds_name == "C4_News":
                import json
                c4_file = os.path.join(CACHE_DIR, "datasets--allenai--c4", "realnewslike-train.jsonl")
                if os.path.exists(c4_file):
                    print(f"  从本地文件加载C4: {c4_file}")
                    c4_data = []
                    with open(c4_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            item = json.loads(line)
                            c4_data.append({"text": item["text"]})
                    dataset = iter(c4_data)
                    print(f"  ✓ C4数据集加载成功: {len(c4_data)} 条记录")
                else:
                    print(f"  ✗ C4文件不存在: {c4_file}")
                    continue
            else:
                # 其他数据集使用正常加载
                from datasets import DownloadMode
                dataset = load_dataset(
                    ds_info["path"],
                    ds_info["name"],
                    split="train",
                    streaming=False,
                    cache_dir=CACHE_DIR,
                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                )
                dataset = iter(dataset)
        except Exception as e:
            print(f"数据集 {ds_name} 加载失败，跳过。报错: {e}")
            continue'''

# 查找并替换旧代码
old_pattern = r'    for ds_name, ds_info in DATASET_CONFIGS\.items\(\):.*?except Exception as e:\s+print\(f"数据集 \{ds_name\} 加载失败，跳过。报错: \{e\}"\)\s+continue'

# 使用 DOTALL 模式匹配多行
content_new = re.sub(old_pattern, new_loading_code, content, flags=re.DOTALL)

if content_new != content:
    # 备份原文件
    backup_path = file_path + '.backup_c4'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ 已备份原文件: {backup_path}")

    # 写入新内容
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content_new)
    print(f"✓ 已修复 {file_path}")
else:
    print("⚠ 未找到需要替换的代码，可能已经修复过了")

print("\n" + "=" * 60)
print("修复完成！")
print("=" * 60)
print("\n现在运行:")
print("  nohup python run_experiment.py > run_experiment.log 2>&1 &")
