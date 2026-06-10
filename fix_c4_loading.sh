#!/bin/bash
# 修复 C4 数据集加载问题

cd /root/复现/TEST || exit 1

echo "开始修复 run_experiment.py ..."

# 备份
cp run_experiment.py run_experiment.py.before_c4_fix

# 创建补丁文件
cat > /tmp/c4_fix.py << 'PYEOF'
import sys

with open('run_experiment.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到需要修改的位置
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]

    # 找到 "for ds_name, ds_info in DATASET_CONFIGS.items():"
    if 'for ds_name, ds_info in DATASET_CONFIGS.items():' in line:
        new_lines.append(line)
        i += 1

        # 跳过旧的加载逻辑，插入新逻辑
        indent = '        '
        new_lines.append(f'{indent}print(f"\\n>>> 正在连接并处理数据集: {{ds_name}} <<<")\n')
        new_lines.append(f'{indent}try:\n')
        new_lines.append(f'{indent}    # 特殊处理：C4数据集直接从本地文件加载\n')
        new_lines.append(f'{indent}    if ds_name == "C4_News":\n')
        new_lines.append(f'{indent}        import json\n')
        new_lines.append(f'{indent}        c4_file = os.path.join(CACHE_DIR, "datasets--allenai--c4", "realnewslike-train.jsonl")\n')
        new_lines.append(f'{indent}        if os.path.exists(c4_file):\n')
        new_lines.append(f'{indent}            print(f"  从本地文件加载C4: {{c4_file}}")\n')
        new_lines.append(f'{indent}            c4_data = []\n')
        new_lines.append(f'{indent}            with open(c4_file, \'r\', encoding=\'utf-8\') as f:\n')
        new_lines.append(f'{indent}                for line in f:\n')
        new_lines.append(f'{indent}                    item = json.loads(line)\n')
        new_lines.append(f'{indent}                    c4_data.append({{"text": item["text"]}})\n')
        new_lines.append(f'{indent}            dataset = iter(c4_data)\n')
        new_lines.append(f'{indent}            print(f"  ✓ C4数据集加载成功: {{len(c4_data)}} 条记录")\n')
        new_lines.append(f'{indent}        else:\n')
        new_lines.append(f'{indent}            print(f"  ✗ C4文件不存在: {{c4_file}}")\n')
        new_lines.append(f'{indent}            continue\n')
        new_lines.append(f'{indent}    else:\n')
        new_lines.append(f'{indent}        # 其他数据集使用正常加载\n')
        new_lines.append(f'{indent}        from datasets import DownloadMode\n')
        new_lines.append(f'{indent}        dataset = load_dataset(\n')
        new_lines.append(f'{indent}            ds_info["path"],\n')
        new_lines.append(f'{indent}            ds_info["name"],\n')
        new_lines.append(f'{indent}            split="train",\n')
        new_lines.append(f'{indent}            streaming=False,\n')
        new_lines.append(f'{indent}            cache_dir=CACHE_DIR,\n')
        new_lines.append(f'{indent}            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,\n')
        new_lines.append(f'{indent}        )\n')
        new_lines.append(f'{indent}        dataset = iter(dataset)\n')

        # 跳过原来的代码直到 except
        while i < len(lines) and 'except Exception as e:' not in lines[i]:
            i += 1
        continue

    new_lines.append(line)
    i += 1

with open('run_experiment.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✓ 修复完成！")
PYEOF

# 执行补丁
python /tmp/c4_fix.py

echo ""
echo "✓ run_experiment.py 已修复"
echo ""
echo "验证修改:"
grep -A20 "for ds_name, ds_info in DATASET_CONFIGS" run_experiment.py | head -25
echo ""
echo "现在可以运行: nohup python run_experiment.py > run_experiment.log 2>&1 &"
