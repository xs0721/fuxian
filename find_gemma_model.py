#!/usr/bin/env python3
"""查找并验证服务器上的 Gemma 模型"""
import os
import glob
from pathlib import Path

print("🔍 正在查找 Gemma 模型...")
print()

# 可能的位置
search_paths = [
    "/root/autodl-tmp/gemma-2-2b-it",
    "/root/autodl-tmp/hf_cache/models--google--gemma-2-2b-it",
    "/root/.cache/huggingface/hub/models--google--gemma-2-2b-it",
    os.path.expanduser("~/autodl-tmp/gemma-2-2b-it"),
    os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-2b-it"),
]

found_models = []

print("=== 检查已知位置 ===")
for path in search_paths:
    if os.path.exists(path):
        print(f"✅ 找到: {path}")

        # 检查是否是有效的模型目录
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            print(f"   ✓ 包含 config.json")
            found_models.append(path)
        else:
            # 检查是否是 HF 缓存格式 (有 snapshots 目录)
            snapshots_dir = os.path.join(path, "snapshots")
            if os.path.exists(snapshots_dir):
                print(f"   → HuggingFace 缓存格式，检查 snapshots...")
                snapshots = sorted(os.listdir(snapshots_dir))
                if snapshots:
                    latest = snapshots[-1]
                    snapshot_path = os.path.join(snapshots_dir, latest)
                    snapshot_config = os.path.join(snapshot_path, "config.json")
                    if os.path.exists(snapshot_config):
                        print(f"   ✓ 最新 snapshot: {latest}")
                        print(f"   ✓ 完整路径: {snapshot_path}")
                        found_models.append(snapshot_path)
        print()
    else:
        print(f"❌ 不存在: {path}")

# 搜索所有可能的 gemma 目录
print("\n=== 搜索所有 gemma 相关目录 ===")
search_roots = ["/root/autodl-tmp", "/root/.cache", os.path.expanduser("~")]
for root in search_roots:
    if os.path.exists(root):
        print(f"搜索: {root}")
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                if 'gemma' in dirpath.lower():
                    if 'config.json' in filenames:
                        print(f"  ✅ {dirpath}")
                        if dirpath not in found_models:
                            found_models.append(dirpath)
                # 限制搜索深度
                if dirpath.count(os.sep) - root.count(os.sep) > 5:
                    dirnames[:] = []
        except PermissionError:
            pass

print("\n" + "="*60)
if found_models:
    print(f"✨ 找到 {len(found_models)} 个 Gemma 模型:")
    print()
    for i, model_path in enumerate(found_models, 1):
        print(f"{i}. {model_path}")

        # 尝试加载验证
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            print(f"   ✓ 模型类型: {config.model_type}")
            print(f"   ✓ 隐藏层大小: {config.hidden_size}")
        except Exception as e:
            print(f"   ⚠️  加载失败: {e}")
        print()

    print("="*60)
    print("\n📝 修复方法：")
    print("\n在 test_common.py 中找到这几行:")
    print("=" * 40)
    print('_local_gemma = _os.path.expanduser("~/autodl-tmp/gemma-2-2b-it")')
    print('if _os.path.exists(_local_gemma):')
    print('    GEMMA_DIR = _local_gemma')
    print("=" * 40)
    print("\n将第一行改为:")
    print(f'_local_gemma = "{found_models[0]}"')
    print()
else:
    print("❌ 未找到 Gemma 模型")
    print()
    print("💡 可能的原因:")
    print("1. 模型还没有下载")
    print("2. 模型在其他位置")
    print("3. 模型名称不同")
    print()
    print("建议:")
    print("1. 手动运行: ls -lh /root/autodl-tmp/")
    print("2. 或运行: ls -lh ~/.cache/huggingface/hub/")
    print("3. 如果模型在其他位置，告诉我路径，我来修改代码")
