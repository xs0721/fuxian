#!/usr/bin/env python3
"""
优化版测试脚本 - model1 共享 model0（节省内存）
使用方法: python optimize_test9.py
"""

FILE_PATH = "/root/复现/TEST/test9_multibit_watermark.py"

print("=" * 60)
print("优化 test9 - 让 model1 共享 model0（节省内存）")
print("=" * 60)
print()

# 读取文件
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# 备份
with open(FILE_PATH + '.original', 'w', encoding='utf-8') as f:
    f.write(content)

# 找到 model1 加载部分并替换为共享 model0
old_code = '''    # ── Model1: CPU fp16 (最后加载,避免与其他模型争CPU RAM) ──
    print("加载 model1 (CPU fp16)...")
    from transformers import PreTrainedModel

    # 兼容不同版本的 transformers
    _orig_init_missing = None
    if hasattr(PreTrainedModel, '_initialize_missing_keys'):
        _orig_init_missing = PreTrainedModel._initialize_missing_keys
        PreTrainedModel._initialize_missing_keys = lambda self, is_quantized: None

    try:
        _actor_model1 = AutoModelForCausalLM.from_pretrained(
            MODEL1_PATH, torch_dtype=torch.float16,
            device_map="cpu", low_cpu_mem_usage=True,
        ).eval()
    finally:
        if _orig_init_missing is not None:
            PreTrainedModel._initialize_missing_keys = _orig_init_missing
    print("  model1 加载完成")'''

new_code = '''    # ── Model1: 共享 model0 (节省内存) ──
    print("model1 共享 model0 (节省内存)...")
    _actor_model1 = _actor_model0  # 使用相同的模型
    print("  model1 已设置 (共享)")'''

if old_code in content:
    content = content.replace(old_code, new_code)

    # 写回文件
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ 优化完成！")
    print()
    print("修改内容:")
    print("  - model1 现在共享 model0，不再单独加载")
    print("  - 节省约 5GB 内存")
    print("  - 备份已保存到: test9_multibit_watermark.py.original")
    print()
    print("注意: 这是简化版本，两个模型相同")
    print("      适合测试框架验证，结果可能与完整版不同")
else:
    print("⚠️  未找到需要修改的代码")
    print("   文件可能已经被修改")

print()
print("现在运行测试:")
print("  python test9_multibit_watermark.py")
print()
print("=" * 60)
