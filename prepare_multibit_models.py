#!/usr/bin/env python3
"""
模型准备脚本：为test9创建简化版模型
使用facebook/opt-2.7b作为model0和model1的替代
"""
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def prepare_fp16_models():
    print("=" * 60)
    print("准备Multi-bit水印测试所需的模型")
    print("=" * 60)

    # 设置HuggingFace镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    FP16_DIR = os.path.join(os.path.expanduser("~"), ".cache", "watermark_fp16")
    MODEL0_PATH = os.path.join(FP16_DIR, "model0")
    MODEL1_PATH = os.path.join(FP16_DIR, "model1")

    os.makedirs(FP16_DIR, exist_ok=True)

    # 检查是否已存在
    if os.path.exists(MODEL0_PATH) and os.path.exists(MODEL1_PATH):
        print("模型已存在，跳过下载")
        return

    print("\n使用facebook/opt-2.7b作为双模型替代")
    print("这是简化版实现，用于快速测试")
    print("使用HF镜像: https://hf-mirror.com\n")

    BASE_MODEL = "facebook/opt-2.7b"
    CACHE_DIR = "/root/autodl-tmp/hf_cache"

    # 下载并转换为FP16
    print(f"下载模型: {BASE_MODEL}")
    print("这可能需要5-15分钟，请耐心等待...\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    print(f"\n保存model0到: {MODEL0_PATH}")
    model.save_pretrained(MODEL0_PATH)
    tokenizer.save_pretrained(MODEL0_PATH)

    print(f"保存model1到: {MODEL1_PATH}")
    model.save_pretrained(MODEL1_PATH)
    tokenizer.save_pretrained(MODEL1_PATH)

    print("\n✅ 模型准备完成！")
    print(f"   model0: {MODEL0_PATH}")
    print(f"   model1: {MODEL1_PATH}")
    print("\n注意：这是简化版，使用相同的基础模型")
    print("      完整版需要专门训练的双模型释义器")

if __name__ == "__main__":
    try:
        prepare_fp16_models()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n如果下载失败，请检查网络或使用镜像源")
        sys.exit(1)
