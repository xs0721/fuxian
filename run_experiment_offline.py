import pyarrow  # 必须最先导入, 避免与Anaconda的pyarrow DLL冲突
import os
import sys

# ================= ===================================
# 离线模式配置（使用本地缓存的数据集）
# ================= ===================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
import math
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

# ================= ===================================
# 1. 基础配置（离线模式）
# ================= ===================================
if os.name == 'nt':  # Windows
    CACHE_DIR = "E:/Your_Cloud_Drive/hf_cache"
else:  # Linux/Mac
    CACHE_DIR = os.path.expanduser("~/autodl-tmp/hf_cache") if os.path.exists(os.path.expanduser("~/autodl-tmp")) else "./hf_cache"

os.makedirs(CACHE_DIR, exist_ok=True)
MODEL_NAME = "facebook/opt-125m"
TEST_SAMPLE_SIZE = 200  # 每个数据集200个样本
DELTA_VALUE = 2.0
PROMPT_LENGTH = 30
GENERATE_LENGTH = 50
CSV_FILENAME = "watermark_benchmark_results.csv"

# 获取当前脚本的绝对路径，并强制切换工作目录到这里
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载 {MODEL_NAME} 到 {device}...")
print(f"离线模式: HF_DATASETS_OFFLINE={os.environ['HF_DATASETS_OFFLINE']}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True
).to(device)

print(f"✓ 模型加载成功！")
print(f"✓ 词表大小: {model.config.vocab_size}")
print(f"✓ 设备: {device}")
print()

# ================= ===================================
# 2. 使用本地文本数据而非在线数据集
# ================= ===================================
SYNTHETIC_PROMPTS = [
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is transforming the world",
    "Climate change poses significant challenges",
    "Machine learning models can learn from data",
    "The history of computing dates back to",
    "Natural language processing enables computers to",
    "Deep learning has revolutionized computer vision",
    "Quantum computing promises to solve complex problems",
    "Renewable energy sources are essential for",
    "The future of technology depends on innovation"
]

def generate_synthetic_dataset(name, size):
    """生成合成数据集（用于离线测试）"""
    print(f"[离线模式] 生成合成数据集: {name}")
    dataset = []
    for i in range(size):
        # 循环使用预定义的提示词
        prompt = SYNTHETIC_PROMPTS[i % len(SYNTHETIC_PROMPTS)]
        # 添加变化以增加多样性
        if i > len(SYNTHETIC_PROMPTS):
            prompt = f"{prompt} and furthermore"
        dataset.append({"text": prompt, "id": i})
    return dataset

# 这是复现代码的其余部分，从run_experiment.py的第60行开始复制
# ================= ===================================
# 水印算法模块
# ================= ===================================
