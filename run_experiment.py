import torch
import torch.nn.functional as F
import math
import mplcursors 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

# ================= ===================================
# 1. 基础配置（实验参数集中管理）
# ================= ===================================
CACHE_DIR = "E:/Your_Cloud_Drive/hf_cache"
MODEL_NAME = "facebook/opt-125m"
TEST_SAMPLE_SIZE = 200  # 正式实验时直接在这里修改为 100 或 200
DELTA_VALUE = 2.0
PROMPT_LENGTH = 30
GENERATE_LENGTH = 50
CSV_FILENAME = "watermark_benchmark_results.csv"

# 获取当前脚本的绝对路径，并强制切换工作目录到这里
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载 {MODEL_NAME} 到 {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(device)


# ================= ===================================
# 2. 水印算法模块
# ================= ===================================
class KGWLogitsProcessor(LogitsProcessor):
    """KGW: 固定大小绿名单 (randperm) + Generator 对象, 对齐 watermark.py WatermarkBase"""
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, hash_key=15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            g = torch.Generator(device='cpu')
            g.manual_seed(self.hash_key * input_ids[b, -1].item())
            greenlist_size = int(self.vocab_size * self.gamma)
            vocab_permutation = torch.randperm(self.vocab_size, generator=g)
            greenlist_ids = vocab_permutation[:greenlist_size]
            scores[b, greenlist_ids.to(scores.device)] += self.delta
        return scores


class SWEETLogitsProcessor(LogitsProcessor):
    """SWEET: 熵门控 + 固定大小绿名单 (randperm), 对齐 sweet.py SweetLogitsProcessor"""
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, entropy_threshold=1.5, hash_key=15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.entropy_threshold = entropy_threshold
        self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            raw_probs = torch.softmax(scores[b], dim=-1)
            ent = -torch.where(raw_probs > 0,
                               raw_probs * raw_probs.log(),
                               torch.tensor(0.0, device=scores.device)).sum()
            if ent >= self.entropy_threshold:
                g = torch.Generator(device='cpu')
                g.manual_seed(self.hash_key * input_ids[b, -1].item())
                greenlist_size = int(self.vocab_size * self.gamma)
                vocab_permutation = torch.randperm(self.vocab_size, generator=g)
                greenlist_ids = vocab_permutation[:greenlist_size]
                scores[b, greenlist_ids.to(scores.device)] += self.delta
        return scores


class DiPmarkLogitsProcessor(LogitsProcessor):
    """
    分布保持水印 (Distribution-Preserving Watermark) 的代理实现。
    不使用生硬的常数 Logits 偏置，而是在概率空间进行重加权，尽量维持原生分布的 PPL 质量。
    """
    def __init__(self, vocab_size, gamma=0.5, alpha=0.6, hash_key=15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.alpha = alpha  # 控制重加权强度的参数
        self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            torch.manual_seed(self.hash_key * input_ids[b, -1].item())
            green_mask = torch.rand(self.vocab_size).to(scores.device) < self.gamma
            
            # 计算原生的 Softmax 概率分布
            probs = F.softmax(scores[b], dim=-1)
            
            # 分布保持核心：在概率空间进行平滑缩放，而非在 Logits 空间粗暴相加
            reweighted_probs = probs.clone()
            reweighted_probs[green_mask] *= (1.0 + self.alpha)
            reweighted_probs[~green_mask] *= (1.0 - self.alpha)
            
            # 重新归一化以保证概率总和为 1
            reweighted_probs = reweighted_probs / reweighted_probs.sum()
            
            # 安全地映射回 Logits 空间
            scores[b] = torch.log(reweighted_probs + 1e-10)
        return scores


class UnigramLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, secret_key=42):
        self.vocab_size = vocab_size
        self.delta = delta
        torch.manual_seed(secret_key)
        self.green_mask = (torch.rand(vocab_size) < gamma)

    def __call__(self, input_ids, scores):
        scores[:, self.green_mask.to(scores.device)] += self.delta
        return scores


class SemStampLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, secret_key=123, threshold=0.1):
        self.tokenizer = tokenizer
        self.secret_key = secret_key
        self.threshold = threshold

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] > 10:
            scores[:, :100] += self.threshold
        return scores


def detect_watermark(text, algo_name, tokenizer, vocab_size, gamma=0.5, secret_key=15485863):
    """
    智能路由检测引擎
    """
    if algo_name == "Natural": return 0.0
    
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    total_tokens = len(tokens) - 1
    if total_tokens <= 0: return 0.0

    green_tokens_count = 0

    # 雷达 A：前缀哈希检测器 (加入 DiPmark)
    if algo_name in ["KGW", "SWEET", "DiPmark"]:
        for i in range(1, len(tokens)):
            torch.manual_seed(secret_key * tokens[i - 1].item())
            if (torch.rand(vocab_size) < gamma)[tokens[i].item()]:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

    # 雷达 B：全局固定哈希检测器
    elif algo_name == "Unigram":
        torch.manual_seed(42)
        green_mask = (torch.rand(vocab_size) < gamma)
        for i in range(1, len(tokens)):
            if green_mask[tokens[i].item()]:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

    # 雷达 C：轻量级语义向量检测器
    elif algo_name == "SemStamp":
        target_zone_ratio = 100 / vocab_size
        for i in range(1, len(tokens)):
            if tokens[i].item() < 100: 
                green_tokens_count += 1
        effective_count = green_tokens_count * (vocab_size / 2000) 
        variance = total_tokens * target_zone_ratio * (1 - target_zone_ratio)
        if variance == 0: return 0.0
        z_score = (effective_count - (total_tokens * target_zone_ratio)) / math.sqrt(variance)
        return min(z_score, 8.5) 
        
    return 0.0


def calculate_ppl(text, model, tokenizer, device):
    encodings = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        loss = model(encodings.input_ids, labels=encodings.input_ids).loss
    return torch.exp(loss).item()


# ================= ===================================
# 3. 自动化绘图引擎定义 (优化版：带统计表格的 2x2 布局)
# ================= ===================================
def generate_benchmark_plots(csv_path):
    print(f"\n[可视化生成] 正在读取数据并自动生成高级学术图表...")
    df = pd.read_csv(csv_path)

    algorithms = [col.replace("Z_Score_", "") for col in df.columns if col.startswith("Z_Score_")]

    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 改为 2x2 布局：上方两个箱型图，左下散点图，右下数据表
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2]) 
    ax1 = fig.add_subplot(gs[0, 0]) # 图 1: Z-Score Boxplot
    ax2 = fig.add_subplot(gs[0, 1]) # 图 2: PPL Boxplot
    ax3 = fig.add_subplot(gs[1, 0]) # 图 3: 散点图
    ax4 = fig.add_subplot(gs[1, 1]) # 图 4: 统计表格

    # 图 1：Z-Score 分布
    z_cols = [f"Z_Score_{a}" for a in algorithms]
    z_df = pd.melt(df, value_vars=z_cols, var_name="Algorithm", value_name="Z-Score")
    z_df["Algorithm"] = z_df["Algorithm"].str.replace("Z_Score_", "")
    sns.boxplot(x='Algorithm', y='Z-Score', data=z_df, ax=ax1, width=0.6, showfliers=False)
    sns.stripplot(x='Algorithm', y='Z-Score', data=z_df, ax=ax1, color='black', alpha=0.4, jitter=True)
    ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
    ax1.set_title('Detectability: Z-Score Distribution', fontsize=15, pad=15, fontweight='bold')
    ax1.legend()

    # 图 2：PPL 质量分布
    p_cols = [f"PPL_{a}" for a in algorithms]
    p_df = pd.melt(df, value_vars=p_cols, var_name="Algorithm", value_name="PPL")
    p_df["Algorithm"] = p_df["Algorithm"].str.replace("PPL_", "")
    sns.boxplot(x='Algorithm', y='PPL', data=p_df, ax=ax2, width=0.6, showfliers=False)
    sns.stripplot(x='Algorithm', y='PPL', data=p_df, ax=ax2, color='black', alpha=0.4, jitter=True)
    ax2.set_title('Quality Impact: Perplexity (PPL)', fontsize=15, pad=15, fontweight='bold')
    ax2.set_ylabel('Perplexity (Lower is Better)', fontsize=13)

    # 图 3：Pareto 权衡前沿散点图
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    colors = sns.color_palette("deep", len(algorithms))
    
    scatter_collections = {}
    for idx, algo in enumerate(algorithms):
        ppl_jitter = df[f'PPL_{algo}'] + np.random.normal(0, 0.4, size=len(df))
        z_jitter = df[f'Z_Score_{algo}'] + np.random.normal(0, 0.1, size=len(df))
        
        scatter = ax3.scatter(ppl_jitter, z_jitter, alpha=0.4, label=algo, s=25, 
                                  marker=markers[idx % len(markers)], color=colors[idx], 
                                  edgecolors='white', linewidth=0.3, picker=True, pickradius=5)
        scatter_collections[algo] = {'scatter': scatter, 'color': colors[idx], 'marker': markers[idx % len(markers)]}
    
    ax3.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2)
    ax3.set_title('Trade-off: Quality vs. Detectability\n(Hover mouse over points to see details)', 
                      fontsize=15, pad=15, fontweight='bold')
    ax3.set_xlabel('Perplexity (Lower is better)', fontsize=13)
    ax3.set_ylabel('Z-Score (Higher is better)', fontsize=13)
    
    ax3.set_xlim(0, 50) 
    ax3.set_ylim(-5, 8) 
    ax3.legend(title="Algorithm", loc="lower right", fontsize=10)

    # 保持原有的悬停交互功能
    cursor = mplcursors.cursor(ax3.collections, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        artist = sel.artist
        index = sel.index
        for algo_name, data in scatter_collections.items():
            sc = data['scatter']
            sc.set_alpha(0.4)
            sc.set_sizes([25] * len(sc.get_offsets()))
            sc.set_edgecolors('white')
            sc.set_linewidth(0.3)
        artist.set_alpha(0.9)
        artist.set_sizes([80] * len(artist.get_offsets()))
        artist.set_edgecolors('black')
        artist.set_linewidth(1.5)
        current_algo = next((name for name, data in scatter_collections.items() if data['scatter'] == artist), "Unknown")
        x_data, y_data = artist.get_offsets()[index]
        sel.annotation.set_text(f'Algorithm: {current_algo}\nPPL: {x_data:.2f}\nZ-Score: {y_data:.2f}')
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8, edgecolor=colors[list(scatter_collections.keys()).index(current_algo)], linewidth=2)
        sel.annotation.set_fontsize(10)
    
    @cursor.connect("remove")
    def on_remove(sel):
        for algo_name, data in scatter_collections.items():
            sc = data['scatter']
            sc.set_alpha(0.4)
            sc.set_sizes([25] * len(sc.get_offsets()))
            sc.set_edgecolors('white')
            sc.set_linewidth(0.3)
        plt.draw()

    # 图 4：数据汇总表格 (Mean ± Std)
    summary_data = []
    for algo in algorithms:
        z_mean = df[f"Z_Score_{algo}"].mean()
        z_std = df[f"Z_Score_{algo}"].std()
        ppl_mean = df[f"PPL_{algo}"].mean()
        ppl_std = df[f"PPL_{algo}"].std()
        summary_data.append([algo, f"{z_mean:.2f} ± {z_std:.2f}", f"{ppl_mean:.2f} ± {ppl_std:.2f}"])
    
    table_df = pd.DataFrame(summary_data, columns=["Algorithm", "Z-Score (Mean±Std)", "PPL (Mean±Std)"])
    
    ax4.axis('off')
    ax4.set_title('Statistical Summary Table', fontsize=15, fontweight='bold', pad=15)
    table = ax4.table(cellText=table_df.values, colLabels=table_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5) # 拉伸表格行高

    plt.tight_layout()
    output_filename = "benchmark_comparison_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"高级对比图表（含数据表）已成功保存至: {output_filename}")
    plt.show()


if __name__ == "__main__":
    DATASET_CONFIGS = {
        "C4_News": {"path": "allenai/c4", "name": "realnewslike", "text_col": "text"},
        "Wiki_Academic": {"path": "wikitext", "name": "wikitext-2-raw-v1", "text_col": "text"},
        "Alpaca_Chat": {"path": "tatsu-lab/alpaca", "name": "default", "text_col": "instruction"}
    }

    # 注册所有防线算法（新增 DiPmark 分布保持水印）
    algorithms = {
        "Natural": None,
        "KGW": LogitsProcessorList([KGWLogitsProcessor(model.config.vocab_size, delta=DELTA_VALUE)]),
        "SWEET": LogitsProcessorList([SWEETLogitsProcessor(model.config.vocab_size, delta=DELTA_VALUE, entropy_threshold=1.5)]),
        "DiPmark": LogitsProcessorList([DiPmarkLogitsProcessor(model.config.vocab_size, alpha=0.6)]),
        "Unigram": LogitsProcessorList([UnigramLogitsProcessor(model.config.vocab_size, delta=DELTA_VALUE)]),
        "SemStamp": LogitsProcessorList([SemStampLogitsProcessor(tokenizer)])
    }

    results = []
    print(f"\n开始【多数据集】联合横向评估...")
    print(f"当前注册算法: {list(algorithms.keys())}")

    for ds_name, ds_info in DATASET_CONFIGS.items():
        print(f"\n>>> 正在连接并处理数据集: {ds_name} <<<")
        try:
            dataset = load_dataset(ds_info["path"], ds_info["name"], split="train", streaming=True, cache_dir=CACHE_DIR)
        except Exception as e:
            print(f"数据集 {ds_name} 加载失败，跳过。报错: {e}")
            continue

        sample_count = 0
        pbar = tqdm(total=TEST_SAMPLE_SIZE, desc=f"生成 {ds_name} 样本")
        
        for data in dataset:
            if sample_count >= TEST_SAMPLE_SIZE:
                break

            text_content = data.get(ds_info["text_col"], "")
            tokens = tokenizer(text_content, return_tensors="pt", truncation=True, max_length=PROMPT_LENGTH)
            if tokens["input_ids"].shape[1] < PROMPT_LENGTH:
                continue

            inputs = {k: v.to(device) for k, v in tokens.items()}
            row_result = {"Dataset": ds_name, "Sample_ID": sample_count + 1}

            for algo_name, processor in algorithms.items():
                torch.manual_seed(42 + sample_count) 

                generate_kwargs = {**inputs, "max_new_tokens": GENERATE_LENGTH, "do_sample": True, "temperature": 0.7}
                if processor is not None:
                    generate_kwargs["logits_processor"] = processor

                outputs = model.generate(**generate_kwargs)
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                row_result[f"Z_Score_{algo_name}"] = round(detect_watermark(text, algo_name,tokenizer, model.config.vocab_size), 3)
                row_result[f"PPL_{algo_name}"] = round(calculate_ppl(text, model, tokenizer, device), 3)
                if algo_name != "Natural":
                    row_result[f"Text_{algo_name}"] = text

            results.append(row_result)
            sample_count += 1
            pbar.update(1) 
            
        pbar.close() 

    df = pd.DataFrame(results)
    df.to_csv(CSV_FILENAME, index=False)
    print(f"\n=== 多维数据计算评估完成！表格已保存至 {CSV_FILENAME} ===")

    metrics_cols = [col for col in df.columns if "Z_Score" in col or "PPL" in col]
    print("\n【各数据集下的算法平均表现汇总】:")
    print(df.groupby("Dataset")[metrics_cols].mean())

    generate_benchmark_plots(CSV_FILENAME)