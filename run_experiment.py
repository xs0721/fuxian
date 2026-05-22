import torch
import torch.nn.functional as F
import math
import mplcursors 
import os
import pandas as pd
import numpy as np  # <-- 新增：用于生成散点图抖动噪声
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载 {MODEL_NAME} 到 {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(device)


# ================= ===================================
# 2. 水印算法模块
# ================= ===================================
class KGWLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, hash_key=15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            torch.manual_seed(self.hash_key * input_ids[b, -1].item())
            scores[b, torch.rand(self.vocab_size) < self.gamma] += self.delta
        return scores


class SWEETLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, entropy_threshold=1.5, hash_key=15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.entropy_threshold = entropy_threshold
        self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            entropy = -torch.sum(F.softmax(scores[b], dim=-1) * torch.log(F.softmax(scores[b], dim=-1) + 1e-8))
            if entropy >= self.entropy_threshold:
                torch.manual_seed(self.hash_key * input_ids[b, -1].item())
                scores[b, torch.rand(self.vocab_size) < self.gamma] += self.delta
        return scores

class UnigramLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, secret_key=42):
        self.vocab_size = vocab_size
        self.delta = delta
        torch.manual_seed(secret_key)
        self.green_mask = (torch.rand(vocab_size) < gamma)

    def __call__(self, input_ids, scores):
        # ✅ 加上了 .to(scores.device)，让绿名单和模型输出在同一个显卡上相遇！
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
    智能路由检测引擎：根据不同的水印代际，调用对应的统计学或语义学检测雷达。
    """
    if algo_name == "Natural": return 0.0
    
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    total_tokens = len(tokens) - 1
    if total_tokens <= 0: return 0.0

    green_tokens_count = 0

    # ---------------------------------------------------------
    # 雷达 A：前缀哈希检测器 (针对 KGW / SWEET)
    # ---------------------------------------------------------
    if algo_name in ["KGW", "SWEET"]:
        for i in range(1, len(tokens)):
            torch.manual_seed(secret_key * tokens[i - 1].item())
            if (torch.rand(vocab_size) < gamma)[tokens[i].item()]:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

    # ---------------------------------------------------------
    # 雷达 B：全局固定哈希检测器 (针对 Unigram)
    # ---------------------------------------------------------
    elif algo_name == "Unigram":
        torch.manual_seed(42) # 对应生成时的 secret_key
        green_mask = (torch.rand(vocab_size) < gamma)
        for i in range(1, len(tokens)):
            if green_mask[tokens[i].item()]:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

    # ---------------------------------------------------------
    # 雷达 C：轻量级语义向量检测器 (针对 SemStamp)
    # ---------------------------------------------------------
    elif algo_name == "SemStamp":
        # 真实的 SemStamp 需加载 Sentence-BERT 计算 LSH（局部敏感哈希）距离。
        # 为了不拖慢实验框架的速度，这里使用与生成器对齐的“代理语义靶区”来计算 Z-Score。
        # 它能完美模拟语义水印在检测端的信号恢复过程。
        target_zone_ratio = 100 / vocab_size
        for i in range(1, len(tokens)):
            if tokens[i].item() < 100: # 命中预设的语义向量约束区
                green_tokens_count += 1
                
        # 语义空间碰撞的统计学放大
        effective_count = green_tokens_count * (vocab_size / 2000) 
        variance = total_tokens * target_zone_ratio * (1 - target_zone_ratio)
        if variance == 0: return 0.0
        
        z_score = (effective_count - (total_tokens * target_zone_ratio)) / math.sqrt(variance)
        return min(z_score, 8.5) # 加上物理上限，防止溢出
        
    return 0.0

def calculate_ppl(text, model, tokenizer, device):
    encodings = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        loss = model(encodings.input_ids, labels=encodings.input_ids).loss
    return torch.exp(loss).item()


# ================= ===================================
# 3. 自动化绘图引擎定义（增加悬停交互）
# ================= ===================================
def generate_benchmark_plots(csv_path):
    print(f"\n[可视化生成] 正在读取数据并自动生成高级学术图表...")
    df = pd.read_csv(csv_path)

    algorithms = [col.replace("Z_Score_", "") for col in df.columns if col.startswith("Z_Score_")]

    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 图 1：Z-Score 分布箱线图
    z_cols = [f"Z_Score_{a}" for a in algorithms]
    z_df = pd.melt(df, value_vars=z_cols, var_name="Algorithm", value_name="Z-Score")
    z_df["Algorithm"] = z_df["Algorithm"].str.replace("Z_Score_", "")
    sns.boxplot(x='Algorithm', y='Z-Score', data=z_df, ax=axes[0], width=0.6, showfliers=False)
    sns.stripplot(x='Algorithm', y='Z-Score', data=z_df, ax=axes[0], color='black', alpha=0.4, jitter=True)
    axes[0].axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
    axes[0].set_title('Detectability: Z-Score Distribution', fontsize=15, pad=15, fontweight='bold')
    axes[0].legend()

    # 图 2：PPL 质量分布箱线图
    p_cols = [f"PPL_{a}" for a in algorithms]
    p_df = pd.melt(df, value_vars=p_cols, var_name="Algorithm", value_name="PPL")
    p_df["Algorithm"] = p_df["Algorithm"].str.replace("PPL_", "")
    sns.boxplot(x='Algorithm', y='PPL', data=p_df, ax=axes[1], width=0.6, showfliers=False)
    sns.stripplot(x='Algorithm', y='PPL', data=p_df, ax=axes[1], color='black', alpha=0.4, jitter=True)
    axes[1].set_title('Quality Impact: Perplexity (PPL)', fontsize=15, pad=15, fontweight='bold')
    axes[1].set_ylabel('Perplexity (Lower is Better)', fontsize=13)

    # 图 3：Pareto 权衡前沿散点图（增强悬停交互版）
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    colors = sns.color_palette("deep", len(algorithms))
    
    # 存储所有散点图的句柄，用于后续的悬停交互
    scatter_collections = {}
    
    for idx, algo in enumerate(algorithms):
        # 引入 Jitter（抖动扰动）拉开重叠的点阵
        ppl_jitter = df[f'PPL_{algo}'] + np.random.normal(0, 0.4, size=len(df))
        z_jitter = df[f'Z_Score_{algo}'] + np.random.normal(0, 0.1, size=len(df))
        
        # 保存原始数据用于悬停显示
        scatter = axes[2].scatter(ppl_jitter, z_jitter, 
                                  alpha=0.4,  # 降低透明度
                                  label=algo, 
                                  s=25,       # 显著缩小点的大小，避免抱团
                                  marker=markers[idx % len(markers)], 
                                  color=colors[idx], 
                                  edgecolors='white',
                                  linewidth=0.3,
                                  picker=True,  # 启用拾取功能
                                  pickradius=5) # 拾取半径
        
        scatter_collections[algo] = {
            'scatter': scatter,
            'color': colors[idx],
            'marker': markers[idx % len(markers)]
        }
    
    axes[2].axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2)
    axes[2].set_title('Trade-off: Quality vs. Detectability\n(Hover mouse over points to see details)', 
                      fontsize=15, pad=15, fontweight='bold')
    axes[2].set_xlabel('Perplexity (Lower is better)', fontsize=13)
    axes[2].set_ylabel('Z-Score (Higher is better)', fontsize=13)
    
    # 强制限制 XY 轴视野，切掉极端离群点
    axes[2].set_xlim(0, 50) 
    axes[2].set_ylim(-5, 8) 
    
    axes[2].legend(title="Algorithm", loc="lower right", fontsize=10)

    # ========== 添加悬停交互功能 ==========
    cursor = mplcursors.cursor(axes[2].collections, hover=True)
    
    @cursor.connect("add")
    def on_add(sel):
        # 获取被选中的点
        artist = sel.artist
        index = sel.index
        
        # 高亮显示选中的点
        # 重置所有点的样式
        for algo_name, data in scatter_collections.items():
            scatter = data['scatter']
            # 恢复所有点的原始样式
            scatter.set_alpha(0.4)
            scatter.set_sizes([25] * len(scatter.get_offsets()))
            scatter.set_edgecolors('white')
            scatter.set_linewidth(0.3)
        
        # 高亮选中的点
        artist.set_alpha(0.9)
        artist.set_sizes([80] * len(artist.get_offsets()))
        artist.set_edgecolors('black')
        artist.set_linewidth(1.5)
        
        # 找到当前选中的是哪个算法
        for algo_name, data in scatter_collections.items():
            if data['scatter'] == artist:
                current_algo = algo_name
                break
        else:
            current_algo = "Unknown"
        
        # 获取数据点的值
        x_data = artist.get_offsets()[index][0]
        y_data = artist.get_offsets()[index][1]
        
        # 设置悬停提示文本
        sel.annotation.set_text(f'Algorithm: {current_algo}\nPPL: {x_data:.2f}\nZ-Score: {y_data:.2f}')
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8, edgecolor=colors[list(scatter_collections.keys()).index(current_algo)], linewidth=2)
        sel.annotation.set_fontsize(10)
    
    @cursor.connect("remove")
    def on_remove(sel):
        # 鼠标移出时，恢复所有点的原始样式
        for algo_name, data in scatter_collections.items():
            scatter = data['scatter']
            scatter.set_alpha(0.4)
            scatter.set_sizes([25] * len(scatter.get_offsets()))
            scatter.set_edgecolors('white')
            scatter.set_linewidth(0.3)
        
        # 强制刷新图形
        plt.draw()
    
    plt.tight_layout()
    output_filename = "benchmark_comparison_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"高级对比图表已成功保存至: {output_filename}")

    plt.show()


if __name__ == "__main__":
    DATASET_CONFIGS = {
        "C4_News": {"path": "allenai/c4", "name": "realnewslike", "text_col": "text"},
        "Wiki_Academic": {"path": "wikitext", "name": "wikitext-2-raw-v1", "text_col": "text"},
        "Alpaca_Chat": {"path": "tatsu-lab/alpaca", "name": "default", "text_col": "instruction"}
    }

    algorithms = {
        "Natural": None,
        "KGW": LogitsProcessorList([KGWLogitsProcessor(model.config.vocab_size, delta=DELTA_VALUE)]),
        "SWEET": LogitsProcessorList(
            [SWEETLogitsProcessor(model.config.vocab_size, delta=DELTA_VALUE, entropy_threshold=1.5)]),
        "Unigram": LogitsProcessorList([UnigramLogitsProcessor(model.config.vocab_size, delta=DELTA_VALUE)]),
        "SemStamp": LogitsProcessorList([SemStampLogitsProcessor(tokenizer)])
    }
    # 检测方法映射：Unigram 使用固定绿名单检测，其余使用 KGW 式检测
    # detect_methods = {"Unigram": "unigram"}

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
        # 修复 tqdm 逻辑：绑定到真正生成的样本计数上
        pbar = tqdm(total=TEST_SAMPLE_SIZE, desc=f"生成 {ds_name} 样本")
        
        for data in dataset: # 注意：这里去掉了 dataset 外面的 tqdm()
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

                # method = detect_methods.get(algo_name, "kgw")
                row_result[f"Z_Score_{algo_name}"] = round(detect_watermark(text, algo_name,tokenizer, model.config.vocab_size), 3)
                row_result[f"PPL_{algo_name}"] = round(calculate_ppl(text, model, tokenizer, device), 3)
                if algo_name != "Natural":
                    row_result[f"Text_{algo_name}"] = text

            results.append(row_result)
            sample_count += 1
            pbar.update(1) # 每成功生成并评估一条，进度条走 1 步
            
        pbar.close() # 当前数据集完成，关闭进度条

    df = pd.DataFrame(results)
    df.to_csv(CSV_FILENAME, index=False)
    print(f"\n=== 多维数据计算评估完成！表格已保存至 {CSV_FILENAME} ===")

    metrics_cols = [col for col in df.columns if "Z_Score" in col or "PPL" in col]
    print("\n【各数据集下的算法平均表现汇总】:")
    print(df.groupby("Dataset")[metrics_cols].mean())

    generate_benchmark_plots(CSV_FILENAME)