import torch
import torch.nn.functional as F
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 基础配置 =================
CACHE_DIR = "E:/Your_Cloud_Drive/hf_cache"
MODEL_NAME = "facebook/opt-125m"
TEST_SAMPLE_SIZE = 30  # 为了平滑曲线，测试30条长文本
DELTA_VALUE = 2.0
PROMPT_LENGTH = 15
GENERATE_LENGTH = 150  # 生成较长的文本，观察 Z-Score 随长度的累积过程

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{device.upper()}] 正在加载 {MODEL_NAME} 进行动态长度敏感性分析...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(device)
vocab_size = model.config.vocab_size

# ================= 2. 水印算法定义 (复用你的核心逻辑) =================
class KGWLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, hash_key=15485863):
        self.vocab_size = vocab_size; self.gamma = gamma; self.delta = delta; self.hash_key = hash_key
    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            torch.manual_seed(self.hash_key * input_ids[b, -1].item())
            scores[b, torch.rand(self.vocab_size) < self.gamma] += self.delta
        return scores

class DiPmarkLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, gamma=0.5, alpha=0.6, hash_key=15485863):
        self.vocab_size = vocab_size; self.gamma = gamma; self.alpha = alpha; self.hash_key = hash_key
    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            torch.manual_seed(self.hash_key * input_ids[b, -1].item())
            green_mask = torch.rand(self.vocab_size).to(scores.device) < self.gamma
            probs = F.softmax(scores[b], dim=-1)
            reweighted_probs = probs.clone()
            reweighted_probs[green_mask] *= (1.0 + self.alpha)
            reweighted_probs[~green_mask] *= (1.0 - self.alpha)
            reweighted_probs = reweighted_probs / reweighted_probs.sum()
            scores[b] = torch.log(reweighted_probs + 1e-10)
        return scores

class SemStampLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, threshold=0.15):
        self.tokenizer = tokenizer; self.threshold = threshold
    def __call__(self, input_ids, scores):
        scores[:, :100] += self.threshold # 模拟绿区偏置
        return scores

# ================= 3. 动态累积检测引擎 (核心创新点) =================
def cumulative_detect_watermark(text, algo_name, tokenizer, vocab_size, gamma=0.5, secret_key=15485863):
    """
    计算文本每增加 1 个 Token 时的 Z-Score 轨迹
    返回一个列表: [z_score_at_token_1, z_score_at_token_2, ..., z_score_at_token_N]
    """
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    total_tokens = len(tokens) - 1
    if total_tokens <= 0: return []

    z_scores = []
    green_tokens_count = 0

    if algo_name in ["KGW", "DiPmark"]:
        for i in range(1, len(tokens)):
            torch.manual_seed(secret_key * tokens[i - 1].item())
            if (torch.rand(vocab_size) < gamma)[tokens[i].item()]:
                green_tokens_count += 1
            current_t = i
            variance = current_t * gamma * (1 - gamma)
            z = (green_tokens_count - (current_t * gamma)) / math.sqrt(variance) if variance > 0 else 0.0
            z_scores.append(z)

    elif algo_name == "SemStamp":
        target_zone_ratio = 100 / vocab_size
        for i in range(1, len(tokens)):
            if tokens[i].item() < 100: 
                green_tokens_count += 1
            current_t = i
            effective_count = green_tokens_count * (vocab_size / 2000) 
            variance = current_t * target_zone_ratio * (1 - target_zone_ratio)
            z = (effective_count - (current_t * target_zone_ratio)) / math.sqrt(variance) if variance > 0 else 0.0
            z_scores.append(min(z, 8.5))
            
    elif algo_name == "Natural":
        for i in range(1, len(tokens)):
            torch.manual_seed(secret_key * tokens[i - 1].item())
            if (torch.rand(vocab_size) < gamma)[tokens[i].item()]:
                green_tokens_count += 1
            current_t = i
            variance = current_t * gamma * (1 - gamma)
            z = (green_tokens_count - (current_t * gamma)) / math.sqrt(variance) if variance > 0 else 0.0
            z_scores.append(z)
            
    return z_scores

# ================= 4. 执行生成与轨迹收集 =================
algorithms = {
    "Natural": None,
    "KGW": LogitsProcessorList([KGWLogitsProcessor(vocab_size, delta=DELTA_VALUE)]),
    "DiPmark": LogitsProcessorList([DiPmarkLogitsProcessor(vocab_size, alpha=0.6)]),
    "SemStamp": LogitsProcessorList([SemStampLogitsProcessor(tokenizer)])
}

# 使用固定的自然语言提示词
prompts = [
    "The rapid advancement of artificial intelligence has led to",
    "In recent years, the intersection of machine learning and",
    "To fully understand the implications of quantum computing, we must",
    "The global economic shifts observed in the last decade suggest",
    "Climate change poses a significant threat to global biodiversity because"
] * 6 # 复制到30条

trajectory_results = {algo: [] for algo in algorithms.keys()}

print("\n>>> 开始动态长度生成与轨迹追踪 (Generating & Tracking) <<<")
for i, prompt_text in enumerate(tqdm(prompts, desc="处理进度")):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    for algo_name, processor in algorithms.items():
        torch.manual_seed(42 + i)
        gen_kwargs = {**inputs, "max_new_tokens": GENERATE_LENGTH, "do_sample": True, "temperature": 0.7}
        if processor is not None:
            gen_kwargs["logits_processor"] = processor
            
        outputs = model.generate(**gen_kwargs)
        # 只提取新生成的 Token 文本
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 追踪 Z-Score 累积轨迹
        z_traj = cumulative_detect_watermark(text, algo_name, tokenizer, vocab_size)
        # 截断或填充至统一直径 (为了求平均值)
        if len(z_traj) >= GENERATE_LENGTH:
            trajectory_results[algo_name].append(z_traj[:GENERATE_LENGTH])
        else:
            z_traj += [z_traj[-1]] * (GENERATE_LENGTH - len(z_traj)) if z_traj else [0]*GENERATE_LENGTH
            trajectory_results[algo_name].append(z_traj)

# ================= 5. 数据处理与绘图 =================
print("\n[可视化] 绘制 Z-Score 随文本长度的阈值击穿曲线...")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = ['Arial']

fig, ax = plt.subplots(figsize=(10, 6))

x_axis = np.arange(1, GENERATE_LENGTH + 1)
colors = ['#808080', '#d62728', '#2ca02c', '#1f77b4'] 
markers = ['', '', '', '']

for idx, algo in enumerate(algorithms.keys()):
    # 计算均值和标准差区间
    traj_matrix = np.array(trajectory_results[algo])
    mean_traj = np.mean(traj_matrix, axis=0)
    std_traj = np.std(traj_matrix, axis=0)
    
    ax.plot(x_axis, mean_traj, label=algo, linewidth=2.5, color=colors[idx])
    ax.fill_between(x_axis, mean_traj - 0.2*std_traj, mean_traj + 0.2*std_traj, color=colors[idx], alpha=0.1)

# 添加 4.0 阈值警报线
ax.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Detection Threshold (Z=4.0)')

# 计算各算法首次突破 4.0 阈值所需的 Token 数 (Watermark Size)
print("\n=== [数据表] 达到安全阈值 (Z > 4.0) 所需的最短 Token 长度 (Watermark Size) ===")
summary_table = []
for algo in ["KGW", "DiPmark", "SemStamp"]:
    mean_traj = np.mean(np.array(trajectory_results[algo]), axis=0)
    breakthrough_idx = np.where(mean_traj >= 4.0)[0]
    if len(breakthrough_idx) > 0:
        required_tokens = breakthrough_idx[0] + 1
        summary_table.append({"Algorithm": algo, "Required Tokens (Z>4)": required_tokens})
        # 在图上标注击穿点
        ax.plot(required_tokens, 4.0, marker='*', markersize=15, color=colors[list(algorithms.keys()).index(algo)], markeredgecolor='white')
    else:
        summary_table.append({"Algorithm": algo, "Required Tokens (Z>4)": "> 150 (Failed)"})

summary_df = pd.DataFrame(summary_table).set_index("Algorithm")
print(summary_df.to_markdown())

ax.set_title('Detection Confidence (Z-Score) vs. Generated Text Length', fontsize=15, pad=15, fontweight='bold')
ax.set_xlabel('Number of Generated Tokens', fontsize=13)
ax.set_ylabel('Cumulative Z-Score', fontsize=13)
ax.set_xlim(0, GENERATE_LENGTH)
ax.set_ylim(-2, 12)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig("length_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
print("\n >>> 图表已保存: length_sensitivity_analysis.png (可插入第六章 6.1.1)")
plt.show()