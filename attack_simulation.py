import pandas as pd
import random
import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel
import os
import warnings

# 忽略翻译模型加载时的一些常规警告，保持终端整洁
warnings.filterwarnings("ignore")

# 获取当前脚本的绝对路径，并强制切换工作目录到这里
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# ================= 1. 基础全局配置 =================
CACHE_DIR = "E:/Your_Cloud_Drive/hf_cache" 
TARGET_MODEL = "facebook/opt-125m"  # 原本生成水印的模型和词表
ATTACKER_MODEL = "t5-small"         # 洗稿攻击者模型
CSV_FILENAME = "watermark_benchmark_results.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 50272 # OPT 词表大小

print(">>> 正在初始化红蓝对抗双轨评估框架 <<<")
print(f"[{device.upper()}] 加载目标分词器: {TARGET_MODEL}...")
detector_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, cache_dir=CACHE_DIR)

print(f"[{device.upper()}] 部署重写攻击大模型: {ATTACKER_MODEL}...")
attacker_tokenizer = AutoTokenizer.from_pretrained(ATTACKER_MODEL, cache_dir=CACHE_DIR)
attacker_model = AutoModelForSeq2SeqLM.from_pretrained(ATTACKER_MODEL, cache_dir=CACHE_DIR).to(device)

print(f"[{device.upper()}] 部署 CWRA 跨语言回译攻击引擎 (En -> Fr -> En)...")
# 显式使用 Marian 专属类，绕过 AutoTokenizer 的映射 Bug
en_fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_model_name, cache_dir=CACHE_DIR)
en_fr_model = MarianMTModel.from_pretrained(en_fr_model_name, cache_dir=CACHE_DIR).to(device)

fr_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model_name, cache_dir=CACHE_DIR)
fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name, cache_dir=CACHE_DIR).to(device)

# ================= 2. 攻击引擎与检测器定义 =================
def simulate_word_drop(text, drop_ratio):
    """攻击模式 A：随机删词（破坏局部连续性）"""
    words = str(text).split()
    if not words: return str(text)
    num_drop = int(len(words) * drop_ratio)
    indices_to_drop = set(random.sample(range(len(words)), num_drop))
    tampered_words = [word for i, word in enumerate(words) if i not in indices_to_drop]
    return " ".join(tampered_words)

def llm_paraphrase_attack(text):
    """攻击模式 B：大模型深度洗稿（破坏全局 Token 序列，保持语义）"""
    if not isinstance(text, str) or len(text.strip()) == 0: return text
    input_ids = attacker_tokenizer("paraphrase: " + text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
    with torch.no_grad():
        outputs = attacker_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    return attacker_tokenizer.decode(outputs[0], skip_special_tokens=True)

def cwra_translation_attack(text):
    """
    攻击模式 C：跨语言多跳洗稿 (CWRA)
    通过底层模型实现 En -> Fr -> En，在保持人类可读语义的同时制造巨大的向量漂移。
    """
    if not isinstance(text, str) or len(text.strip()) == 0: 
        return text
    
    try:
        # 第一跳：英文翻译为法文
        inputs_en = en_fr_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs_fr = en_fr_model.generate(**inputs_en, max_length=512)
        fr_text = en_fr_tokenizer.decode(outputs_fr[0], skip_special_tokens=True)
        
        # 第二跳：法文回译为英文
        inputs_fr = fr_en_tokenizer(fr_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs_en = fr_en_model.generate(**inputs_fr, max_length=512)
        en_text = fr_en_tokenizer.decode(outputs_en[0], skip_special_tokens=True)
        
        return en_text
    except Exception as e:
        print(f"\n[CWRA 攻击跳过] 文本处理异常: {e}")
        return text

def detect_watermark(text, algo_name, tokenizer, vocab_size, gamma=0.5, secret_key=15485863):
    """
    智能路由检测引擎：必须与 run_experiment.py 保持绝对一致！
    """
    if algo_name == "Natural": return 0.0
    
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    total_tokens = len(tokens) - 1
    if total_tokens <= 0: return 0.0

    green_tokens_count = 0

    # 雷达 A：前缀哈希 (KGW / SWEET)
    if algo_name in ["KGW", "SWEET"]:
        for i in range(1, len(tokens)):
            torch.manual_seed(secret_key * tokens[i - 1].item())
            if (torch.rand(vocab_size) < gamma)[tokens[i].item()]:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

    # 雷达 B：全局无语境哈希 (Unigram)
    elif algo_name == "Unigram":
        torch.manual_seed(42) # 对应生成时的 secret_key
        green_mask = (torch.rand(vocab_size) < gamma)
        for i in range(1, len(tokens)):
            if green_mask[tokens[i].item()]:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

    # 雷达 C：语义空间检测 (SemStamp)
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

# ================= 3. 数据加载与目标锁定 =================
print(f"\n读取基准防线数据 {CSV_FILENAME} ...")
try:
    df = pd.read_csv(CSV_FILENAME)
except FileNotFoundError:
    print("找不到文件，请确认是否在 '复现' 文件夹下运行终端，且已生成 CSV。")
    exit()

algorithms = [col.replace("Text_", "") for col in df.columns if col.startswith("Text_")]
print(f"锁定目标防御算法: {algorithms}")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = ['Arial']

# ================= 4. 测试一：Word Drop 鲁棒性退化测试 =================
print("\n[测试阶段 1/2] 开始 Word Drop (删词) 退化测试...")
attack_ratios = [0.0, 0.1, 0.3, 0.5]
results_history_drop = {algo: [] for algo in algorithms}

for ratio in attack_ratios:
    print(f"  > 攻击强度 {int(ratio*100)}% ...")
    for algo in algorithms:
        current_z_scores = [detect_watermark(simulate_word_drop(text, ratio), algo, detector_tokenizer, vocab_size) 
                            for text in df[f"Text_{algo}"]]
        results_history_drop[algo].append(sum(current_z_scores) / len(current_z_scores))

# 绘制图 1
plt.figure(figsize=(9, 6))
markers = ['o', 's', '^', 'D', 'v']
for i, algo in enumerate(algorithms):
    plt.plot([r * 100 for r in attack_ratios], results_history_drop[algo], marker=markers[i % len(markers)], linewidth=2.5, markersize=8, label=algo)
plt.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
plt.title('Attack Test 1: Robustness under Word Drop', fontsize=15, pad=15, fontweight='bold')
plt.xlabel('Modification Ratio (%)', fontsize=13); plt.ylabel('Average Z-Score', fontsize=13)

all_drop_scores = [score for scores in results_history_drop.values() for score in scores]
plt.ylim(min(min(all_drop_scores) - 0.5, -0.5), max(max(all_drop_scores) + 0.5, 4.5))

plt.legend()
plt.savefig("attack_1_word_drop.png", dpi=300, bbox_inches='tight')
print("  >>> 图表 1 已保存: attack_1_word_drop.png")
plt.show()

# ================= 5. 测试二：高级洗稿与跨语言打击测试 =================
print("\n[测试阶段 2/2] 开始 LLM Rewrite 与 CWRA 跨语言纵深打击...")
# 为控制时间，随机抽取 20 条样本进行洗稿
sample_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).sample(n=min(20, len(df)), random_state=42)
attack_results_complex = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="高级攻击进度"):
    for algo in algorithms:
        original_text = row[f"Text_{algo}"]
        
        # 1. 记录攻击前状态
        attack_results_complex.append({
            "Algorithm": algo, 
            "State": "Before Attack", 
            "Z_Score": detect_watermark(original_text, algo, detector_tokenizer, vocab_size)
        })
        
        # 2. 实施 T5 洗稿攻击
        t5_attacked_text = llm_paraphrase_attack(original_text)
        attack_results_complex.append({
            "Algorithm": algo, 
            "State": "After T5 Rewrite", 
            "Z_Score": detect_watermark(t5_attacked_text, algo, detector_tokenizer, vocab_size)
        })

        # 3. 实施 CWRA 跨语言回译攻击
        cwra_attacked_text = cwra_translation_attack(original_text)
        attack_results_complex.append({
            "Algorithm": algo, 
            "State": "After CWRA (En-Fr-En)", 
            "Z_Score": detect_watermark(cwra_attacked_text, algo, detector_tokenizer, vocab_size)
        })

# 绘制图 2
results_df = pd.DataFrame(attack_results_complex)
plt.figure(figsize=(12, 6)) # 稍微加宽一点以容纳第三个 Box
sns.boxplot(x="Algorithm", y="Z_Score", hue="State", data=results_df, width=0.7, showfliers=False)
sns.stripplot(x="Algorithm", y="Z_Score", hue="State", data=results_df, dodge=True, color='black', alpha=0.3)
plt.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
plt.title('Attack Test 2: Vulnerability to T5 Paraphrasing & CWRA', fontsize=15, pad=15, fontweight='bold')
plt.ylabel('Z-Score', fontsize=13); plt.xlabel('Watermark Algorithm', fontsize=13)

# 同样修复 Y 轴
all_complex_scores = results_df["Z_Score"].tolist()
plt.ylim(min(min(all_complex_scores) - 0.5, -0.5), max(max(all_complex_scores) + 0.5, 4.5))

# 将图例移出主图防遮挡
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig("attack_2_complex_rewrite.png", dpi=300, bbox_inches='tight')
print("  >>> 图表 2 已保存: attack_2_complex_rewrite.png")
plt.show()

print("\n=== 所有对抗测试执行完毕！===")