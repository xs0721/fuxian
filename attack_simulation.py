import pandas as pd
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel, AutoModelForCausalLM
import os
import warnings
import re

# 忽略警告，保持终端整洁
warnings.filterwarnings("ignore")

# 获取当前脚本的绝对路径，并强制切换工作目录到这里
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# ================= 1. 基础全局配置 =================
CACHE_DIR = "E:/Your_Cloud_Drive/hf_cache" 
TARGET_MODEL = "facebook/opt-125m"  
ATTACKER_MODEL = "t5-small"         
CSV_FILENAME = "watermark_benchmark_results.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 50272 # OPT 词表大小

print(">>> 正在初始化红蓝对抗双轨评估框架 <<<")
print(f"[{device.upper()}] 加载目标分词器与模型 (用于 SIRA 白盒/灰盒特征提取): {TARGET_MODEL}...")
detector_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, cache_dir=CACHE_DIR)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, cache_dir=CACHE_DIR).to(device)

print(f"[{device.upper()}] 部署重写攻击大模型: {ATTACKER_MODEL}...")
attacker_tokenizer = AutoTokenizer.from_pretrained(ATTACKER_MODEL, cache_dir=CACHE_DIR)
attacker_model = AutoModelForSeq2SeqLM.from_pretrained(ATTACKER_MODEL, cache_dir=CACHE_DIR).to(device)

print(f"[{device.upper()}] 部署 CWRA 跨语言回译攻击引擎 (En -> Fr -> En)...")
en_fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_model_name, cache_dir=CACHE_DIR)
en_fr_model = MarianMTModel.from_pretrained(en_fr_model_name, cache_dir=CACHE_DIR).to(device)

fr_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model_name, cache_dir=CACHE_DIR)
fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name, cache_dir=CACHE_DIR).to(device)

# ================= 2. 攻击引擎与检测器定义 =================
def simulate_word_drop(text, drop_ratio):
    words = str(text).split()
    if not words: return str(text)
    num_drop = int(len(words) * drop_ratio)
    indices_to_drop = set(random.sample(range(len(words)), num_drop))
    tampered_words = [word for i, word in enumerate(words) if i not in indices_to_drop]
    return " ".join(tampered_words)

def llm_paraphrase_attack(text):
    if not isinstance(text, str) or len(text.strip()) == 0: return text
    input_ids = attacker_tokenizer("paraphrase: " + text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
    with torch.no_grad():
        outputs = attacker_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    return attacker_tokenizer.decode(outputs[0], skip_special_tokens=True)

def cwra_translation_attack(text):
    if not isinstance(text, str) or len(text.strip()) == 0: return text
    try:
        inputs_en = en_fr_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs_fr = en_fr_model.generate(**inputs_en, max_length=512)
        fr_text = en_fr_tokenizer.decode(outputs_fr[0], skip_special_tokens=True)
        
        inputs_fr = fr_en_tokenizer(fr_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs_en = fr_en_model.generate(**inputs_fr, max_length=512)
        en_text = fr_en_tokenizer.decode(outputs_en[0], skip_special_tokens=True)
        return en_text
    except Exception as e:
        return text

def watermark_stealing_attack(text, algo_name, attack_type, tokenizer, vocab_size, gamma=0.5, secret_key=15485863):
    if algo_name in ["Natural", "SemStamp"]: return text 
    if not isinstance(text, str) or len(text.strip()) == 0: return text

    tokens = tokenizer.encode(text, return_tensors="pt")[0].tolist()
    if len(tokens) <= 1: return text

    tampered_tokens = [tokens[0]]
    for i in range(1, len(tokens)):
        prev = tokens[i-1]
        curr = tokens[i]

        torch.manual_seed(secret_key * prev if algo_name in ["KGW", "SWEET", "DiPmark"] else 42)
        green_mask = (torch.rand(vocab_size) < gamma)
        is_green = green_mask[curr].item()

        if attack_type == "scrubbing" and is_green and random.random() < 0.6:
            candidate = (curr + random.randint(1, 100)) % vocab_size
            while green_mask[candidate].item():
                candidate = (candidate + 1) % vocab_size
            tampered_tokens.append(candidate)
        
        elif attack_type == "spoofing" and not is_green and random.random() < 0.6:
            candidate = (curr + random.randint(1, 100)) % vocab_size
            while not green_mask[candidate].item():
                candidate = (candidate + 1) % vocab_size
            tampered_tokens.append(candidate)
        else:
            tampered_tokens.append(curr)

    return tokenizer.decode(tampered_tokens, skip_special_tokens=True)

# --- SIRA 核心组件 ---
def calculate_token_self_information(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    if input_ids.shape[1] <= 1: return [], []
    with torch.no_grad(): outputs = model(input_ids)
    logits = outputs.logits[0, :-1, :]
    target_ids = input_ids[0, 1:]
    probs = F.softmax(logits, dim=-1)
    token_probs = probs.gather(1, target_ids.unsqueeze(1)).squeeze(-1)
    self_info = -torch.log2(token_probs + 1e-10)
    tokens = tokenizer.convert_ids_to_tokens(target_ids)
    return tokens, self_info.cpu().numpy()

def sira_masking(text, model, tokenizer, mask_ratio=0.2, device="cuda"):
    tokens, self_info_scores = calculate_token_self_information(text, model, tokenizer, device)
    if not tokens or len(tokens) < 5: return text
    num_to_mask = max(1, int(len(tokens) * mask_ratio))
    highest_entropy_indices = np.argsort(self_info_scores)[-num_to_mask:]
    masked_tokens = []
    mask_count = 0
    i = 0
    while i < len(tokens):
        if i in highest_entropy_indices:
            masked_tokens.append(f"<extra_id_{mask_count}>")
            mask_count += 1
            while i < len(tokens) and i in highest_entropy_indices: i += 1
            continue
        else:
            clean_token = tokens[i].replace('Ġ', ' ').replace('Ċ', '\n')
            masked_tokens.append(clean_token)
            i += 1
    return "".join(masked_tokens).strip()

def sira_t5_infilling(masked_text, attacker_model, attacker_tokenizer, device):
    if "<extra_id_" not in masked_text: return masked_text
    inputs = attacker_tokenizer(masked_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = attacker_model.generate(**inputs, max_length=512, num_beams=4, temperature=0.8, do_sample=True, early_stopping=True)
    filled_content = attacker_tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_text = masked_text
    parts = [p.strip() for p in filled_content.split("<extra_id_") if p.strip()]
    for i, part in enumerate(parts):
        if '>' in part:
            clean_part = part.split('>', 1)[1].strip()
            final_text = final_text.replace(f"<extra_id_{i}>", " " + clean_part + " ")
    final_text = re.sub(r'<extra_id_\d+>', '', final_text)
    return final_text.replace("  ", " ").strip()
# ---------------------

def detect_watermark(text, algo_name, tokenizer, vocab_size, gamma=0.5, secret_key=15485863):
    if algo_name == "Natural": return 0.0
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    total_tokens = len(tokens) - 1
    if total_tokens <= 0: return 0.0
    green_tokens_count = 0

    if algo_name in ["KGW", "SWEET", "DiPmark"]:
        for i in range(1, len(tokens)):
            torch.manual_seed(secret_key * tokens[i - 1].item())
            if (torch.rand(vocab_size) < gamma)[tokens[i].item()]:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

    elif algo_name == "Unigram":
        torch.manual_seed(42) 
        green_mask = (torch.rand(vocab_size) < gamma)
        for i in range(1, len(tokens)):
            if green_mask[tokens[i].item()]:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

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
    print("找不到文件，请确认是否已生成 CSV。")
    exit()

algorithms = [col.replace("Text_", "") for col in df.columns if col.startswith("Text_")]
print(f"锁定目标防御算法: {algorithms}")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = ['Arial']

# ================= 4. 测试一：Word Drop 鲁棒性退化测试 =================
print("\n[测试阶段 1/4] 开始 Word Drop (删词) 退化测试...")
attack_ratios = [0.0, 0.1, 0.3, 0.5]
results_history_drop = {algo: [] for algo in algorithms}

for ratio in attack_ratios:
    for algo in algorithms:
        current_z_scores = [detect_watermark(simulate_word_drop(text, ratio), algo, detector_tokenizer, vocab_size) for text in df[f"Text_{algo}"]]
        avg_z = sum(current_z_scores) / len(current_z_scores) if current_z_scores else 0.0
        results_history_drop[algo].append(avg_z)

df_drop_summary = pd.DataFrame(results_history_drop)
df_drop_summary.index = [f"Drop {int(r*100)}%" for r in attack_ratios]
summary_table_1 = df_drop_summary.T.round(3)

fig = plt.figure(figsize=(14, 6)) 
gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1]) 
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

markers = ['o', 's', '^', 'D', 'v', 'p']
for i, algo in enumerate(algorithms):
    ax1.plot([r * 100 for r in attack_ratios], results_history_drop[algo], marker=markers[i % len(markers)], linewidth=2.5, markersize=8, label=algo)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
ax1.set_title('Attack Test 1: Robustness under Word Drop', fontsize=15, pad=15, fontweight='bold')
ax1.set_xlabel('Modification Ratio (%)', fontsize=13)
ax1.set_ylabel('Average Z-Score', fontsize=13)
all_drop_scores = [score for scores in results_history_drop.values() for score in scores]
ax1.set_ylim(min(min(all_drop_scores) - 0.5, -0.5), max(max(all_drop_scores) + 0.5, 4.5))
ax1.legend()

table1_data = summary_table_1.reset_index()
table1_data.rename(columns={'index': 'Algorithm\n(Z-Score)'}, inplace=True)
ax2.axis('off')
ax2.set_title('Data Summary (Metric: Z-Score)', fontsize=13, fontweight='bold', pad=10)
table1 = ax2.table(cellText=table1_data.values, colLabels=table1_data.columns, loc='center', cellLoc='center')
table1.auto_set_font_size(False)
table1.set_fontsize(10)
table1.scale(1, 2.5) 
plt.tight_layout()
plt.savefig("attack_1_word_drop.png", dpi=300, bbox_inches='tight')
plt.close()

# ================= 5. 测试二：高级洗稿与跨语言打击测试 =================
print("\n[测试阶段 2/4] 开始 LLM Rewrite 与 CWRA 跨语言纵深打击...")
sample_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).sample(n=min(20, len(df)), random_state=42)
attack_results_complex = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="高级攻击进度"):
    for algo in algorithms:
        original_text = row[f"Text_{algo}"]
        attack_results_complex.append({"Algorithm": algo, "State": "1_Before Attack", "Z_Score": detect_watermark(original_text, algo, detector_tokenizer, vocab_size)})
        t5_attacked_text = llm_paraphrase_attack(original_text)
        attack_results_complex.append({"Algorithm": algo, "State": "2_After T5 Rewrite", "Z_Score": detect_watermark(t5_attacked_text, algo, detector_tokenizer, vocab_size)})
        cwra_attacked_text = cwra_translation_attack(original_text)
        attack_results_complex.append({"Algorithm": algo, "State": "3_After CWRA", "Z_Score": detect_watermark(cwra_attacked_text, algo, detector_tokenizer, vocab_size)})

results_df = pd.DataFrame(attack_results_complex)
summary_table_2 = results_df.groupby(['Algorithm', 'State'])['Z_Score'].mean().unstack('State').round(3)
summary_table_2.columns = [col.split('_')[1] for col in summary_table_2.columns]

results_df['State'] = results_df['State'].apply(lambda x: x.split('_')[1])
fig2 = plt.figure(figsize=(16, 6))
gs2 = fig2.add_gridspec(1, 2, width_ratios=[2.5, 1])
ax2_1 = fig2.add_subplot(gs2[0])
ax2_2 = fig2.add_subplot(gs2[1])

sns.boxplot(x="Algorithm", y="Z_Score", hue="State", data=results_df, ax=ax2_1, width=0.7, showfliers=False)
sns.stripplot(x="Algorithm", y="Z_Score", hue="State", data=results_df, ax=ax2_1, dodge=True, color='black', alpha=0.3)
ax2_1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
ax2_1.set_title('Attack Test 2: Vulnerability to T5 Paraphrasing & CWRA', fontsize=15, pad=15, fontweight='bold')
ax2_1.set_ylabel('Z-Score', fontsize=13)
ax2_1.set_xlabel('Watermark Algorithm', fontsize=13)
all_complex_scores = results_df["Z_Score"].tolist()
ax2_1.set_ylim(min(min(all_complex_scores) - 0.5, -0.5), max(max(all_complex_scores) + 0.5, 4.5))
ax2_1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

table2_data = summary_table_2.reset_index()
table2_data.rename(columns={'Algorithm': 'Algorithm\n(Z-Score)'}, inplace=True)
ax2_2.axis('off')
ax2_2.set_title('Data Summary (Metric: Z-Score)', fontsize=13, fontweight='bold', pad=10)
table2 = ax2_2.table(cellText=table2_data.values, colLabels=table2_data.columns, loc='center', cellLoc='center')
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1, 2.5)
plt.tight_layout()
plt.savefig("attack_2_complex_rewrite.png", dpi=300, bbox_inches='tight')
plt.close()

# ================= 6. 测试三：黑盒 API 逆向窃取与伪造攻击 (WS) =================
print("\n[测试阶段 3/4] 开始黑盒 API 逆向窃取 (Watermark Stealing) 模拟...")
ws_results = []
sample_ws_df = df.head(20)

for idx, row in tqdm(sample_ws_df.iterrows(), total=len(sample_ws_df), desc="窃取与伪造进度"):
    natural_text = row["Text_Natural"] if "Text_Natural" in row else "The quick brown fox jumps over the lazy dog."
    for algo in algorithms:
        if algo in ["Natural", "SemStamp"]: continue
        
        watermarked_text = row[f"Text_{algo}"]
        original_z = detect_watermark(watermarked_text, algo, detector_tokenizer, vocab_size)
        scrubbed_text = watermark_stealing_attack(watermarked_text, algo, "scrubbing", detector_tokenizer, vocab_size)
        scrubbed_z = detect_watermark(scrubbed_text, algo, detector_tokenizer, vocab_size)
        
        ws_results.append({"Algorithm": algo, "Category": "Scrubbing (Remove)", "State": "1_Original Watermark", "Z_Score": original_z})
        ws_results.append({"Algorithm": algo, "Category": "Scrubbing (Remove)", "State": "2_After Scrubbing", "Z_Score": scrubbed_z})

        natural_z = detect_watermark(natural_text, algo, detector_tokenizer, vocab_size)
        spoofed_text = watermark_stealing_attack(natural_text, algo, "spoofing", detector_tokenizer, vocab_size)
        spoofed_z = detect_watermark(spoofed_text, algo, detector_tokenizer, vocab_size)

        ws_results.append({"Algorithm": algo, "Category": "Spoofing (Forge)", "State": "3_Natural Text", "Z_Score": natural_z})
        ws_results.append({"Algorithm": algo, "Category": "Spoofing (Forge)", "State": "4_After Spoofing", "Z_Score": spoofed_z})

df_ws = pd.DataFrame(ws_results)
summary_table_3 = df_ws.pivot_table(values='Z_Score', index=['Algorithm', 'Category'], columns='State', aggfunc='mean').round(3)
summary_table_3.columns = [col.split('_')[1] for col in summary_table_3.columns]
summary_table_3 = summary_table_3.fillna("-")

df_ws['State'] = df_ws['State'].apply(lambda x: x.split('_')[1])
fig3 = plt.figure(figsize=(19, 6))
gs3 = fig3.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 1.2])
ax3_1 = fig3.add_subplot(gs3[0])
ax3_2 = fig3.add_subplot(gs3[1])
ax3_3 = fig3.add_subplot(gs3[2])

df_scrub = df_ws[df_ws["Category"] == "Scrubbing (Remove)"]
sns.boxplot(x="Algorithm", y="Z_Score", hue="State", data=df_scrub, ax=ax3_1, width=0.6, showfliers=False)
sns.stripplot(x="Algorithm", y="Z_Score", hue="State", data=df_scrub, ax=ax3_1, dodge=True, color='black', alpha=0.3)
ax3_1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold')
ax3_1.set_title('WS Attack: Scrubbing (Removal)', fontsize=14, fontweight='bold')
ax3_1.set_ylabel('Z-Score', fontsize=13)
ax3_1.legend(loc='lower left')

df_spoof = df_ws[df_ws["Category"] == "Spoofing (Forge)"]
sns.boxplot(x="Algorithm", y="Z_Score", hue="State", data=df_spoof, ax=ax3_2, width=0.6, showfliers=False)
sns.stripplot(x="Algorithm", y="Z_Score", hue="State", data=df_spoof, ax=ax3_2, dodge=True, color='black', alpha=0.3)
ax3_2.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold')
ax3_2.set_title('WS Attack: Spoofing (Forge)', fontsize=14, fontweight='bold')
ax3_2.set_ylabel('Z-Score', fontsize=13)
ax3_2.legend(loc='upper left')

table3_data = summary_table_3.reset_index()
table3_data.rename(columns={'Algorithm': 'Algo\n(Z-Score)', 'Category': 'Attack Type'}, inplace=True)
ax3_3.axis('off')
ax3_3.set_title('Data Summary (Metric: Z-Score)', fontsize=13, fontweight='bold', pad=10)
table3 = ax3_3.table(cellText=table3_data.values, colLabels=table3_data.columns, loc='center', cellLoc='center')
table3.auto_set_font_size(False)
table3.set_fontsize(10)
table3.scale(1, 2.5)
table3.auto_set_column_width(col=list(range(len(table3_data.columns))))
plt.tight_layout()
plt.savefig("attack_3_watermark_stealing.png", dpi=300, bbox_inches='tight')
plt.close()

# ================= 7. 测试四：SIRA 自信息靶向重写攻击 =================
print("\n[测试阶段 4/4] 开始 SIRA 高熵靶向重写降维打击...")
sira_results = []
sample_sira_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).sample(n=min(20, len(df)), random_state=42)

for idx, row in tqdm(sample_sira_df.iterrows(), total=len(sample_sira_df), desc="SIRA 定向手术刀攻击"):
    for algo in algorithms:
        if algo == "Natural": continue
        original_text = row[f"Text_{algo}"]
        z_before = detect_watermark(original_text, algo, detector_tokenizer, vocab_size)
        masked_text = sira_masking(original_text, target_model, detector_tokenizer, mask_ratio=0.2, device=device)
        sira_attacked_text = sira_t5_infilling(masked_text, attacker_model, attacker_tokenizer, device)
        z_after = detect_watermark(sira_attacked_text, algo, detector_tokenizer, vocab_size)
        
        sira_results.append({"Algorithm": algo, "State": "1_Before SIRA", "Z_Score": z_before})
        sira_results.append({"Algorithm": algo, "State": "2_After SIRA (20% Edit)", "Z_Score": z_after})

sira_df = pd.DataFrame(sira_results)
summary_table_4 = sira_df.groupby(['Algorithm', 'State'])['Z_Score'].mean().unstack('State').round(3)
summary_table_4.columns = [col.split('_')[1] for col in summary_table_4.columns]
print("\n=== [数据表] SIRA 高熵靶向重写 Z-Score 骤降汇总 ===")
print(summary_table_4.to_markdown())

sira_df['State'] = sira_df['State'].apply(lambda x: x.split('_')[1])
fig4 = plt.figure(figsize=(14, 6))
gs4 = fig4.add_gridspec(1, 2, width_ratios=[2, 1.2])
ax4_1 = fig4.add_subplot(gs4[0])
ax4_2 = fig4.add_subplot(gs4[1])

sns.boxplot(x="Algorithm", y="Z_Score", hue="State", data=sira_df, ax=ax4_1, width=0.6, showfliers=False)
sns.stripplot(x="Algorithm", y="Z_Score", hue="State", data=sira_df, ax=ax4_1, dodge=True, color='black', alpha=0.3)
ax4_1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
ax4_1.set_title('Attack Test 4: SIRA (Self-Information Rewrite Attack)', fontsize=14, fontweight='bold')
ax4_1.set_ylabel('Z-Score', fontsize=13)
ax4_1.legend(loc='upper right')

table4_data = summary_table_4.reset_index()
table4_data.rename(columns={'Algorithm': 'Algo\n(Z-Score)'}, inplace=True)
ax4_2.axis('off')
ax4_2.set_title('Data Summary (Metric: Z-Score)', fontsize=13, fontweight='bold', pad=10)
table4 = ax4_2.table(cellText=table4_data.values, colLabels=table4_data.columns, loc='center', cellLoc='center')
table4.auto_set_font_size(False)
table4.set_fontsize(10)
table4.scale(1, 2.5)
table4.auto_set_column_width(col=list(range(len(table4_data.columns))))
plt.tight_layout()
plt.savefig("attack_4_sira_targeted.png", dpi=300, bbox_inches='tight')
print("  >>> 图表 4 已保存: attack_4_sira_targeted.png")
plt.close()

print("\n=== 所有攻击模拟与评估测试圆满完成！===")