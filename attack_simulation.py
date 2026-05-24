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

# --- 新增：字符级擦除攻击 (CharacterRemoval4WM 代理) ---
def character_removal_attack(text, drop_ratio):
    """
    代理实现 引用8: CharacterRemoval4WM
    直接在字符维度制造离散破坏，破坏 BPE Tokenizer 的边界划分。
    """
    if not text: return str(text)
    chars = list(text)
    num_drop = int(len(chars) * drop_ratio)
    indices_to_drop = set(random.sample(range(len(chars)), num_drop))
    return "".join([c for i, c in enumerate(chars) if i not in indices_to_drop])

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

# --- SIRA 核心组件 (引用10) ---
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

# ================= 4. 测试一：Word Drop & Char Removal 退化测试 =================
print("\n[测试阶段 1/5] 开始物理层扰动 (删词 & 删字符) 鲁棒性退化测试...")
attack_ratios = [0.0, 0.1, 0.3, 0.5]
results_word_drop = {algo: [] for algo in algorithms}
results_char_drop = {algo: [] for algo in algorithms}

for ratio in attack_ratios:
    for algo in algorithms:
        w_z = [detect_watermark(simulate_word_drop(text, ratio), algo, detector_tokenizer, vocab_size) for text in df[f"Text_{algo}"]]
        results_word_drop[algo].append(sum(w_z) / len(w_z) if w_z else 0.0)
        
        c_z = [detect_watermark(character_removal_attack(text, ratio), algo, detector_tokenizer, vocab_size) for text in df[f"Text_{algo}"]]
        results_char_drop[algo].append(sum(c_z) / len(c_z) if c_z else 0.0)

df_w_summary = pd.DataFrame(results_word_drop)
df_w_summary.index = [f"Drop {int(r*100)}%" for r in attack_ratios]
summary_table_1_w = df_w_summary.T.round(3)

df_c_summary = pd.DataFrame(results_char_drop)
df_c_summary.index = [f"Drop {int(r*100)}%" for r in attack_ratios]
summary_table_1_c = df_c_summary.T.round(3)

# --- 改为 2x2 布局 ---
fig = plt.figure(figsize=(18, 12)) 
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1.2], height_ratios=[1, 1]) 
ax1 = fig.add_subplot(gs[0, 0]) # 删词 图
ax2 = fig.add_subplot(gs[0, 1]) # 删词 表
ax3 = fig.add_subplot(gs[1, 0]) # 删字 图
ax4 = fig.add_subplot(gs[1, 1]) # 删字 表

markers = ['o', 's', '^', 'D', 'v', 'p']
# 绘制删词
for i, algo in enumerate(algorithms):
    ax1.plot([r * 100 for r in attack_ratios], results_word_drop[algo], marker=markers[i % len(markers)], linewidth=2.5, markersize=8, label=algo)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
ax1.set_title('Test 1A: Robustness under Word Drop', fontsize=14, fontweight='bold')
ax1.set_ylabel('Average Z-Score', fontsize=13)
all_w_scores = [score for scores in results_word_drop.values() for score in scores]
ax1.set_ylim(min(min(all_w_scores) - 0.5, -0.5), max(max(all_w_scores) + 0.5, 4.5))
ax1.legend(loc='lower left')

# 绘制删字
for i, algo in enumerate(algorithms):
    ax3.plot([r * 100 for r in attack_ratios], results_char_drop[algo], marker=markers[i % len(markers)], linewidth=2.5, markersize=8, label=algo)
ax3.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
ax3.set_title('Test 1B: Robustness under Character Removal', fontsize=14, fontweight='bold')
ax3.set_xlabel('Modification Ratio (%)', fontsize=13)
ax3.set_ylabel('Average Z-Score', fontsize=13)
all_c_scores = [score for scores in results_char_drop.values() for score in scores]
ax3.set_ylim(min(min(all_c_scores) - 0.5, -0.5), max(max(all_c_scores) + 0.5, 4.5))
ax3.legend(loc='lower left')

# 删词表格
t1_w_data = summary_table_1_w.reset_index()
t1_w_data.rename(columns={'index': 'Algo(Z)'}, inplace=True)
ax2.axis('off')
ax2.set_title('Word Drop Summary', fontsize=13, fontweight='bold', pad=10)
tab1 = ax2.table(cellText=t1_w_data.values, colLabels=t1_w_data.columns, loc='center', cellLoc='center')
tab1.auto_set_font_size(False); tab1.set_fontsize(11); tab1.scale(1, 2.5)

# 删字表格
t1_c_data = summary_table_1_c.reset_index()
t1_c_data.rename(columns={'index': 'Algo(Z)'}, inplace=True)
ax4.axis('off')
ax4.set_title('Character Removal Summary', fontsize=13, fontweight='bold', pad=10)
tab2 = ax4.table(cellText=t1_c_data.values, colLabels=t1_c_data.columns, loc='center', cellLoc='center')
tab2.auto_set_font_size(False); tab2.set_fontsize(11); tab2.scale(1, 2.5)

plt.tight_layout()
plt.savefig("attack_1_physical_perturbation.png", dpi=300, bbox_inches='tight')
print("  >>> 图表 1 已保存: attack_1_physical_perturbation.png")
plt.close()

# ================= 5. 测试二：高级洗稿与跨语言打击测试 =================
print("\n[测试阶段 2/5] 开始 LLM Rewrite 与 CWRA 跨语言纵深打击...")
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
print("  >>> 图表 2 已保存: attack_2_complex_rewrite.png")
plt.show()
plt.close()

# ================= 6. 测试三：黑盒 API 逆向窃取与伪造攻击 (WS) =================
print("\n[测试阶段 3/5] 开始黑盒 API 逆向窃取 (Watermark Stealing) 模拟...")
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
        
        ws_results.append({"Algorithm": algo, "Category": "Scrubbing", "State": "1_Original", "Z_Score": original_z})
        ws_results.append({"Algorithm": algo, "Category": "Scrubbing", "State": "2_After Scrub", "Z_Score": scrubbed_z})

        natural_z = detect_watermark(natural_text, algo, detector_tokenizer, vocab_size)
        spoofed_text = watermark_stealing_attack(natural_text, algo, "spoofing", detector_tokenizer, vocab_size)
        spoofed_z = detect_watermark(spoofed_text, algo, detector_tokenizer, vocab_size)

        ws_results.append({"Algorithm": algo, "Category": "Spoofing", "State": "3_Natural", "Z_Score": natural_z})
        ws_results.append({"Algorithm": algo, "Category": "Spoofing", "State": "4_After Spoof", "Z_Score": spoofed_z})

df_ws = pd.DataFrame(ws_results)
df_ws['State'] = df_ws['State'].apply(lambda x: x.split('_')[1])

# 独立生成两个子表
df_scrub = df_ws[df_ws["Category"] == "Scrubbing"]
summary_table_3_scrub = df_scrub.pivot_table(values='Z_Score', index='Algorithm', columns='State', aggfunc='mean').round(3)

df_spoof = df_ws[df_ws["Category"] == "Spoofing"]
summary_table_3_spoof = df_spoof.pivot_table(values='Z_Score', index='Algorithm', columns='State', aggfunc='mean').round(3)

# --- 改为 2x2 布局 ---
fig3 = plt.figure(figsize=(18, 12))
gs3 = fig3.add_gridspec(2, 2, width_ratios=[2, 1.2], height_ratios=[1, 1])
ax3_1 = fig3.add_subplot(gs3[0, 0]) # 擦除 图
ax3_2 = fig3.add_subplot(gs3[0, 1]) # 擦除 表
ax3_3 = fig3.add_subplot(gs3[1, 0]) # 伪造 图
ax3_4 = fig3.add_subplot(gs3[1, 1]) # 伪造 表

# 擦除区域
sns.boxplot(x="Algorithm", y="Z_Score", hue="State", data=df_scrub, ax=ax3_1, width=0.6, showfliers=False)
sns.stripplot(x="Algorithm", y="Z_Score", hue="State", data=df_scrub, ax=ax3_1, dodge=True, color='black', alpha=0.3)
ax3_1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold')
ax3_1.set_title('Test 3A: WS Attack - Scrubbing (Targeted Removal)', fontsize=14, fontweight='bold')
ax3_1.set_ylabel('Z-Score', fontsize=13)
ax3_1.legend(loc='lower left')

t3_scrub_data = summary_table_3_scrub.reset_index()
ax3_2.axis('off')
ax3_2.set_title('Scrubbing Summary', fontsize=13, fontweight='bold', pad=10)
tab3 = ax3_2.table(cellText=t3_scrub_data.values, colLabels=t3_scrub_data.columns, loc='center', cellLoc='center')
tab3.auto_set_font_size(False); tab3.set_fontsize(11); tab3.scale(1, 2.5)

# 伪造区域
sns.boxplot(x="Algorithm", y="Z_Score", hue="State", data=df_spoof, ax=ax3_3, width=0.6, showfliers=False)
sns.stripplot(x="Algorithm", y="Z_Score", hue="State", data=df_spoof, ax=ax3_3, dodge=True, color='black', alpha=0.3)
ax3_3.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold')
ax3_3.set_title('Test 3B: WS Attack - Spoofing (Framing Natural Text)', fontsize=14, fontweight='bold')
ax3_3.set_ylabel('Z-Score', fontsize=13)
ax3_3.legend(loc='upper left')

t3_spoof_data = summary_table_3_spoof.reset_index()
ax3_4.axis('off')
ax3_4.set_title('Spoofing Summary', fontsize=13, fontweight='bold', pad=10)
tab4 = ax3_4.table(cellText=t3_spoof_data.values, colLabels=t3_spoof_data.columns, loc='center', cellLoc='center')
tab4.auto_set_font_size(False); tab4.set_fontsize(11); tab4.scale(1, 2.5)

plt.tight_layout()
plt.savefig("attack_3_watermark_stealing.png", dpi=300, bbox_inches='tight')
print("  >>> 图表 3 已保存: attack_3_watermark_stealing.png")
plt.close()

# ================= 7. 测试四：SIRA 自信息靶向重写攻击 =================
print("\n[测试阶段 4/5] 开始 SIRA 高熵靶向重写降维打击...")
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
plt.show()
plt.close()

# ================= 8. 测试五：哈希窗口大小的博弈权衡 =================
print("\n[测试阶段 5/5] 开始哈希窗口大小 (h) 的多维博弈权衡分析...")
print("  >>> 正在验证第五章 5.2: 鲁棒性与窃取难度的天然冲突")

def simulate_kgw_with_window(text, window_size, drop_ratio=0.0):
    tokens = detector_tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= window_size: return 0.0, 0
    
    tampered_tokens = tokens.copy()
    if drop_ratio > 0:
        num_drop = int(len(tokens) * drop_ratio)
        drop_indices = set(random.sample(range(window_size, len(tokens)), num_drop))
        tampered_tokens = [t for i, t in enumerate(tokens) if i not in drop_indices]
        
    if len(tampered_tokens) <= window_size: return 0.0, 0

    green_count = 0
    gamma = 0.5
    total = len(tampered_tokens) - window_size
    stealing_complexity_log = window_size * math.log10(vocab_size)
    
    for i in range(window_size, len(tampered_tokens)):
        prefix_sum = sum(tampered_tokens[i-window_size:i])
        torch.manual_seed(15485863 * prefix_sum) 
        if (torch.rand(vocab_size) < gamma)[tampered_tokens[i]]:
            green_count += 1
            
    variance = total * gamma * (1 - gamma)
    z_score = (green_count - (total * gamma)) / math.sqrt(variance) if variance > 0 else 0.0
    return z_score, stealing_complexity_log

window_sizes = [1, 2, 3]
tradeoff_results = []
sample_tradeoff_df = df.head(30)

for h in window_sizes:
    avg_z_robust = 0
    avg_complexity = 0
    for _, row in sample_tradeoff_df.iterrows():
        base_text = row.get("Text_Natural", "The quick brown fox jumps over the lazy dog.")
        z_robust, complexity = simulate_kgw_with_window(base_text, h, drop_ratio=0.3)
        avg_z_robust += z_robust
        avg_complexity = complexity 
    avg_z_robust /= len(sample_tradeoff_df)
    
    tradeoff_results.append({
        "Window Size (h)": h,
        "Robustness (Z-Score after 30% Drop)": avg_z_robust,
        "Security (Log10 Stealing Cost)": avg_complexity
    })

df_tradeoff = pd.DataFrame(tradeoff_results)
print("\n=== [数据表] 哈希窗口博弈权衡分析 ===")
print(df_tradeoff.to_markdown(index=False))

fig5, ax5_1 = plt.subplots(figsize=(10, 6))

color1 = '#d9534f' 
ax5_1.set_xlabel('Hash Window Size ($h$)', fontsize=13, fontweight='bold')
ax5_1.set_ylabel('Robustness: Z-Score (under 30% Drop Attack)', color=color1, fontsize=13, fontweight='bold')
ax5_1.plot(df_tradeoff["Window Size (h)"], df_tradeoff["Robustness (Z-Score after 30% Drop)"], 
           color=color1, marker='o', linewidth=3, markersize=10, label="Robustness")
ax5_1.tick_params(axis='y', labelcolor=color1)
ax5_1.set_xticks(window_sizes)
ax5_1.axhline(y=4.0, color=color1, linestyle='--', linewidth=1.5, alpha=0.6)

ax5_2 = ax5_1.twinx()  
color2 = '#1f77b4' 
ax5_2.set_ylabel('Security: Log10 Stealing Cost (API Queries)', color=color2, fontsize=13, fontweight='bold')
ax5_2.plot(df_tradeoff["Window Size (h)"], df_tradeoff["Security (Log10 Stealing Cost)"], 
           color=color2, marker='s', linewidth=3, markersize=10, label="Security (Stealing Cost)")
ax5_2.tick_params(axis='y', labelcolor=color2)

plt.title('Game Theory Trade-off: Window Size $h$ vs. Robustness & Security', fontsize=15, pad=20, fontweight='bold')
fig5.tight_layout()
plt.savefig("attack_5_window_tradeoff.png", dpi=300, bbox_inches='tight')
print("  >>> 图表 5 已保存: attack_5_window_tradeoff.png")
plt.show() 
plt.close()

print("\n=== [实验终局] 所有理论推演与攻防模拟全部圆满完成！ ===")