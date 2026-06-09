"""测试16: 鲁棒性利用伪造攻击 — 有毒词插入 + 内容修改, 水印检测仍通过"""
from test_common import *
import re

print_test_header("鲁棒性利用伪造 (Robustness Spoofing) — NeurIPS 2024 No Free Lunch")

load_attacker()  # T5 用于内容修改模拟

# ── 攻击1: 有毒词插入 ──────────────────────────────
# 对齐 attack_robustness.py insert_banned_words
BANNED_WORDS = [
    'fuck', 'shit', 'damn', 'bitch', 'penis', 'cum', 'ass', 'balls', 'tits',
    'pussy', 'dick', 'slut', 'cock', 'clit', 'fetish', 'kike', 'nude',
    'fucking', 'porn', 'creampie', 'doggystyle', 'whore', 'ejaculate',
    'masturbate', 'butt', 'asshole', 'rape', 'racism', 'blackmail', 'piss',
    'cunt', 'motherfucker'
]
BANNED_WORDS += [w.capitalize() for w in BANNED_WORDS]


def _insert_banned_words(text, tokenizer, vocab_size, gamma=0.5,
                          secret_key=15485863, max_inserts=20):
    """向水印文本随机插入有毒词, 只要z-score≥4就继续.

    攻击原理: 水印检测基于n-gram绿名单统计, 对token插入具有一定鲁棒性.
    插入位置随机选择, 插入token可能落入绿名单(概率γ), 也可能不落入.
    只要总体绿名单密度高于阈值, 检测仍然通过.
    攻击者利用这点: 在保持水印检测的同时, 让文本包含有害内容.
    """
    tokens = tokenizer.encode(text, return_tensors='pt')[0].tolist()
    orig_z = detect_watermark(text, "KGW", tokenizer, vocab_size,
                               gamma=gamma, secret_key=secret_key)
    if orig_z < 4.0:
        return text, 0, orig_z, orig_z

    inserted_count = 0
    for _ in range(max_inserts):
        # 随机选择插入位置
        insert_pos = random.randint(1, len(tokens) - 1)
        banned_word = random.choice(BANNED_WORDS)
        banned_tokens = tokenizer.encode(" " + banned_word, add_special_tokens=False)

        # 插入
        new_tokens = tokens[:insert_pos] + banned_tokens + tokens[insert_pos:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        new_z = detect_watermark(new_text, "KGW", tokenizer, vocab_size,
                                  gamma=gamma, secret_key=secret_key)

        if new_z < 4.0:
            break  # 检测失败, 停止插入

        tokens = new_tokens
        inserted_count += 1

    final_text = tokenizer.decode(tokens, skip_special_tokens=True)
    final_z = detect_watermark(final_text, "KGW", tokenizer, vocab_size,
                                gamma=gamma, secret_key=secret_key)

    return final_text, inserted_count, orig_z, final_z


# ── 攻击2: 内容修改 (模拟GPT-4, 用T5改写局部替代) ──
def _modify_content(text, tokenizer, vocab_size, gamma=0.5, secret_key=15485863):
    """修改水印文本内容使其不准确/含义相反, 同时保持水印检测.

    真实攻击用 GPT-4 API: "modify <3 words, make it inaccurate or opposite"
    这里用 T5 改写 + 随机扰动模拟, 捕获核心思想.
    """
    orig_z = detect_watermark(text, "KGW", tokenizer, vocab_size,
                               gamma=gamma, secret_key=secret_key)

    # T5 改写 → 保持大部分水印信号的同时改变语义
    modified = llm_paraphrase_attack(text)

    mod_z = detect_watermark(modified, "KGW", tokenizer, vocab_size,
                              gamma=gamma, secret_key=secret_key)

    return modified, orig_z, mod_z


# ── 主流程 ─────────────────────────────────────────
print("  >>> 生成KGW水印文本, 对比: 原始 vs 有毒词插入 vs 内容修改")

GAMMA = 0.5; DELTA = 2.0; HASH_KEY = 15485863
kgw_processor = LogitsProcessorList([
    KGWLogitsProcessor(vocab_size, gamma=GAMMA, delta=DELTA, hash_key=HASH_KEY)
])

sample_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).head(8)
robust_results = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="鲁棒性伪造"):
    prompt = str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))[:60]
    inputs = detector_tokenizer(prompt, return_tensors="pt").to(device)

    # ── 1. 生成 KGW 水印文本 ──
    torch.manual_seed(42)
    with torch.no_grad():
        out = target_model.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.7,
            logits_processor=kgw_processor,
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    kgw_text = detector_tokenizer.decode(out[0], skip_special_tokens=True)
    z_orig = detect_watermark(kgw_text, "KGW", detector_tokenizer, vocab_size,
                               gamma=GAMMA, secret_key=HASH_KEY)

    # ── 2. 有毒词插入攻击 ──
    toxic_text, n_inserted, _, z_toxic = _insert_banned_words(
        kgw_text, detector_tokenizer, vocab_size,
        gamma=GAMMA, secret_key=HASH_KEY, max_inserts=15,
    )

    # ── 3. 内容修改攻击 ──
    modified_text, _, z_modify = _modify_content(
        kgw_text, detector_tokenizer, vocab_size,
        gamma=GAMMA, secret_key=HASH_KEY,
    )

    robust_results.append({"Method": "KGW (Original)", "Z_Score": z_orig,
                           "Toxic_Inserts": 0})
    robust_results.append({"Method": "KGW + Toxic Insert", "Z_Score": z_toxic,
                           "Toxic_Inserts": n_inserted})
    robust_results.append({"Method": "KGW + Modified", "Z_Score": z_modify,
                           "Toxic_Inserts": 0})

# ── 汇总 ──────────────────────────────────────────
df_rb = pd.DataFrame(robust_results)
summary_rb = df_rb.groupby('Method').agg(
    Z_Mean=('Z_Score', 'mean'), Z_Std=('Z_Score', 'std'),
    Avg_Inserts=('Toxic_Inserts', 'mean'),
).round(3)
summary_rb['Still Detected?'] = summary_rb['Z_Mean'].apply(
    lambda x: 'YES (>4)' if x >= 4.0 else 'NO  (<4)'
)

print(f"\n=== [数据表] 鲁棒性利用伪造 (Toxic Insert + Content Modify) ===")
print("  对齐: LLM-Watermark-Attacks attack_robustness.py")
print("  原理: 水印检测对插入/改写具有鲁棒性, 攻击者利用这点")
print("  在保持检测通过的同时, 让文本变得有毒或不准确\n")
print(summary_rb.to_string())

# ── 绘图 ──────────────────────────────────────────
fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.2])
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

order = ["KGW (Original)", "KGW + Toxic Insert", "KGW + Modified"]
pal = {"KGW (Original)": "#2c7bb6", "KGW + Toxic Insert": "#d7191c",
       "KGW + Modified": "#fdae61"}
sns.boxplot(x="Method", y="Z_Score", data=df_rb, ax=ax1,
            order=order, palette=pal, width=0.5, showfliers=False)
sns.stripplot(x="Method", y="Z_Score", data=df_rb, ax=ax1,
              order=order, color='black', alpha=0.3, size=4)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2,
            label='Detection Threshold (z=4.0)')
ax1.set_title('Test 16: Robustness Exploitation Spoofing\n'
              'Toxic word insertion + content modification while z ≥ 4',
              fontsize=12, fontweight='bold', pad=15)
ax1.set_ylabel('Z-Score', fontsize=12); ax1.set_xlabel('')
ax1.legend(loc='lower left'); ax1.tick_params(axis='x', rotation=8)

table_data = summary_rb.reset_index()
ax2.axis('off'); ax2.set_title('Mean Z-Score', fontsize=13, fontweight='bold', pad=10)
tab = ax2.table(cellText=table_data.values, colLabels=table_data.columns,
                loc='center', cellLoc='center')
tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1.1, 2.5)
for j in range(len(table_data.columns)):
    tab[0, j].set_facecolor('#40466e'); tab[0, j].set_text_props(color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("attack_16_robustness_spoofing.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_16_robustness_spoofing.png")
plt.show(); plt.close()
print("=== 测试16完成 ===\n")
