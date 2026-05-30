"""测试10: 水印平滑 (Watermark Smoothing) — 连续绿度分布对抗 SIRA 评估"""
from test_common import *
import math

print_test_header("水印平滑 (Watermark Smoothing) 连续绿度分布 vs KGW 硬二元对抗 SIRA")

load_attacker()  # SIRA Stage 1 + Stage 3 需要 T5


# ── 检测器 ─────────────────────────────────────────
def detect_kgw(text, tokenizer, vocab_size, gamma=0.5, hash_key=15485863):
    """KGW 标准检测: z = (green_count - gamma*T) / sqrt(T*gamma*(1-gamma))"""
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    T = len(tokens) - 1
    if T <= 0:
        return 0.0

    green_count = 0
    greenlist_size = int(vocab_size * gamma)
    for i in range(1, len(tokens)):
        g = torch.Generator(device=tokens.device)
        g.manual_seed(hash_key * tokens[i - 1].item())
        vocab_permutation = torch.randperm(vocab_size, generator=g)
        greenlist = vocab_permutation[:greenlist_size]
        if tokens[i] in greenlist:
            green_count += 1

    expected = gamma * T
    denom = math.sqrt(T * gamma * (1 - gamma))
    return (green_count - expected) / denom if denom > 0 else 0.0


def detect_smoothed_watermark(text, tokenizer, vocab_size, hash_key=15485863):
    """连续平滑检测: z = (sum_of_greenness - T*0.5) / sqrt(T/12)"""
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    T = len(tokens) - 1
    if T <= 0:
        return 0.0

    greenness_sum = 0.0
    for i in range(1, len(tokens)):
        g = torch.Generator(device=tokens.device)
        g.manual_seed(hash_key * tokens[i - 1].item())
        continuous_greenness = torch.rand(vocab_size, generator=g)
        greenness_sum += continuous_greenness[tokens[i].item()].item()

    expected_sum = T * 0.5
    variance = T * (1.0 / 12.0)
    return (greenness_sum - expected_sum) / math.sqrt(variance) if variance > 0 else 0.0


# ── 主流程 ─────────────────────────────────────────
print("  >>> 生成平滑水印与 KGW 对照文本...")

kgw_processor = LogitsProcessorList([
    KGWLogitsProcessor(vocab_size, gamma=0.5, delta=2.0)
])
smoothed_processor = LogitsProcessorList([
    SmoothedWatermarkLogitsProcessor(vocab_size, delta=3.5)
])

sample_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).head(15)

smooth_results = []
for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="平滑水印 vs KGW"):
    base_text = str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))
    prompt = base_text[:100]
    inputs = detector_tokenizer(prompt, return_tensors="pt").to(device)

    # ── 1. 生成平滑水印文本 ──
    torch.manual_seed(42 + idx)
    outputs_smooth = target_model.generate(
        **inputs, max_new_tokens=200, do_sample=True,
        temperature=0.7, logits_processor=smoothed_processor,
        pad_token_id=detector_tokenizer.eos_token_id,
    )
    smoothed_text = detector_tokenizer.decode(outputs_smooth[0], skip_special_tokens=True)

    z_smooth_before = detect_smoothed_watermark(
        smoothed_text, detector_tokenizer, vocab_size
    )

    # ── 2. SIRA 攻击平滑水印 ──
    masked_smooth = sira_masking(
        smoothed_text, target_model, detector_tokenizer,
        threshold_percentile=20, device=device,
    )
    sira_smooth = sira_t5_infilling(masked_smooth, reference_text=None)
    z_smooth_after = detect_smoothed_watermark(
        sira_smooth, detector_tokenizer, vocab_size
    )

    # ── 3. 生成 KGW 对照 ──
    torch.manual_seed(42 + idx)
    outputs_kgw = target_model.generate(
        **inputs, max_new_tokens=200, do_sample=True,
        temperature=0.7, logits_processor=kgw_processor,
        pad_token_id=detector_tokenizer.eos_token_id,
    )
    kgw_text = detector_tokenizer.decode(outputs_kgw[0], skip_special_tokens=True)

    z_kgw_before = detect_kgw(kgw_text, detector_tokenizer, vocab_size)

    # ── 4. SIRA 攻击 KGW ──
    masked_kgw = sira_masking(
        kgw_text, target_model, detector_tokenizer,
        threshold_percentile=20, device=device,
    )
    sira_kgw = sira_t5_infilling(masked_kgw, reference_text=None)
    z_kgw_after = detect_kgw(sira_kgw, detector_tokenizer, vocab_size)

    smooth_results.append({"Algorithm": "KGW (Hard)", "State": "1_Before SIRA", "Z_Score": z_kgw_before})
    smooth_results.append({"Algorithm": "KGW (Hard)", "State": "2_After SIRA", "Z_Score": z_kgw_after})
    smooth_results.append({"Algorithm": "Smoothed WM", "State": "1_Before SIRA", "Z_Score": z_smooth_before})
    smooth_results.append({"Algorithm": "Smoothed WM", "State": "2_After SIRA", "Z_Score": z_smooth_after})

# ── 汇总 ──────────────────────────────────────────
df_smooth = pd.DataFrame(smooth_results)
df_smooth['State'] = df_smooth['State'].str.split('_', n=1).str[1]

summary_pivot = df_smooth.groupby(['Algorithm', 'State'])['Z_Score'].mean().unstack('State').round(3)
summary_pivot['Retention %'] = (
    summary_pivot['After SIRA'] / summary_pivot['Before SIRA'] * 100
).round(1)

print("\n=== [数据表] 水印平滑 (Smoothing) 对抗 SIRA 效果 ===")
print(summary_pivot.to_string())

# ── 绘图 ──────────────────────────────────────────
fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.2])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

custom_palette = {"Before SIRA": "#2c7bb6", "After SIRA": "#d7191c"}
sns.boxplot(x="Algorithm", y="Z_Score", hue="State", data=df_smooth, ax=ax1,
            width=0.6, showfliers=False, palette=custom_palette)
sns.stripplot(x="Algorithm", y="Z_Score", hue="State", data=df_smooth, ax=ax1,
              dodge=True, color='black', alpha=0.3, size=4)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2,
            label='Detection Threshold (z=4.0)')
ax1.set_title('Test 10: Hard Binary (KGW) vs Continuous Smoothing\n'
              'Survival under SIRA Self-Information Attack',
              fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Z-Score', fontsize=13)
ax1.set_xlabel('')
ax1.legend(loc='upper right')

table_data = summary_pivot.reset_index()
table_data.rename(columns={'Algorithm': 'Strategy'}, inplace=True)
ax2.axis('off')
ax2.set_title('Mean Z-Score Summary', fontsize=13, fontweight='bold', pad=10)
tab = ax2.table(cellText=table_data.values, colLabels=table_data.columns,
                loc='center', cellLoc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(10)
tab.scale(1.2, 2.5)

for j in range(len(table_data.columns)):
    tab[0, j].set_facecolor('#40466e')
    tab[0, j].set_text_props(color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("attack_10_watermark_smoothing.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_10_watermark_smoothing.png")
plt.show()
plt.close()

print("=== 测试10完成 ===\n")
