"""测试6: 知识蒸馏(KD)的水印放射性与 WN 中和攻击"""
from test_common import *

print("\n" + "="*60)
print("[测试 6/7] 知识蒸馏(KD)的水印放射性与 WN 中和攻击")
print("="*60)

def get_kgw_green_mask(prefix_token, vocab_size, gamma=0.5, secret_key=15485863):
    torch.manual_seed(secret_key * prefix_token)
    return (torch.rand(vocab_size) < gamma)

def simulate_kd_radioactivity(teacher_text, tokenizer, vocab_size, gamma=0.5, secret_key=15485863):
    tokens = tokenizer.encode(teacher_text, return_tensors="pt")[0].tolist()
    if len(tokens) <= 1: return teacher_text

    student_tokens = [tokens[0]]
    for i in range(1, len(tokens)):
        green_mask = get_kgw_green_mask(tokens[i-1], vocab_size, gamma, secret_key)
        is_green = green_mask[tokens[i]].item()

        if random.random() < 0.85:
            student_tokens.append(tokens[i])
        else:
            if is_green and random.random() < 0.55:
                student_tokens.append(tokens[i])
            elif not is_green and random.random() < 0.40:
                student_tokens.append(tokens[i])
            else:
                student_tokens.append((tokens[i] + random.randint(1, 200)) % vocab_size)

    return tokenizer.decode(student_tokens, skip_special_tokens=True)

def simulate_wn_neutralization(teacher_text, tokenizer, vocab_size, gamma=0.5, secret_key=15485863, delta=3.0):
    tokens = tokenizer.encode(teacher_text, return_tensors="pt")[0].tolist()
    if len(tokens) <= 1: return teacher_text

    stolen_greenlist = {}
    for i in range(1, len(tokens)):
        prev = tokens[i-1]
        if prev not in stolen_greenlist:
            actual_green = get_kgw_green_mask(prev, vocab_size, gamma, secret_key)
            inferred = set()
            for tid in range(vocab_size):
                if actual_green[tid].item():
                    if random.random() < 0.90: inferred.add(tid)
                else:
                    if random.random() < 0.05: inferred.add(tid)
            stolen_greenlist[prev] = inferred

    neutralized_tokens = [tokens[0]]
    for i in range(1, len(tokens)):
        prev_token = tokens[i-1]
        inferred_green = stolen_greenlist.get(prev_token, set())
        current = tokens[i]

        if current in inferred_green:
            if random.random() < gamma:
                neutralized_tokens.append(current)
            else:
                actual_green = get_kgw_green_mask(prev_token, vocab_size, gamma, secret_key)
                candidate = (current + random.randint(1, 500)) % vocab_size
                attempts = 0
                while actual_green[candidate].item() and attempts < 200:
                    candidate = (candidate + 1) % vocab_size
                    attempts += 1
                neutralized_tokens.append(candidate)
        else:
            if random.random() < 0.85:
                neutralized_tokens.append(current)
            else:
                neutralized_tokens.append((current + random.randint(1, 200)) % vocab_size)

    return tokenizer.decode(neutralized_tokens, skip_special_tokens=True)

radioactivity_results = []
sample_kd_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).sample(n=min(20, len(df)), random_state=42)

for idx, row in tqdm(sample_kd_df.iterrows(), total=len(sample_kd_df), desc="知识蒸馏与WN中和"):
    for algo in algorithms:
        if algo == "Natural": continue

        teacher_text = row[f"Text_{algo}"]
        z_teacher = detect_watermark(teacher_text, algo, detector_tokenizer, vocab_size)

        student_radioactive = simulate_kd_radioactivity(teacher_text, detector_tokenizer, vocab_size)
        z_radioactive = detect_watermark(student_radioactive, algo, detector_tokenizer, vocab_size)

        student_wn = simulate_wn_neutralization(teacher_text, detector_tokenizer, vocab_size)
        z_wn = detect_watermark(student_wn, algo, detector_tokenizer, vocab_size)

        radioactivity_results.append({"Algorithm": algo, "State": "1_Teacher Model", "Z_Score": z_teacher})
        radioactivity_results.append({"Algorithm": algo, "State": "2_Student (Radioactive)", "Z_Score": z_radioactive})
        radioactivity_results.append({"Algorithm": algo, "State": "3_Student (WN Neutralized)", "Z_Score": z_wn})

kd_df = pd.DataFrame(radioactivity_results)
summary_table_kd = kd_df.groupby(['Algorithm', 'State'])['Z_Score'].mean().unstack('State').round(3)
summary_table_kd.columns = [col.split('_', 1)[1] for col in summary_table_kd.columns]

print("\n=== [数据表] KD 放射性与 WN 中和 Z-Score 汇总 ===")
print(summary_table_kd.to_string())

kd_df['State'] = kd_df['State'].apply(lambda x: x.split('_', 1)[1])

fig6 = plt.figure(figsize=(16, 6))
gs6 = fig6.add_gridspec(1, 2, width_ratios=[2.5, 1])
ax6_1 = fig6.add_subplot(gs6[0])
ax6_2 = fig6.add_subplot(gs6[1])

hue_order = ["Teacher Model", "Student (Radioactive)", "Student (WN Neutralized)"]
sns.boxplot(x="Algorithm", y="Z_Score", hue="State", data=kd_df, ax=ax6_1, width=0.7, showfliers=False, hue_order=hue_order)
sns.stripplot(x="Algorithm", y="Z_Score", hue="State", data=kd_df, ax=ax6_1, dodge=True, color='black', alpha=0.3, hue_order=hue_order)
ax6_1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Detection Threshold (z=4.0)')
ax6_1.set_title('Task 1: KD Radioactivity vs. Watermark Neutralization (WN)\n(Pan et al., ACL 2025)', fontsize=14, pad=15, fontweight='bold')
ax6_1.set_ylabel('Z-Score', fontsize=13)
ax6_1.legend(loc='upper right')

table6_data = summary_table_kd.reset_index()
table6_data.rename(columns={'Algorithm': 'Algo\n(Z-Score)'}, inplace=True)
ax6_2.axis('off')
ax6_2.set_title('Data Summary (Metric: Z-Score)', fontsize=13, fontweight='bold', pad=10)
table6 = ax6_2.table(cellText=table6_data.values, colLabels=table6_data.columns, loc='center', cellLoc='center')
table6.auto_set_font_size(False)
table6.set_fontsize(10)
table6.scale(1, 2.5)

plt.tight_layout()
plt.savefig("attack_6_kd_radioactivity_wn.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_6_kd_radioactivity_wn.png")
plt.close()
print("=== 测试6完成 ===\n")
