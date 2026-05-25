"""测试2: LLM Rewrite 与 CWRA 跨语言纵深打击"""
from test_common import *

print_test_header("LLM Rewrite 与 CWRA 跨语言纵深打击")

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

print("\n=== [数据表] LLM Rewrite & CWRA Z-Score 汇总 ===")
print(summary_table_2.to_string())

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
print(">>> 图表已保存: attack_2_complex_rewrite.png")
plt.show()
plt.close()
print("=== 测试2完成 ===\n")
