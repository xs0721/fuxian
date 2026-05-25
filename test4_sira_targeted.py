"""测试4: SIRA 高熵靶向重写降维打击"""
from test_common import *

print_test_header("SIRA 高熵靶向重写降维打击")

load_attacker()  # SIRA 需要 T5 做文本填充

sira_results = []
sample_sira_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).sample(n=min(20, len(df)), random_state=42)

for idx, row in tqdm(sample_sira_df.iterrows(), total=len(sample_sira_df), desc="SIRA 定向手术刀攻击"):
    for algo in algorithms:
        if algo == "Natural": continue
        original_text = row[f"Text_{algo}"]
        z_before = detect_watermark(original_text, algo, detector_tokenizer, vocab_size)
        masked_text = sira_masking(original_text, target_model, detector_tokenizer, mask_ratio=0.2, device=device)
        sira_attacked_text = sira_t5_infilling(masked_text)
        z_after = detect_watermark(sira_attacked_text, algo, detector_tokenizer, vocab_size)

        sira_results.append({"Algorithm": algo, "State": "1_Before SIRA", "Z_Score": z_before})
        sira_results.append({"Algorithm": algo, "State": "2_After SIRA (20% Edit)", "Z_Score": z_after})

sira_df = pd.DataFrame(sira_results)
summary_table_4 = sira_df.groupby(['Algorithm', 'State'])['Z_Score'].mean().unstack('State').round(3)
summary_table_4.columns = [col.split('_')[1] for col in summary_table_4.columns]

print("\n=== [数据表] SIRA Z-Score 汇总 ===")
print(summary_table_4.to_string())

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
print(">>> 图表已保存: attack_4_sira_targeted.png")
plt.show()
plt.close()
print("=== 测试4完成 ===\n")
