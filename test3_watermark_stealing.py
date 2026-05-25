"""测试3: 黑盒 API 逆向窃取与伪造攻击 (Watermark Stealing)"""
from test_common import *

print_test_header("黑盒 API 逆向窃取 (Watermark Stealing) 模拟")

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

df_scrub = df_ws[df_ws["Category"] == "Scrubbing"]
summary_table_3_scrub = df_scrub.pivot_table(values='Z_Score', index='Algorithm', columns='State', aggfunc='mean').round(3)

df_spoof = df_ws[df_ws["Category"] == "Spoofing"]
summary_table_3_spoof = df_spoof.pivot_table(values='Z_Score', index='Algorithm', columns='State', aggfunc='mean').round(3)

print("\n=== [数据表] Scrubbing Z-Score 汇总 ===")
print(summary_table_3_scrub.to_string())
print("\n=== [数据表] Spoofing Z-Score 汇总 ===")
print(summary_table_3_spoof.to_string())

fig3 = plt.figure(figsize=(18, 12))
gs3 = fig3.add_gridspec(2, 2, width_ratios=[2, 1.2], height_ratios=[1, 1])
ax3_1 = fig3.add_subplot(gs3[0, 0])
ax3_2 = fig3.add_subplot(gs3[0, 1])
ax3_3 = fig3.add_subplot(gs3[1, 0])
ax3_4 = fig3.add_subplot(gs3[1, 1])

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
print(">>> 图表已保存: attack_3_watermark_stealing.png")
plt.show()
plt.close()
print("=== 测试3完成 ===\n")
