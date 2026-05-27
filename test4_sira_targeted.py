"""测试4: SIRA 三阶段自信息重写攻击 (Self-Information Rewrite Attack)"""
from test_common import *

print_test_header("SIRA 三阶段自信息重写攻击 (Self-Information Rewrite Attack)")

load_attacker()  # SIRA 需要 T5 做 Stage1 改写 + Stage3 填充

sira_results = []
# 使用 percentile 阈值列表 (参考 SIRA 默认 P30)
thresholds = [10, 30, 50]  # P10=保留10%低自信息token, P30=保留30%, P50=保留50%
sample_sira_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).sample(n=min(15, len(df)), random_state=42)

for idx, row in tqdm(sample_sira_df.iterrows(), total=len(sample_sira_df), desc="SIRA 三阶段攻击"):
    for algo in algorithms:
        if algo == "Natural":
            continue
        original_text = row[f"Text_{algo}"]
        z_before = detect_watermark(original_text, algo, detector_tokenizer, vocab_size)
        sira_results.append({"Algorithm": algo, "Stage": "0_Before SIRA", "Z_Score": z_before})

        # ---- Stage 1: 生成参考文本 (改写) ----
        ref_text = sira_generate_reference(original_text)

        # ---- Stage 2+3: 不同阈值下空白化+填充 ----
        for pct in thresholds:
            blank_text = sira_masking(original_text, target_model, detector_tokenizer,
                                      threshold_percentile=pct, device=device)
            attacked_text = sira_t5_infilling(blank_text, reference_text=ref_text)
            z_after = detect_watermark(attacked_text, algo, detector_tokenizer, vocab_size)
            sira_results.append({
                "Algorithm": algo,
                "Stage": f"SIRA P{pct}",
                "Z_Score": z_after
            })

sira_df = pd.DataFrame(sira_results)
pivot = sira_df.pivot_table(index='Algorithm', columns='Stage', values='Z_Score', aggfunc='mean')
# 排序列: Before 在前, 然后按 P10, P30, P50
stage_order = ["0_Before SIRA"] + [f"SIRA P{p}" for p in thresholds]
pivot = pivot.reindex(columns=[c for c in stage_order if c in pivot.columns])
pivot = pivot.round(3)

print("\n=== [数据表] SIRA 三阶段攻击 Z-Score 汇总 ===")
print(pivot.to_string())

# ---- 绘图 ----
fig4 = plt.figure(figsize=(16, 6))
gs4 = fig4.add_gridspec(1, 2, width_ratios=[2, 1.2])
ax4_1 = fig4.add_subplot(gs4[0])
ax4_2 = fig4.add_subplot(gs4[1])

# 箱型图
sns.boxplot(x="Algorithm", y="Z_Score", hue="Stage", data=sira_df, ax=ax4_1, width=0.6, showfliers=False)
sns.stripplot(x="Algorithm", y="Z_Score", hue="Stage", data=sira_df, ax=ax4_1, dodge=True, color='black', alpha=0.3)
ax4_1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
ax4_1.set_title('Test 4: SIRA 3-Stage Self-Information Rewrite Attack', fontsize=14, fontweight='bold')
ax4_1.set_ylabel('Z-Score', fontsize=13)
ax4_1.legend(loc='upper right', fontsize=8)

# 数据表
table4_data = pivot.reset_index()
ax4_2.axis('off')
ax4_2.set_title('SIRA Mean Z-Score by Threshold', fontsize=13, fontweight='bold', pad=10)
tab4 = ax4_2.table(cellText=table4_data.values, colLabels=table4_data.columns, loc='center', cellLoc='center')
tab4.auto_set_font_size(False)
tab4.set_fontsize(10)
tab4.scale(1, 2.5)

plt.tight_layout()
plt.savefig("attack_4_sira_targeted.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_4_sira_targeted.png")
plt.show()
plt.close()
print("=== 测试4完成 ===\n")
