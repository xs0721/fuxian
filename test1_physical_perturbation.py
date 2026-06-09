"""测试1: 物理层扰动 (删词 & 删字符 & 同义词替换 & 复制粘贴) 鲁棒性退化测试"""
from test_common import *

print_test_header("物理层扰动 (删词 & 删字符 & 同义词替换 & 复制粘贴) 鲁棒性退化测试")

attack_ratios = [0.0, 0.1, 0.3, 0.5]
results_word_drop = {algo: [] for algo in algorithms}
results_char_drop = {algo: [] for algo in algorithms}
results_syn_sub = {algo: [] for algo in algorithms}
results_copy_paste = {algo: [] for algo in algorithms}

# CopyPaste 参考文本: 使用跨算法的文本作为"无 watermark"参考
# 对于每个算法的文本, 取其他算法中同一样本的文本作为参考
algo_list_for_ref = [a for a in algorithms if a != "Natural"]

# 采样加速(使用统一索引保证各算法间对齐)
sample_idx = df.sample(n=min(50, len(df)), random_state=42).index
sample_texts = {algo: df.loc[sample_idx, f"Text_{algo}"].tolist()
                for algo in algorithms if algo != "Natural"}

for ratio in attack_ratios:
    for algo in algorithms:
        if algo == "Natural":
            for key in [results_word_drop, results_char_drop, results_syn_sub, results_copy_paste]:
                key[algo].append(0.0)
            continue

        texts = sample_texts[algo]

        w_z = [detect_watermark(simulate_word_drop(text, ratio), algo, detector_tokenizer, vocab_size) for text in texts]
        results_word_drop[algo].append(sum(w_z) / len(w_z) if w_z else 0.0)

        c_z = [detect_watermark(character_removal_attack(text, ratio), algo, detector_tokenizer, vocab_size) for text in texts]
        results_char_drop[algo].append(sum(c_z) / len(c_z) if c_z else 0.0)

        s_z = [detect_watermark(synonym_substitution_attack(text, ratio), algo, detector_tokenizer, vocab_size) for text in texts]
        results_syn_sub[algo].append(sum(s_z) / len(s_z) if s_z else 0.0)

        # CopyPaste: 使用同一样本的另一算法文本作为"无watermark"参考
        ref_algo = algo_list_for_ref[(algo_list_for_ref.index(algo) + 1) % len(algo_list_for_ref)]
        ref_texts = sample_texts[ref_algo]  # 同索引, 同一组样本
        cp_z = [detect_watermark(copy_paste_attack(texts[i], ref_texts[i], ratio), algo, detector_tokenizer, vocab_size) for i in range(len(texts))]
        results_copy_paste[algo].append(sum(cp_z) / len(cp_z) if cp_z else 0.0)

# ---- 数据表 ----
def build_summary(results_dict):
    df_s = pd.DataFrame(results_dict)
    df_s.index = [f"{int(r*100)}%" for r in attack_ratios]
    return df_s.T.round(3)

summary_w = build_summary(results_word_drop)
summary_c = build_summary(results_char_drop)
summary_s = build_summary(results_syn_sub)
summary_p = build_summary(results_copy_paste)

print("\n=== [数据表] Word Drop Z-Score 汇总 ===")
print(summary_w.to_string())
print("\n=== [数据表] Character Removal Z-Score 汇总 ===")
print(summary_c.to_string())
print("\n=== [数据表] Synonym Substitution Z-Score 汇总 (ASW) ===")
print(summary_s.to_string())
print("\n=== [数据表] CopyPaste Z-Score 汇总 (ASW) ===")
print(summary_p.to_string())

# ---- 绘图: 4行x2列 ----
fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(4, 2, width_ratios=[2, 1.2], height_ratios=[1, 1, 1, 1])

axes = [
    (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), "A: Word Drop", results_word_drop, summary_w),
    (fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), "B: Character Removal", results_char_drop, summary_c),
    (fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), "C: Synonym Substitution (ASW)", results_syn_sub, summary_s),
    (fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), "D: Copy-Paste Attack (ASW)", results_copy_paste, summary_p),
]

markers = ['o', 's', '^', 'D', 'v', 'p']
for ax_line, ax_table, label, results, summary in axes:
    for i, algo in enumerate(algorithms):
        if algo == "Natural":
            continue
        ax_line.plot([r * 100 for r in attack_ratios], results[algo],
                     marker=markers[i % len(markers)], linewidth=2.5, markersize=8, label=algo)
    ax_line.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
    ax_line.set_title(f'Test 1{label}', fontsize=13, fontweight='bold')
    ax_line.set_ylabel('Average Z-Score', fontsize=12)
    ax_line.set_xlabel('Attack Ratio (%)', fontsize=12)
    all_s = [s for scores in results.values() for s in scores]
    ax_line.set_ylim(min(min(all_s) - 0.5, -0.5), max(max(all_s) + 0.5, 4.5))
    ax_line.legend(loc='lower left', fontsize=9)
    # X 轴标签旋转
    ax_line.tick_params(axis='x', rotation=0)  # 折线图的 X 轴是数字，不需要旋转

    t_data = summary.reset_index()
    t_data.rename(columns={'index': 'Algo(Z)'}, inplace=True)
    ax_table.axis('off')
    ax_table.set_title(f'{label.split(": ", 1)[1]} Summary', fontsize=12, fontweight='bold', pad=8)
    tab = ax_table.table(cellText=t_data.values, colLabels=t_data.columns, loc='center', cellLoc='center')
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 2.5)

plt.tight_layout()
plt.savefig("attack_1_physical_perturbation.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_1_physical_perturbation.png")
plt.show()
plt.close()
print("=== 测试1完成 ===\n")
