"""测试1: 物理层扰动 (删词 & 删字符) 鲁棒性退化测试"""
from test_common import *

print_test_header("物理层扰动 (删词 & 删字符) 鲁棒性退化测试")

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

print("\n=== [数据表] Word Drop Z-Score 汇总 ===")
print(summary_table_1_w.to_string())
print("\n=== [数据表] Character Removal Z-Score 汇总 ===")
print(summary_table_1_c.to_string())

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1.2], height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

markers = ['o', 's', '^', 'D', 'v', 'p']
for i, algo in enumerate(algorithms):
    ax1.plot([r * 100 for r in attack_ratios], results_word_drop[algo], marker=markers[i % len(markers)], linewidth=2.5, markersize=8, label=algo)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
ax1.set_title('Test 1A: Robustness under Word Drop', fontsize=14, fontweight='bold')
ax1.set_ylabel('Average Z-Score', fontsize=13)
all_w_scores = [score for scores in results_word_drop.values() for score in scores]
ax1.set_ylim(min(min(all_w_scores) - 0.5, -0.5), max(max(all_w_scores) + 0.5, 4.5))
ax1.legend(loc='lower left')

for i, algo in enumerate(algorithms):
    ax3.plot([r * 100 for r in attack_ratios], results_char_drop[algo], marker=markers[i % len(markers)], linewidth=2.5, markersize=8, label=algo)
ax3.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
ax3.set_title('Test 1B: Robustness under Character Removal', fontsize=14, fontweight='bold')
ax3.set_xlabel('Modification Ratio (%)', fontsize=13)
ax3.set_ylabel('Average Z-Score', fontsize=13)
all_c_scores = [score for scores in results_char_drop.values() for score in scores]
ax3.set_ylim(min(min(all_c_scores) - 0.5, -0.5), max(max(all_c_scores) + 0.5, 4.5))
ax3.legend(loc='lower left')

t1_w_data = summary_table_1_w.reset_index()
t1_w_data.rename(columns={'index': 'Algo(Z)'}, inplace=True)
ax2.axis('off')
ax2.set_title('Word Drop Summary', fontsize=13, fontweight='bold', pad=10)
tab1 = ax2.table(cellText=t1_w_data.values, colLabels=t1_w_data.columns, loc='center', cellLoc='center')
tab1.auto_set_font_size(False); tab1.set_fontsize(11); tab1.scale(1, 2.5)

t1_c_data = summary_table_1_c.reset_index()
t1_c_data.rename(columns={'index': 'Algo(Z)'}, inplace=True)
ax4.axis('off')
ax4.set_title('Character Removal Summary', fontsize=13, fontweight='bold', pad=10)
tab2 = ax4.table(cellText=t1_c_data.values, colLabels=t1_c_data.columns, loc='center', cellLoc='center')
tab2.auto_set_font_size(False); tab2.set_fontsize(11); tab2.scale(1, 2.5)

plt.tight_layout()
plt.savefig("attack_1_physical_perturbation.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_1_physical_perturbation.png")
plt.show()
plt.close()
print("=== 测试1完成 ===\n")
