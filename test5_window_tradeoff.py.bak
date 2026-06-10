"""测试5: 哈希窗口大小 (h) 的多维博弈权衡分析"""
from test_common import *

print_test_header("哈希窗口大小 (h) 的多维博弈权衡分析")

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
print("\n=== [数据表] 窗口大小权衡分析 ===")
print(df_tradeoff.to_string(index=False))

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
print(">>> 图表已保存: attack_5_window_tradeoff.png")
plt.show()
plt.close()
print("=== 测试5完成 ===\n")
