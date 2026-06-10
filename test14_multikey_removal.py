"""测试14: 多密钥水印稀释攻击 — N个密钥同时施加, 单密钥检测信号被1/N稀释"""
from test_common import *
import math

print_test_header("多密钥水印稀释 (Multiple Keys Removal) — NeurIPS 2024 No Free Lunch")

# ── 多密钥 LogitsProcessor ────────────────────────
class MultiKeyLogitsProcessor(LogitsProcessor):
    """对齐 LLM-Watermark-Attacks watermark_processor.py WatermarkLogitsProcessor

    核心: 同时用 N 个不同 hash_key 生成 N 个绿名单,
    每个绿名单分别 +delta, 最终 logits 取 N 个版本的平均。
    结果: 对任意单个密钥检测器, 仅有约 1/N 的 token 来自"它的"绿名单,
    z-score 被稀释至约 1/sqrt(N).
    """
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, num_keys=5,
                 base_hash_key=15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        # 与论文一致的固定素数列表
        self.hash_keys = [15485863, 5823667, 68425619, 1107276647, 751783477,
                          563167303, 440817757, 368345293, 259336153, 131807699,
                          65535, 13, 17, 19, 23, 29, 31, 37, 41, 43,
                          47, 53, 59, 61, 67][:num_keys]
        self.generators = [torch.Generator(device='cpu') for _ in self.hash_keys]

    def __call__(self, input_ids, scores):
        greenlist_size = int(self.vocab_size * self.gamma)
        score_accum = torch.zeros_like(scores)

        for b in range(input_ids.shape[0]):
            prev = input_ids[b, -1].item()
            for k, g in enumerate(self.hash_keys):
                self.generators[k].manual_seed(g * prev)
                perm = torch.randperm(self.vocab_size, generator=self.generators[k])
                greenlist = perm[:greenlist_size]
                tmp = scores[b].clone()
                tmp[greenlist.to(scores.device)] += self.delta
                score_accum[b] += tmp

            score_accum[b] /= len(self.hash_keys)  # 平均 → 稀释

        return score_accum


# ── 主流程 ─────────────────────────────────────────
print("  >>> 对比: 单密钥 KGW vs 多密钥稀释 (N=2,5,10)")

GAMMA = 0.5; DELTA = 2.0; HASH_KEY = 15485863

sample_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).head(10)
multikey_results = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="多密钥稀释"):
    prompt = str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))[:60]
    inputs = detector_tokenizer(prompt, return_tensors="pt").to(device)

    # ── 1. 单密钥 KGW (上界) ──
    torch.manual_seed(42)
    with torch.no_grad():
        out_1 = target_model.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([
                KGWLogitsProcessor(vocab_size, gamma=GAMMA, delta=DELTA, hash_key=HASH_KEY)
            ]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    text_1 = detector_tokenizer.decode(out_1[0], skip_special_tokens=True)
    z_1 = detect_watermark(text_1, "KGW", detector_tokenizer, vocab_size)

    # ── 2. 多密钥 N=2 ──
    torch.manual_seed(42)
    with torch.no_grad():
        out_2 = target_model.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([
                MultiKeyLogitsProcessor(vocab_size, gamma=GAMMA, delta=DELTA, num_keys=2)
            ]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    text_2 = detector_tokenizer.decode(out_2[0], skip_special_tokens=True)
    z_2 = detect_watermark(text_2, "KGW", detector_tokenizer, vocab_size)

    # ── 3. 多密钥 N=5 ──
    torch.manual_seed(42)
    with torch.no_grad():
        out_5 = target_model.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([
                MultiKeyLogitsProcessor(vocab_size, gamma=GAMMA, delta=DELTA, num_keys=5)
            ]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    text_5 = detector_tokenizer.decode(out_5[0], skip_special_tokens=True)
    z_5 = detect_watermark(text_5, "KGW", detector_tokenizer, vocab_size)

    # ── 4. 多密钥 N=10 ──
    torch.manual_seed(42)
    with torch.no_grad():
        out_10 = target_model.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([
                MultiKeyLogitsProcessor(vocab_size, gamma=GAMMA, delta=DELTA, num_keys=10)
            ]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    text_10 = detector_tokenizer.decode(out_10[0], skip_special_tokens=True)
    z_10 = detect_watermark(text_10, "KGW", detector_tokenizer, vocab_size)

    # ── 5. 无密钥 (下界) ──
    z_clean = detect_watermark(
        str(row.get(f"Text_{algorithms[0]}", prompt)), "KGW", detector_tokenizer, vocab_size
    )

    multikey_results.append({"Method": "KGW (N=1)", "Z_Score": z_1})
    multikey_results.append({"Method": "MultiKey (N=2)", "Z_Score": z_2})
    multikey_results.append({"Method": "MultiKey (N=5)", "Z_Score": z_5})
    multikey_results.append({"Method": "MultiKey (N=10)", "Z_Score": z_10})
    multikey_results.append({"Method": "Clean (no WM)", "Z_Score": z_clean})

# ── 汇总 ──────────────────────────────────────────
df_mk = pd.DataFrame(multikey_results)
summary = df_mk.groupby('Method')['Z_Score'].agg(['mean', 'std']).round(3)
# 理论稀释因子: z ~ z_orig / sqrt(N)
n_keys = [1, 2, 5, 10]
theory = [summary.loc["KGW (N=1)", "mean"] / math.sqrt(n) for n in n_keys]
summary['Theory ~z/√N'] = ['—'] + [f"{t:.1f}" for t in theory[1:]] + ['—']

print(f"\n=== [数据表] 多密钥水印稀释 (理论: z_N ≈ z_1 / √N) ===")
print("  对齐: LLM-Watermark-Attacks attack_multiple_keys.py")
print(summary.to_string())

# ── 绘图 ──────────────────────────────────────────
fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

order = ["KGW (N=1)", "MultiKey (N=2)", "MultiKey (N=5)", "MultiKey (N=10)", "Clean (no WM)"]
pal = {"KGW (N=1)": "#2c7bb6", "MultiKey (N=2)": "#fdae61",
       "MultiKey (N=5)": "#f46d43", "MultiKey (N=10)": "#d7191c",
       "Clean (no WM)": "#999999"}
sns.boxplot(x="Method", y="Z_Score", data=df_mk, ax=ax1,
            order=order, palette=pal, width=0.5, showfliers=False)
sns.stripplot(x="Method", y="Z_Score", data=df_mk, ax=ax1,
              order=order, color='black', alpha=0.3, size=4)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2,
            label='Detection Threshold (z=4.0)')
ax1.set_title('Test 14: Multiple Keys Watermark Dilution\n'
              'N Keys → z-score ≈ z₁/√N (NeurIPS 2024)',
              fontsize=12, fontweight='bold', pad=15)
ax1.set_ylabel('Z-Score', fontsize=12); ax1.set_xlabel('')
ax1.legend(loc='upper right'); ax1.tick_params(axis='x', rotation=10)
    # X 轴标签旋转
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

table_data = summary.reset_index()
ax2.axis('off'); ax2.set_title('Mean Z-Score', fontsize=13, fontweight='bold', pad=10)
tab = ax2.table(cellText=table_data.values, colLabels=table_data.columns,
                loc='center', cellLoc='center')
tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1.1, 2.2)
for j in range(len(table_data.columns)):
    tab[0, j].set_facecolor('#40466e'); tab[0, j].set_text_props(color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("attack_14_multikey_dilution.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_14_multikey_dilution.png")
plt.show(); plt.close()
print("=== 测试14完成 ===\n")
