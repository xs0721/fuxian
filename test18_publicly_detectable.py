"""测试18: 公开可检测水印 — 非对称密钥架构, 公钥验证不可伪造 (Christ et al., 2023)"""
from test_common import *
import hashlib

print_test_header("公开可检测水印 (Publicly-Detectable) — 非对称密钥/公钥验证")


# ── 非对称水印: 私钥生成 + 公钥检测 ─────────────────

def detect_publicly_detectable(text, tokenizer, vocab_size, public_salt=9876543,
                                gamma=0.5):
    """公钥检测: 用公钥重建绿名单 → z-test

    关键: 公钥绿名单 ≠ 私钥绿名单, 但有统计重叠 (gamma 比例).
    这使检测成为可能 (z-score > 0), 但伪造困难 (无法精确重建私钥绿名单)
    """
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    T = len(tokens) - 1
    if T <= 0: return 0.0

    processor = PubliclyDetectableProcessor(vocab_size, gamma=gamma, secret_key=0,
                                             public_salt=public_salt)
    green_count = 0
    for i in range(1, len(tokens)):
        greenlist = processor._get_greenlist(tokens[i - 1].item(), public_salt)
        if tokens[i] in greenlist:
            green_count += 1

    expected = gamma * T
    variance = T * gamma * (1 - gamma)
    return (green_count - expected) / math.sqrt(variance) if variance > 0 else 0.0


# ── 主流程 ─────────────────────────────────────────
print("  >>> 对比: KGW(对称密钥) vs Publicly-Detectable(非对称密钥)")

GAMMA = 0.5; DELTA = 2.0; SECRET = 15485863; PUBLIC = 9876543

sample_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).head(8)
pd_results = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="公开可检测水印"):
    prompt = str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))[:60]
    inputs = detector_tokenizer(prompt, return_tensors="pt").to(device)

    # 1. KGW (对称: 同一密钥生成+检测)
    torch.manual_seed(42)
    with torch.no_grad():
        out_k = target_model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([KGWLogitsProcessor(vocab_size)]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    text_k = detector_tokenizer.decode(out_k[0], skip_special_tokens=True)
    z_k = detect_watermark(text_k, "KGW", detector_tokenizer, vocab_size)

    # 2. Publicly-Detectable (非对称: 私钥生成, 公钥检测)
    torch.manual_seed(42)
    with torch.no_grad():
        out_p = target_model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([
                PubliclyDetectableProcessor(vocab_size, gamma=GAMMA, delta=DELTA,
                                             secret_key=SECRET, public_salt=PUBLIC)
            ]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    text_p = detector_tokenizer.decode(out_p[0], skip_special_tokens=True)

    # 公钥检测
    z_pub = detect_publicly_detectable(text_p, detector_tokenizer, vocab_size,
                                        public_salt=PUBLIC, gamma=GAMMA)
    # 伪造者尝试 (不知道私钥, 只有公钥) — 用公钥生成
    torch.manual_seed(42)
    with torch.no_grad():
        out_f = target_model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([
                PubliclyDetectableProcessor(vocab_size, gamma=GAMMA, delta=DELTA,
                                             secret_key=PUBLIC, public_salt=PUBLIC)
            ]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    text_f = detector_tokenizer.decode(out_f[0], skip_special_tokens=True)
    z_fake = detect_publicly_detectable(text_f, detector_tokenizer, vocab_size,
                                         public_salt=PUBLIC, gamma=GAMMA)

    pd_results.append({"Method": "KGW (symmetric)", "Z_Score": z_k})
    pd_results.append({"Method": "Public-Detect (real)", "Z_Score": z_pub})
    pd_results.append({"Method": "Public-Detect (fake)", "Z_Score": z_fake})

# ── 汇总 ──────────────────────────────────────────
df_pd = pd.DataFrame(pd_results)
summary = df_pd.groupby('Method')['Z_Score'].agg(['mean', 'std']).round(3)

print(f"\n=== [数据表] 公开可检测水印 (非对称密钥架构) ===")
print("  Real: 私钥生成 + 公钥检测 = 可检测")
print("  Fake: 公钥生成 + 公钥检测 = 难检测 (无精确绿名单)\n")
print(summary.to_string())

# ── 绘图 ──────────────────────────────────────────
fig = plt.figure(figsize=(13, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.2])
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

order = ["KGW (symmetric)", "Public-Detect (real)", "Public-Detect (fake)"]
pal = {"KGW (symmetric)": "#2c7bb6", "Public-Detect (real)": "#1b9e77",
       "Public-Detect (fake)": "#d7191c"}
sns.boxplot(x="Method", y="Z_Score", data=df_pd, ax=ax1,
            order=order, palette=pal, width=0.5, showfliers=False)
sns.stripplot(x="Method", y="Z_Score", data=df_pd, ax=ax1,
              order=order, color='black', alpha=0.3, size=4)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2)
ax1.set_title('Test 18: Publicly-Detectable Watermark\n'
              'Asymmetric Key: Private gen → Public verify, fake fails',
              fontsize=11, fontweight='bold', pad=15)
ax1.set_ylabel('Z-Score', fontsize=12); ax1.set_xlabel('')
ax1.tick_params(axis='x', rotation=10)

table_data = summary.reset_index()
ax2.axis('off'); ax2.set_title('Mean Z-Score', fontsize=13, fontweight='bold', pad=10)
tab = ax2.table(cellText=table_data.values, colLabels=table_data.columns,
                loc='center', cellLoc='center')
tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 2.5)
for j in range(len(table_data.columns)):
    tab[0, j].set_facecolor('#40466e'); tab[0, j].set_text_props(color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("attack_18_publicly_detectable.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_18_publicly_detectable.png")
plt.show(); plt.close()
print("=== 测试18完成 ===\n")
