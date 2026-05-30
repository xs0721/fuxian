"""测试19: 不可伪造公开可验证水印 — 神经网络划分器 + LSTM检测器 (Liu et al., ICLR 2024)"""
from test_common import *
import math

print_test_header("不可伪造水印 (Unforgeable UPV) — 神经网络私钥划分器")


# ── 神经网络水印划分器 ──────────────────────────────

def detect_unforgeable(text, tokenizer, vocab_size, net, gamma=0.5):
    """不可伪造水印检测: 使用同一网络重建绿名单 → z-test

    双层检测架构 (代理简化):
      公开检测器: 用网络权重 (私钥持有方) → z-test
      私有检测器: 训练LSTM分类器 (此处省略, 仅演示公开检测器)
    """
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    T = len(tokens) - 1
    if T <= 0: return 0.0

    green_count = 0
    for i in range(1, len(tokens)):
        prev = torch.tensor([tokens[i - 1].item() % 50272], dtype=torch.float32)
        greenness = net(prev)
        effective_gamma = gamma * (0.5 + 0.5 * float(greenness))
        greenlist_size = max(1, int(vocab_size * effective_gamma))

        g = torch.Generator(device='cpu')
        hidden_repr = int(float(greenness) * 1e6) + tokens[i - 1].item()
        g.manual_seed(hidden_repr % (2**31 - 1))
        perm = torch.randperm(vocab_size, generator=g)
        if tokens[i] in perm[:greenlist_size]:
            green_count += 1

    expected = gamma * T
    variance = T * gamma * (1 - gamma)
    return (green_count - expected) / math.sqrt(variance) if variance > 0 else 0.0


# ── 主流程 ─────────────────────────────────────────
print("  >>> 对比: KGW(公开算法) vs Unforgeable(神经网络私钥)")

GAMMA = 0.5; DELTA = 2.0
# 网络实例 (私钥)
net = _SimpleWatermarkNet(context_len=1)

sample_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).head(8)
uf_results = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="不可伪造水印"):
    prompt = str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))[:60]
    inputs = detector_tokenizer(prompt, return_tensors="pt").to(device)

    # 1. KGW (公开算法 — 知道密钥即可伪造)
    torch.manual_seed(42)
    with torch.no_grad():
        out_k = target_model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([KGWLogitsProcessor(vocab_size)]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    z_k = detect_watermark(detector_tokenizer.decode(out_k[0], skip_special_tokens=True),
                            "KGW", detector_tokenizer, vocab_size)

    # 2. Unforgeable (神经网络私钥 — 不知道权重无法伪造)
    torch.manual_seed(42)
    with torch.no_grad():
        out_u = target_model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([
                UnforgeableLogitsProcessor(vocab_size, gamma=GAMMA, delta=DELTA)
            ]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    text_u = detector_tokenizer.decode(out_u[0], skip_special_tokens=True)
    z_real = detect_unforgeable(text_u, detector_tokenizer, vocab_size, net, GAMMA)

    # 3. KGW攻击者尝试伪造 Unforgeable (不知道网络权重, 用KGW公钥替代)
    z_fake_kgw = detect_unforgeable(
        detector_tokenizer.decode(out_k[0], skip_special_tokens=True),
        detector_tokenizer, vocab_size, net, GAMMA)

    uf_results.append({"Method": "KGW (public algo)", "Z_Score": z_k})
    uf_results.append({"Method": "Unforgeable (real)", "Z_Score": z_real})
    uf_results.append({"Method": "KGW fake Unforgeable", "Z_Score": z_fake_kgw})

# ── 汇总 ──────────────────────────────────────────
df_uf = pd.DataFrame(uf_results)
summary = df_uf.groupby('Method')['Z_Score'].agg(['mean', 'std']).round(3)

print(f"\n=== [数据表] 不可伪造水印 (神经网络私钥 vs 公开算法) ===")
print("  Unforgeable: 网络权重=私钥, 攻击者无法精确复现绿名单划分")
print("  KGW fake: 用公开算法生成的文本, 在Unforgeable检测器下失效\n")
print(summary.to_string())

# ── 绘图 ──────────────────────────────────────────
fig = plt.figure(figsize=(13, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.2])
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

order = ["KGW (public algo)", "Unforgeable (real)", "KGW fake Unforgeable"]
pal = {"KGW (public algo)": "#2c7bb6", "Unforgeable (real)": "#1b9e77",
       "KGW fake Unforgeable": "#d7191c"}
sns.boxplot(x="Method", y="Z_Score", data=df_uf, ax=ax1,
            order=order, palette=pal, width=0.5, showfliers=False)
sns.stripplot(x="Method", y="Z_Score", data=df_uf, ax=ax1,
              order=order, color='black', alpha=0.3, size=4)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2)
ax1.set_title('Test 19: Unforgeable Publicly Verifiable Watermark (ICLR 2024)\n'
              'Neural Network Private Key → Attacker Cannot Forge Greenlist',
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
plt.savefig("attack_19_unforgeable_watermark.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_19_unforgeable_watermark.png")
plt.show(); plt.close()
print("=== 测试19完成 ===\n")
