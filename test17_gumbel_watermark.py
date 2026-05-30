"""测试17: Gumbel 水印 — 预生成随机矩阵 + Gumbel-max采样 + 置换检验检测"""
from test_common import *
import math

print_test_header("Gumbel 水印 (Gumbel-max Sampling) — Exp-Watermark")

load_detector()


# ── KGW LogitsProcessor ────────────────────────────
class KGWLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, hash_key=15485863):
        self.vocab_size = vocab_size; self.gamma = gamma
        self.delta = delta; self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            g = torch.Generator(device='cpu')
            g.manual_seed(self.hash_key * input_ids[b, -1].item())
            greenlist_size = int(self.vocab_size * self.gamma)
            perm = torch.randperm(self.vocab_size, generator=g)
            scores[b, perm[:greenlist_size].to(scores.device)] += self.delta
        return scores


# ── Gumbel 水印核心 ────────────────────────────────
def _gumbel_key(generator, n, vocab_size):
    """生成水印密钥: xi (n×V 随机矩阵) + pi (恒等排列)"""
    xi = torch.rand((n, vocab_size), generator=generator)
    pi = torch.arange(vocab_size)
    return xi, pi


def _gumbel_sample(probs, pi, xi):
    """Gumbel-max 采样: argmax(xi^(1/probs))"""
    return torch.argmax(xi ** (1 / probs.gather(1, pi.unsqueeze(0).expand(probs.shape[0], -1))),
                        dim=1).unsqueeze(-1)


def _gumbel_score(tokens, xi):
    """Gumbel 检测统计量: -sum(log(1/(1-xi_sampled)))"""
    xi_samp = xi.gather(-1, tokens.unsqueeze(-1)).squeeze()
    return -torch.sum(torch.log(1 / (1 - xi_samp))).item()


def _gumbel_generate(model, tokenizer, prompt, n_tokens_key, gen_len, seed, device="cuda"):
    """Gumbel 水印生成 — 自定义采样循环 (非LogitsProcessor)"""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    xi, pi = _gumbel_key(generator, n_tokens_key, vocab_size)

    offset = torch.randint(n_tokens_key, size=(1,)).item()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attn = inputs.attention_mask
    past = None

    for i in range(gen_len):
        with torch.no_grad():
            if past is not None:
                output = model(input_ids[:, -1:], past_key_values=past,
                               attention_mask=attn)
            else:
                output = model(input_ids)

        probs = torch.softmax(output.logits[:, -1], dim=-1).cpu()
        tok = _gumbel_sample(probs, pi, xi[(offset + i) % n_tokens_key].unsqueeze(0))
        tok = tok.to(device)
        input_ids = torch.cat([input_ids, tok], dim=-1)
        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True), xi, offset


def _gumbel_detect(text, tokenizer, xi, offset, vocab_size, seed, n_runs=200):
    """Gumbel 置换检验 — 对比真实xi得分 vs 随机打乱token后的得分"""
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    if offset > 0:
        tokens = tokens[offset:]
    T = min(len(tokens), len(xi))

    if T < 10:
        return 0.0, 1.0

    test_tokens = tokens[:T]
    xi_slice = xi[:T]

    test_score = _gumbel_score(test_tokens, xi_slice)

    # 置换检验: 打乱token顺序 → 破坏与xi的关联 → 计算null分数
    null_scores = []
    null_gen = torch.Generator()
    null_gen.manual_seed(int(seed + 1))

    for _ in range(n_runs):
        perm = torch.randperm(T, generator=null_gen)
        null_tokens = test_tokens[perm]
        null_scores.append(_gumbel_score(null_tokens, xi_slice))

    null_scores = torch.tensor(null_scores)
    # Gumbel: 真实水印得分应该更低 (xi^(1/probs) 选择xi值更大的token → 1-xi更小)
    p_value = (null_scores <= test_score).float().mean().item()
    z_score = (test_score - null_scores.mean().item()) / null_scores.std().item() if null_scores.std() > 0 else 0.0

    return z_score, p_value


# ── 主流程 ─────────────────────────────────────────
print("  >>> 对比: KGW vs Gumbel (自定义采样, 非LogitsProcessor)")

sample_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).head(6)
gumbel_results = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Gumbel水印"):
    prompt = str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))[:60]
    seed = 42 + idx
    N_KEY = 256  # xi 矩阵行数

    # ── 1. KGW 标准水印 ──
    inputs = detector_tokenizer(prompt, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out_kgw = target_model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([
                KGWLogitsProcessor(vocab_size)
            ]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    text_kgw = detector_tokenizer.decode(out_kgw[0], skip_special_tokens=True)
    z_kgw = detect_watermark(text_kgw, "KGW", detector_tokenizer, vocab_size)

    # ── 2. Gumbel 水印 ──
    text_gumbel, xi, offset = _gumbel_generate(
        target_model, detector_tokenizer, prompt, N_KEY, gen_len=80,
        seed=seed, device=device,
    )
    z_gumbel, p_gumbel = _gumbel_detect(
        text_gumbel, detector_tokenizer, xi, offset, vocab_size, seed=seed,
    )

    # ── 3. 自然文本 ──
    z_clean = detect_watermark(
        str(row.get(f"Text_{algorithms[0]}", prompt)), "KGW", detector_tokenizer, vocab_size
    )

    gumbel_results.append({"Method": "KGW (logit bias)", "Z_Score": z_kgw})
    gumbel_results.append({"Method": "Gumbel (sampling)", "Z_Score": z_gumbel})
    gumbel_results.append({"Method": "Clean (no WM)", "Z_Score": z_clean})

# ── 汇总 ──────────────────────────────────────────
df_g = pd.DataFrame(gumbel_results)
summary = df_g.groupby('Method')['Z_Score'].agg(['mean', 'std']).round(3)

print(f"\n=== [数据表] Gumbel 水印 (自定义采样 vs LogitsProcessor) ===")
print("  Gumbel: 预生成256×V随机矩阵xi → argmax(xi^(1/probs))逐token采样")
print("  KGW:   LogitsProcessor在logits上+delta偏置 → 标准multinomial采样\n")
print(summary.to_string())

# ── 绘图 ──────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.2])
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

order = ["KGW (logit bias)", "Gumbel (sampling)", "Clean (no WM)"]
pal = {"KGW (logit bias)": "#2c7bb6", "Gumbel (sampling)": "#d7191c",
       "Clean (no WM)": "#999999"}
sns.boxplot(x="Method", y="Z_Score", data=df_g, ax=ax1,
            order=order, palette=pal, width=0.5, showfliers=False)
sns.stripplot(x="Method", y="Z_Score", data=df_g, ax=ax1,
              order=order, color='black', alpha=0.3, size=4)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2,
            label='Detection Threshold (z=4.0)')
ax1.set_title('Test 17: Gumbel Watermark (Exp-Watermark)\n'
              'Pre-generated xi + Gumbel-max Trick (not LogitsProcessor)',
              fontsize=12, fontweight='bold', pad=15)
ax1.set_ylabel('Z-Score', fontsize=12); ax1.set_xlabel('')
ax1.legend(loc='upper right')

table_data = summary.reset_index()
ax2.axis('off'); ax2.set_title('Mean Z-Score', fontsize=13, fontweight='bold', pad=10)
tab = ax2.table(cellText=table_data.values, colLabels=table_data.columns,
                loc='center', cellLoc='center')
tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 2.5)
for j in range(len(table_data.columns)):
    tab[0, j].set_facecolor('#40466e'); tab[0, j].set_text_props(color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("attack_17_gumbel_watermark.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_17_gumbel_watermark.png")
plt.show(); plt.close()
print("=== 测试17完成 ===\n")
