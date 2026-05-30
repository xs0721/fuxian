"""测试18: Transform 水印 — 随机排列 + 逆变换采样 + 置换检验检测"""
from test_common import *
import math

print_test_header("Transform 水印 (Inverse Transform Sampling) — Exp-Watermark")

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


# ── Transform 水印核心 ─────────────────────────────
def _transform_key(generator, n, vocab_size):
    """生成水印密钥: xi (n×1 随机) + pi (随机排列)"""
    pi = torch.randperm(vocab_size, generator=generator)
    xi = torch.rand((n, 1), generator=generator)
    return xi, pi


def _transform_sample(probs, pi, xi):
    """逆变换采样: CDF(permuted_probs) → searchsorted(xi) → pi映射回token"""
    cdf = torch.cumsum(probs.gather(1, pi.unsqueeze(0).expand(probs.shape[0], -1)), dim=1)
    idx = torch.searchsorted(cdf, xi.unsqueeze(0).expand(probs.shape[0], -1, -1))
    idx = idx.clamp(0, cdf.shape[1] - 1)
    return pi[idx.squeeze(-1)].unsqueeze(-1)


def _transform_score(tokens, xi):
    """Transform 检测量: L1距离 ||tokens - xi||₁"""
    if len(tokens) == 0:
        return 0.0
    return torch.norm(tokens.float() - xi[:len(tokens)].squeeze(), p=1).item()


def _transform_generate(model, tokenizer, prompt, n_key, gen_len, seed, device="cuda"):
    """Transform 水印生成 — 自定义采样循环"""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    xi, pi = _transform_key(generator, n_key, vocab_size)

    offset = torch.randint(n_key, size=(1,)).item()
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
        cur_xi = xi[(offset + i) % n_key]
        tok = _transform_sample(probs, pi, cur_xi)
        tok = tok.to(device)
        input_ids = torch.cat([input_ids, tok], dim=-1)
        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True), xi, pi, offset


def _transform_detect(text, tokenizer, xi, pi, offset, vocab_size, seed, n_runs=200):
    """Transform 置换检验检测 — 通过pi逆映射后与xi比较"""
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    if offset > 0:
        tokens = tokens[offset:]
    T = min(len(tokens), len(xi))

    if T < 10:
        return 0.0, 1.0

    test_tokens = tokens[:T].float()
    xi_slice = xi[:T]

    # 通过pi的逆映射将token转回"均匀分布空间"
    inv_pi = torch.argsort(pi).float()
    mapped = inv_pi[test_tokens.long()] / vocab_size  # 归一化到[0,1]

    test_score = _transform_score(mapped, xi_slice)

    # 置换检验
    null_scores = []
    null_gen = torch.Generator()
    null_gen.manual_seed(int(seed + 1))
    for _ in range(n_runs):
        perm = torch.randperm(T, generator=null_gen)
        null_tokens = test_tokens[perm]
        null_mapped = inv_pi[null_tokens.long()] / vocab_size
        null_scores.append(_transform_score(null_mapped, xi_slice))

    null_scores = torch.tensor(null_scores)
    p_value = (null_scores >= test_score).float().mean().item()
    z_score = (null_scores.mean().item() - test_score) / null_scores.std().item() if null_scores.std() > 0 else 0.0

    return z_score, p_value


# ── 主流程 ─────────────────────────────────────────
print("  >>> 对比: KGW vs Transform (逆变换采样, 非LogitsProcessor)")

sample_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).head(6)
transform_results = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Transform水印"):
    prompt = str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))[:60]
    seed = 42 + idx
    N_KEY = 256

    # ── 1. KGW 标准 ──
    inputs = detector_tokenizer(prompt, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out_kgw = target_model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([KGWLogitsProcessor(vocab_size)]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    z_kgw = detect_watermark(detector_tokenizer.decode(out_kgw[0], skip_special_tokens=True),
                              "KGW", detector_tokenizer, vocab_size)

    # ── 2. Transform 水印 ──
    text_trans, xi, pi, offset = _transform_generate(
        target_model, detector_tokenizer, prompt, N_KEY, gen_len=80,
        seed=seed, device=device,
    )
    z_trans, p_trans = _transform_detect(
        text_trans, detector_tokenizer, xi, pi, offset, vocab_size, seed=seed,
    )

    # ── 3. 自然文本 (KGW检测) ──
    z_clean = detect_watermark(
        str(row.get(f"Text_{algorithms[0]}", prompt)), "KGW", detector_tokenizer, vocab_size)

    transform_results.append({"Method": "KGW (logit bias)", "Z_Score": z_kgw})
    transform_results.append({"Method": "Transform (sampling)", "Z_Score": z_trans})
    transform_results.append({"Method": "Clean (no WM)", "Z_Score": z_clean})

# ── 汇总 ──────────────────────────────────────────
df_t = pd.DataFrame(transform_results)
summary = df_t.groupby('Method')['Z_Score'].agg(['mean', 'std']).round(3)

print(f"\n=== [数据表] Transform 水印 (逆变换采样 vs LogitsProcessor) ===")
print("  Transform: 随机排列pi + n×1随机xi → CDF逆变换采样 → L1距离检测")
print("  KGW:       LogitsProcessor在logits上+delta偏置 → 标准multinomial采样\n")
print(summary.to_string())

# ── 绘图 ──────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.2])
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

order = ["KGW (logit bias)", "Transform (sampling)", "Clean (no WM)"]
pal = {"KGW (logit bias)": "#2c7bb6", "Transform (sampling)": "#d7191c",
       "Clean (no WM)": "#999999"}
sns.boxplot(x="Method", y="Z_Score", data=df_t, ax=ax1,
            order=order, palette=pal, width=0.5, showfliers=False)
sns.stripplot(x="Method", y="Z_Score", data=df_t, ax=ax1,
              order=order, color='black', alpha=0.3, size=4)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2,
            label='Detection Threshold (z=4.0)')
ax1.set_title('Test 18: Transform Watermark (Exp-Watermark)\n'
              'Random Permutation + Inverse CDF Sampling (not LogitsProcessor)',
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
plt.savefig("attack_18_transform_watermark.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_18_transform_watermark.png")
plt.show(); plt.close()
print("=== 测试18完成 ===\n")
