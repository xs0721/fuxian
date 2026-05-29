"""测试12: De-mark 水印移除攻击 — 成对比较投票 + 逐token对抗性去偏"""
from test_common import *
import math
import hashlib
import struct

print_test_header("De-mark 水印移除 (Token级对抗性去偏 vs T5改写)")

load_attacker()  # T5 用于对比


# ── KGW LogitsProcessor ────────────────────────────
class KGWLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, hash_key=15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            g = torch.Generator(device='cpu')
            g.manual_seed(self.hash_key * input_ids[b, -1].item())
            mask = torch.rand(self.vocab_size, generator=g) < self.gamma
            scores[b, mask.to(scores.device)] += self.delta
        return scores


# ── De-mark 核心组件 ───────────────────────────────
def _ngram_red_green_list(context_tokens, vocab_size, device, hash_key=15485863):
    """KGW 绿名单生成 — 与 run_experiment.py 一致"""
    h = hashlib.sha256()
    h.update(str(hash_key).encode())
    h.update(context_tokens.cpu().numpy().tobytes())
    seed = int.from_bytes(h.digest()[:4], 'big') % (2**31 - 1)

    gamma = 0.5
    green_size = int(vocab_size * gamma)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    perm = torch.randperm(vocab_size, device=device, generator=g)
    return perm[:green_size], perm[green_size:]


def _demark_removal_generate_known_key(model, tokenizer, prompt, max_new=100,
                                        ctx_width=1, delta=2.0, top_k=40, eta=1.0,
                                        hash_key=15485863, vocab_size=50272, device="cuda"):
    """De-mark 去偏生成 (已知密钥版)

    De-mark 核心机制:
      1. 推断绿名单 (论文通过成对比较投票, OPT用已知密钥替代)
      2. 逐token对抗性去偏: logits[绿名单] -= eta * delta
      3. top-k 采样 → 保持文本质量

    与 T5 整段改写的本质区别: De-mark 在 TOKEN 级别精确减去水印偏置,
    而非盲目重写整段文本, 因此理论上能更精确地移除水印同时保留语义.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids
    generated = []
    stats = {"total_tokens": 0, "green_corrected": 0}

    for _step in range(max_new):
        full_ids = torch.cat([prompt_ids,
                              torch.tensor([generated], device=device, dtype=torch.long).view(1, -1)
                              if generated else prompt_ids[:, :0]], dim=1)

        with torch.no_grad():
            logits = model(full_ids).logits[0, -1, :]

        # De-mark 步骤1: 检测绿名单 (已知密钥直接计算)
        ctx_ids = full_ids[0, -ctx_width:]
        cur_green, cur_red = _ngram_red_green_list(ctx_ids, vocab_size, device, hash_key)
        stats["total_tokens"] += 1
        stats["green_corrected"] += len(cur_green)

        # De-mark 步骤2: 对抗性去偏 — 从绿名单logits减去 delta
        debiased = logits.clone()
        debiased[cur_green] -= eta * delta

        # De-mark 步骤3: top-k 采样 (保留语义质量)
        top_k_idx = torch.topk(debiased, k=min(top_k, vocab_size)).indices
        mask = torch.zeros_like(debiased)
        mask[top_k_idx] = 1
        probs = F.softmax(debiased, dim=-1) * mask
        probs /= probs.sum()

        next_token = torch.multinomial(probs, 1).item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    return tokenizer.decode(generated, skip_special_tokens=True), stats


# ── 主流程 ─────────────────────────────────────────
print("  >>> 生成 KGW 水印文本, 对比 T5改写 vs De-mark逐token去偏...")

# KGW 水印参数 (与 run_experiment.py 一致)
GAMMA = 0.5
DELTA = 2.0
CTX_WIDTH = 1  # KGW 使用前1个token的哈希决定绿名单
HASH_KEY = 15485863

sample_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).head(8)

demark_results = []
for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="De-mark vs T5"):
    base_text = str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))
    prompt = base_text[:80]

    # ── 1. 生成 KGW 水印文本 ──
    kgw_processor = LogitsProcessorList([
        KGWLogitsProcessor(vocab_size, gamma=GAMMA, delta=DELTA, hash_key=HASH_KEY)
    ])
    inputs = detector_tokenizer(prompt, return_tensors="pt").to(device)
    torch.manual_seed(42 + idx)
    with torch.no_grad():
        outputs_kgw = target_model.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.7,
            logits_processor=kgw_processor,
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    kgw_text = detector_tokenizer.decode(outputs_kgw[0], skip_special_tokens=True)
    z_kgw = detect_watermark(kgw_text, "KGW", detector_tokenizer, vocab_size)

    demark_results.append({"Method": "KGW (No Attack)", "Z_Score": z_kgw})

    # ── 2. T5 改写攻击 (基线: 盲改写) ──
    t5_text = llm_paraphrase_attack(kgw_text)
    z_t5 = detect_watermark(t5_text, "KGW", detector_tokenizer, vocab_size)
    demark_results.append({"Method": "T5 Paraphrase", "Z_Score": z_t5})

    # ── 3. De-mark 去偏 (eta=1.0: 精确抵消) ──
    torch.manual_seed(42 + idx)
    demark_text_1, _ = _demark_removal_generate_known_key(
        target_model, detector_tokenizer, prompt, max_new=100,
        ctx_width=CTX_WIDTH, delta=DELTA, top_k=40, eta=1.0,
        hash_key=HASH_KEY, vocab_size=vocab_size, device=device,
    )
    z_d1 = detect_watermark(demark_text_1, "KGW", detector_tokenizer, vocab_size)
    demark_results.append({"Method": "De-mark (η=1.0)", "Z_Score": z_d1})

    # ── 4. De-mark 去偏 (eta=1.5: 过度抵消) ──
    torch.manual_seed(42 + idx)
    demark_text_15, _ = _demark_removal_generate_known_key(
        target_model, detector_tokenizer, prompt, max_new=100,
        ctx_width=CTX_WIDTH, delta=DELTA, top_k=40, eta=1.5,
        hash_key=HASH_KEY, vocab_size=vocab_size, device=device,
    )
    z_d15 = detect_watermark(demark_text_15, "KGW", detector_tokenizer, vocab_size)
    demark_results.append({"Method": "De-mark (η=1.5)", "Z_Score": z_d15})

    # ── 5. De-mark 去偏 (eta=2.0: 激进抵消) ──
    torch.manual_seed(42 + idx)
    demark_text_2, _ = _demark_removal_generate_known_key(
        target_model, detector_tokenizer, prompt, max_new=100,
        ctx_width=CTX_WIDTH, delta=DELTA, top_k=40, eta=2.0,
        hash_key=HASH_KEY, vocab_size=vocab_size, device=device,
    )
    z_d2 = detect_watermark(demark_text_2, "KGW", detector_tokenizer, vocab_size)
    demark_results.append({"Method": "De-mark (η=2.0)", "Z_Score": z_d2})

# ── 汇总 ──────────────────────────────────────────
df_demark = pd.DataFrame(demark_results)
summary_demark = df_demark.groupby('Method')['Z_Score'].agg(['mean', 'std']).round(3)

print(f"\n=== [数据表] De-mark 逐Token去偏 vs T5 改写 (KGW δ={DELTA}) ===")
print("  De-mark 核心: 推断绿名单 → 从绿名单logits减去 η·δ → 采样")
print("  (注: 完整De-mark通过成对比较投票推断绿名单, 需Llama等指令模型;")
print("   OPT-125m为基础模型, 此处用已知密钥精确推断绿名单作为上界)\n")
print(summary_demark.to_string())

# ── 绘图 ──────────────────────────────────────────
fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

method_order = ["KGW (No Attack)", "T5 Paraphrase",
                "De-mark (η=1.0)", "De-mark (η=1.5)", "De-mark (η=2.0)"]
palette = {"KGW (No Attack)": "#2c7bb6", "T5 Paraphrase": "#fdae61",
           "De-mark (η=1.0)": "#d7191c", "De-mark (η=1.5)": "#a50026",
           "De-mark (η=2.0)": "#67000d"}
sns.boxplot(x="Method", y="Z_Score", data=df_demark, ax=ax1,
            order=method_order, palette=palette, width=0.5, showfliers=False)
sns.stripplot(x="Method", y="Z_Score", data=df_demark, ax=ax1,
              order=method_order, color='black', alpha=0.3, size=4)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2,
            label='Detection Threshold (z=4.0)')
ax1.set_title('Test 12: De-mark Token-Level Watermark Removal\n'
              f'Green-List Detection → Adversarial Debias logits − η·δ (δ={DELTA})',
              fontsize=13, fontweight='bold', pad=15)
ax1.set_ylabel('Z-Score', fontsize=12)
ax1.set_xlabel('')
ax1.legend(loc='upper right')
ax1.tick_params(axis='x', rotation=15)

table_data = summary_demark.reset_index()
ax2.axis('off')
ax2.set_title('Mean Z-Score Summary', fontsize=13, fontweight='bold', pad=10)
tab = ax2.table(cellText=table_data.values, colLabels=table_data.columns,
                loc='center', cellLoc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(10)
tab.scale(1.2, 2.5)
for j in range(len(table_data.columns)):
    tab[0, j].set_facecolor('#40466e')
    tab[0, j].set_text_props(color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("attack_12_demark_removal.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_12_demark_removal.png")
plt.show()
plt.close()

print("=== 测试12完成 ===\n")
