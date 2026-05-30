"""测试13: DITTO 水印伪造攻击 — 知识蒸馏提取EWS → 无密钥栽赃"""
from test_common import *
import math
import hashlib
from collections import defaultdict, Counter

print_test_header("DITTO 水印伪造 (EWS提取 + 无密钥栽赃 vs 随机生成)")

load_attacker()

# ── DITTO 阶段1: "知识蒸馏" — 从水印文本收集token统计 ──
def _ditto_collect_statistics(model, tokenizer, prompts, max_new=80, delta=2.0,
                               hash_key=15485863, device="cuda"):
    """模拟知识蒸馏: 用受害者模型生成水印文本, 收集 (前缀→下一个token→频次) 统计.

    真实的 DITTO 通过 LoRA 微调替身模型学习这些统计模式.
    这里用直接从受害者输出收集来近似, 等效于理想的知识蒸馏.
    """
    kgw_processor = LogitsProcessorList([
        KGWLogitsProcessor(vocab_size, delta=delta, hash_key=hash_key)
    ])

    # d[n] 统计: prefix_tuple -> {token_id: count}
    d_0 = Counter()       # 无条件 (全局)
    d_1 = defaultdict(Counter)  # 前1个token为条件
    d_2 = defaultdict(Counter)  # 前2个token为条件
    prefix_freq = Counter()     # 前缀出现频次 (用于计算权重w)

    for prompt in tqdm(prompts, desc="  收集水印统计"):
        inputs = tokenizer(prompt[:60], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                logits_processor=kgw_processor,
                pad_token_id=tokenizer.eos_token_id,
            )
        tokens = out[0, inputs.input_ids.shape[1]:].tolist()
        # 扩展: 前缀来自prompt最后几个token + 生成token
        all_tokens = inputs.input_ids[0, -2:].tolist() + tokens

        for i in range(len(all_tokens)):
            tok = all_tokens[i]
            d_0[tok] += 1

            if i >= 1:
                p1 = (all_tokens[i-1],)
                d_1[p1][tok] += 1
                prefix_freq[p1] += 1

            if i >= 2:
                p2 = (all_tokens[i-2], all_tokens[i-1])
                d_2[p2][tok] += 1
                prefix_freq[p2] += 1

    return d_0, d_1, d_2, prefix_freq


# ── DITTO 阶段2: EWS提取 — 计算水印信号 ─────────────
def _ditto_extract_ews(d_0, d_1, d_2, prefix_freq, delta=2.0, top_k_per_prefix=100):
    """从统计中提取 EWS (Extracted Watermark Signal).

    真实 DITTO: d_value = 0.5 * min(2, P_after/P_before)
    简化为: d_value = normalized frequency * delta

    同时计算前缀权重 w = 1/(log(freq)/log(f_max))^0.3
    (稀有前缀权重大, 因为它们对水印更"特异性")
    """
    f_max = max(prefix_freq.values()) if prefix_freq else 1

    def _compute_weight(freq):
        if freq <= 0: return 0.0
        return 1.0 / (math.log(freq) / math.log(f_max)) ** 0.3

    # part_0: 全局 EWS (每个token的无条件偏置)
    total_tokens = sum(d_0.values())
    part_0 = {}
    for tok, count in d_0.most_common(top_k_per_prefix):
        prob = count / total_tokens
        part_0[str(tok)] = min(1.0, prob * 30)  # 归一化到合理范围

    # part_1: 1-gram 条件 EWS
    part_1 = {}
    part_1_weight = {}
    for prefix, counter in d_1.items():
        if len(counter) < 3 or prefix_freq[prefix] < 3:
            continue
        total = sum(counter.values())
        top_tokens = counter.most_common(min(top_k_per_prefix // 5, len(counter)))
        part_1[str(list(prefix))] = {
            str(tok): min(1.0, count / total * 30)
            for tok, count in top_tokens
        }
        part_1_weight[str(list(prefix))] = {
            "weight": round(_compute_weight(prefix_freq[prefix]), 4)
        }

    # part_2: 2-gram 条件 EWS
    part_2 = {}
    part_2_weight = {}
    for prefix, counter in d_2.items():
        if len(counter) < 3 or prefix_freq[prefix] < 3:
            continue
        total = sum(counter.values())
        top_tokens = counter.most_common(min(top_k_per_prefix // 5, len(counter)))
        part_2[str(list(prefix))] = {
            str(tok): min(1.0, count / total * 30)
            for tok, count in top_tokens
        }
        part_2_weight[str(list(prefix))] = {
            "weight": round(_compute_weight(prefix_freq[prefix]), 4)
        }

    return part_0, part_1, part_2, part_1_weight, part_2_weight


# ── DITTO 阶段3: EWS注入 LogitsProcessor ───────────
class DITTOEWSLogitsProcessor(LogitsProcessor):
    """DITTO EWS 注入: 在生成时加上提取的水印信号, 实现无密钥栽赃.

    真实 DITTO spoofing.py ReverseWatermarkLogitsProcessor 的三层叠加:
      part_0: scores[tok] += d_0[tok] * delta
      part_1: scores[tok] += d_1[prefix][tok] * delta * w_1[prefix]
      part_2: scores[tok] += d_2[prefix][tok] * delta * w_2[prefix]
    """
    def __init__(self, part_0, part_1, part_2, part_1_weight, part_2_weight,
                 delta=2.0, enable_all=True):
        self.part_0 = part_0
        self.part_1 = part_1
        self.part_2 = part_2
        self.part_1_weight = part_1_weight
        self.part_2_weight = part_2_weight
        self.delta = delta
        self.enable_all = enable_all

    def __call__(self, input_ids, scores):
        # Part 0: 全局偏置
        if self.part_0:
            tok_ids = torch.tensor([int(k) for k in self.part_0.keys()],
                                   device=scores.device)
            bias = torch.tensor([self.part_0[str(int(k))] for k in tok_ids],
                                device=scores.device, dtype=scores.dtype)
            scores[:, tok_ids] += bias * self.delta

        # Part 1: 1-gram 条件偏置
        for b in range(input_ids.shape[0]):
            p1_key = str([input_ids[b, -1].item()])
            if p1_key in self.part_1:
                score_dict = self.part_1[p1_key]
                w = self.part_1_weight.get(p1_key, {}).get("weight", 0.5)
                tok_ids = torch.tensor([int(k) for k in score_dict.keys()],
                                       device=scores.device)
                bias = torch.tensor(list(score_dict.values()),
                                    device=scores.device, dtype=scores.dtype)
                scores[b, tok_ids] += bias * self.delta * w

        # Part 2: 2-gram 条件偏置
        if self.enable_all and input_ids.shape[1] >= 2:
            for b in range(input_ids.shape[0]):
                p2_key = str([input_ids[b, -2].item(), input_ids[b, -1].item()])
                if p2_key in self.part_2:
                    score_dict = self.part_2[p2_key]
                    w = self.part_2_weight.get(p2_key, {}).get("weight", 0.5)
                    tok_ids = torch.tensor([int(k) for k in score_dict.keys()],
                                           device=scores.device)
                    bias = torch.tensor(list(score_dict.values()),
                                        device=scores.device, dtype=scores.dtype)
                    scores[b, tok_ids] += bias * self.delta * w

        return scores


# ── 用户原始"随机生成"对比 ──────────────────────────
def _random_spoofing_generation(prompt, tokenizer, length=80):
    """用户原始代码的逻辑: 随机选token, 用作下界对照"""
    tokens = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
    generated = []
    for _ in range(length):
        generated.append(random.randint(0, vocab_size - 1))
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── 主流程 ─────────────────────────────────────────
print("  >>> DITTO 阶段1: 从受害者模型收集水印统计 (模拟知识蒸馏)...")

GAMMA = 0.5
DELTA = 2.0
HASH_KEY = 15485863

# 用部分文本作为"训练集"来收集EWS统计
sample_prompts = df.dropna(subset=[f"Text_{algorithms[0]}"]).head(20)
train_prompts = [str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))
                 for _, row in sample_prompts.iterrows()]

d_0, d_1, d_2, prefix_freq = _ditto_collect_statistics(
    target_model, detector_tokenizer, train_prompts,
    max_new=80, delta=DELTA, hash_key=HASH_KEY, device=device,
)

print(f"  EWS 统计: d_0={len(d_0)} tokens, d_1={len(d_1)} prefixes, "
      f"d_2={len(d_2)} prefixes")

print("  >>> DITTO 阶段2: 提取 EWS (Extracted Watermark Signal)...")
part_0, part_1, part_2, part_1_w, part_2_w = _ditto_extract_ews(
    d_0, d_1, d_2, prefix_freq, delta=DELTA)

print(f"  EWS 提取完成: part_0={len(part_0)}, part_1={len(part_1)}, "
      f"part_2={len(part_2)}")

print("  >>> DITTO 阶段3: 注入 EWS 进行无密钥栽赃...")

# 用新的prompt测试 (不同于训练集)
test_sample = df.dropna(subset=[f"Text_{algorithms[0]}"]).iloc[25:35]
ews_processor = LogitsProcessorList([
    DITTOEWSLogitsProcessor(part_0, part_1, part_2, part_1_w, part_2_w, delta=DELTA)
])

ditto_results = []
for idx, row in tqdm(test_sample.iterrows(), total=len(test_sample),
                      desc="DITTO 栽赃测试"):
    prompt = str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))[:60]

    # 1. 真正的 KGW 水印文本 (上界)
    kgw_processor = LogitsProcessorList([
        KGWLogitsProcessor(vocab_size, delta=DELTA, hash_key=HASH_KEY)
    ])
    inputs = detector_tokenizer(prompt, return_tensors="pt").to(device)
    torch.manual_seed(42)
    with torch.no_grad():
        out_kgw = target_model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            logits_processor=kgw_processor,
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    kgw_text = detector_tokenizer.decode(out_kgw[0], skip_special_tokens=True)
    z_real_kgw = detect_watermark(kgw_text, "KGW", detector_tokenizer, vocab_size)

    # 2. 自然文本 (无任何水印, 下界)
    torch.manual_seed(42)
    with torch.no_grad():
        out_clean = target_model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    clean_text = detector_tokenizer.decode(out_clean[0], skip_special_tokens=True)
    z_clean = detect_watermark(clean_text, "KGW", detector_tokenizer, vocab_size)

    # 3. DITTO 栽赃: EWS注入 (无密钥!) ← 核心测试
    torch.manual_seed(42)
    with torch.no_grad():
        out_ditto = target_model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            logits_processor=ews_processor,
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    ditto_text = detector_tokenizer.decode(out_ditto[0], skip_special_tokens=True)
    z_ditto = detect_watermark(ditto_text, "KGW", detector_tokenizer, vocab_size)

    # 4. 随机生成 (完全无意义, 对照)
    rand_text = _random_spoofing_generation(prompt, detector_tokenizer, length=80)
    z_rand = detect_watermark(rand_text, "KGW", detector_tokenizer, vocab_size)

    ditto_results.append({"Method": "Real KGW (with key)", "Z_Score": z_real_kgw})
    ditto_results.append({"Method": "Clean (no WM)", "Z_Score": z_clean})
    ditto_results.append({"Method": "DITTO EWS Spoof", "Z_Score": z_ditto})
    ditto_results.append({"Method": "Random (baseline)", "Z_Score": z_rand})

# ── 汇总 ──────────────────────────────────────────
df_ditto = pd.DataFrame(ditto_results)
summary_ditto = df_ditto.groupby('Method')['Z_Score'].agg(['mean', 'std']).round(3)

print(f"\n=== [数据表] DITTO 无密钥水印栽赃 (EWS提取 + Logits注入) ===")
print("  DITTO 链路: 受害者水印输出 → 知识蒸馏(统计收集) → EWS提取")
print("           → Logits注入(无需密钥) → 栽赃文本被检测为水印\n")
print(summary_ditto.to_string())

# ── 绘图 ──────────────────────────────────────────
fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

order = ["Real KGW (with key)", "DITTO EWS Spoof", "Clean (no WM)", "Random (baseline)"]
pal = {"Real KGW (with key)": "#2c7bb6", "DITTO EWS Spoof": "#d7191c",
       "Clean (no WM)": "#fdae61", "Random (baseline)": "#999999"}
sns.boxplot(x="Method", y="Z_Score", data=df_ditto, ax=ax1,
            order=order, palette=pal, width=0.5, showfliers=False)
sns.stripplot(x="Method", y="Z_Score", data=df_ditto, ax=ax1,
              order=order, color='black', alpha=0.3, size=4)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2,
            label='Detection Threshold (z=4.0)')
ax1.set_title('Test 13: DITTO Spoofing Attack — EWS Extraction\n'
              'Knowledge Distillation → Watermark Signal → Keyless Injection',
              fontsize=12, fontweight='bold', pad=15)
ax1.set_ylabel('Z-Score', fontsize=12)
ax1.set_xlabel('')
ax1.legend(loc='upper left')
ax1.tick_params(axis='x', rotation=12)

table_data = summary_ditto.reset_index()
ax2.axis('off')
ax2.set_title('Mean Z-Score', fontsize=13, fontweight='bold', pad=10)
tab = ax2.table(cellText=table_data.values, colLabels=table_data.columns,
                loc='center', cellLoc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(10)
tab.scale(1.2, 2.5)
for j in range(len(table_data.columns)):
    tab[0, j].set_facecolor('#40466e')
    tab[0, j].set_text_props(color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("attack_13_ditto_spoofing.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_13_ditto_spoofing.png")
plt.show()
plt.close()

print("=== 测试13完成 ===\n")
