"""测试15: 检测API查询优选攻击 — 逐token查询检测器, 选择最优候选token"""
from test_common import *
import math

print_test_header("检测API查询攻击 (Public API Query) — NeurIPS 2024 No Free Lunch")

load_detector()

# ── API查询攻击核心 ────────────────────────────────
def _api_query_generate(model, tokenizer, prompt, detector_fn, max_new=60,
                         top_k=10, mode="removal", device="cuda"):
    """对齐 LLM-Watermark-Attacks attack_query.py

    逐token生成, 每步取top-k候选 → 查询检测器评估z-score →
    选z最小(removal)或最大(spoofing)的候选.

    mode: "removal" 躲避检测 | "spoofing" 伪造栽赃
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids
    generated = []

    for _step in range(max_new):
        full_ids = torch.cat([prompt_ids,
                              torch.tensor([generated], device=device, dtype=torch.long).view(1, -1)
                              if generated else prompt_ids[:, :0]], dim=1)

        with torch.no_grad():
            logits = model(full_ids).logits[0, -1, :]

        # Top-k 候选
        _, top_indices = torch.topk(logits, k=min(top_k, len(logits)))
        candidates = top_indices.cpu().tolist()

        # 对每个候选查询检测器
        best_candidate = None
        best_z = float('inf') if mode == "removal" else float('-inf')

        for tok in candidates:
            trial_text = tokenizer.decode(generated + [tok], skip_special_tokens=True)
            # 加入prompt计算完整z-score
            full_text = prompt + " " + trial_text
            z = detector_fn(full_text)

            if mode == "removal" and z < best_z:
                best_z = z; best_candidate = tok
            elif mode == "spoofing" and z > best_z:
                best_z = z; best_candidate = tok

        if best_candidate is None:
            best_candidate = candidates[0]

        if best_candidate == tokenizer.eos_token_id:
            break
        generated.append(best_candidate)

    return tokenizer.decode(generated, skip_special_tokens=True)


# ── 主流程 ─────────────────────────────────────────
print("  >>> 对比: 标准KGW vs API查询移除 vs API查询栽赃")

GAMMA = 0.5; DELTA = 2.0; HASH_KEY = 15485863

# Wrapper for detection with our params
def _detect_fn(text):
    return detect_watermark(text, "KGW", detector_tokenizer, vocab_size,
                            gamma=GAMMA, secret_key=HASH_KEY)

sample_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).head(6)
api_results = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="API查询攻击"):
    prompt = str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))[:50]
    inputs = detector_tokenizer(prompt, return_tensors="pt").to(device)

    # ── 1. 标准 KGW 水印 ──
    torch.manual_seed(42)
    with torch.no_grad():
        out_kgw = target_model.generate(
            **inputs, max_new_tokens=60, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([
                KGWLogitsProcessor(vocab_size, gamma=GAMMA, delta=DELTA, hash_key=HASH_KEY)
            ]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    text_kgw = detector_tokenizer.decode(out_kgw[0], skip_special_tokens=True)
    z_kgw = _detect_fn(text_kgw)

    # ── 2. API查询移除 (选z最小) ──
    torch.manual_seed(42)
    text_removal = _api_query_generate(
        target_model, detector_tokenizer, prompt, _detect_fn,
        max_new=40, top_k=8, mode="removal", device=device,
    )
    z_removal = _detect_fn(text_removal)

    # ── 3. API查询栽赃 (选z最大) ──
    torch.manual_seed(42)
    text_spoof = _api_query_generate(
        target_model, detector_tokenizer, prompt, _detect_fn,
        max_new=40, top_k=8, mode="spoofing", device=device,
    )
    z_spoof = _detect_fn(text_spoof)

    # ── 4. 自然文本 ──
    z_clean = _detect_fn(str(row.get(f"Text_{algorithms[0]}", prompt)))

    api_results.append({"Method": "KGW (standard)", "Z_Score": z_kgw})
    api_results.append({"Method": "API Query Removal", "Z_Score": z_removal})
    api_results.append({"Method": "API Query Spoofing", "Z_Score": z_spoof})
    api_results.append({"Method": "Clean (no WM)", "Z_Score": z_clean})

# ── 汇总 ──────────────────────────────────────────
df_api = pd.DataFrame(api_results)
summary = df_api.groupby('Method')['Z_Score'].agg(['mean', 'std']).round(3)

print(f"\n=== [数据表] 检测API查询优选攻击 (逐token选最优z-score) ===")
print("  对齐: LLM-Watermark-Attacks attack_query.py")
print("  Removal: 每步选z最小的候选 → 躲避检测")
print("  Spoofing: 每步选z最大的候选 → 伪造水印\n")
print(summary.to_string())

# ── 绘图 ──────────────────────────────────────────
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.2])
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

order = ["KGW (standard)", "API Query Removal", "API Query Spoofing", "Clean (no WM)"]
pal = {"KGW (standard)": "#2c7bb6", "API Query Removal": "#d7191c",
       "API Query Spoofing": "#fdae61", "Clean (no WM)": "#999999"}
sns.boxplot(x="Method", y="Z_Score", data=df_api, ax=ax1,
            order=order, palette=pal, width=0.5, showfliers=False)
sns.stripplot(x="Method", y="Z_Score", data=df_api, ax=ax1,
              order=order, color='black', alpha=0.3, size=4)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2,
            label='Detection Threshold (z=4.0)')
ax1.set_title('Test 15: Public API Query Attack\n'
              'Per-Token: Top-k → Query Detector → argmin/argmax z-score',
              fontsize=12, fontweight='bold', pad=15)
ax1.set_ylabel('Z-Score', fontsize=12); ax1.set_xlabel('')
ax1.legend(loc='upper left'); ax1.tick_params(axis='x', rotation=10)
    # X 轴标签旋转
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

table_data = summary.reset_index()
ax2.axis('off'); ax2.set_title('Mean Z-Score', fontsize=13, fontweight='bold', pad=10)
tab = ax2.table(cellText=table_data.values, colLabels=table_data.columns,
                loc='center', cellLoc='center')
tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1.1, 2.5)
for j in range(len(table_data.columns)):
    tab[0, j].set_facecolor('#40466e'); tab[0, j].set_text_props(color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("attack_15_api_query_attack.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_15_api_query_attack.png")
plt.show(); plt.close()
print("=== 测试15完成 ===\n")
