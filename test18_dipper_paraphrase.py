"""test18: DIPPER深度改写攻击 (NeurIPS 2023)

论文: Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense
作者: Krishna, Song, Karpinska, Wieting, Iyyer
GitHub: https://github.com/martiansideofthemoon/ai-detection-paraphrases

核心思想:
    使用11B参数级别的DIPPER模型进行深度改写，可以控制：
    - 词汇多样性 (Lexical Diversity)
    - 句式重组 (Order Diversity)
    在保持语义的同时破坏基于Token序列的水印

攻击机制:
    1. 将带水印文本输入DIPPER模型
    2. 控制改写强度参数（lex_diversity, order_diversity）
    3. 输出语义保持但Token序列完全不同的文本
    4. 破坏KGW等依赖前缀哈希的水印

实验设置:
    - 目标: KGW水印文本
    - 攻击强度: 低(20,0) → 中(40,20) → 高(60,40)
    - 评估: 检测率下降、语义保持度、困惑度
"""

from test_common import *

print("=" * 80)
print("Test 18: DIPPER深度改写攻击 (NeurIPS 2023)")
print("=" * 80)

# 使用T5作为DIPPER的简化替代（真实DIPPER是11B模型）
print("\n正在加载改写模型（T5作为DIPPER简化版）...")
paraphraser = AutoModelForSeq2SeqLM.from_pretrained(ATTACKER_MODEL, cache_dir=CACHE_DIR).to(device)
para_tokenizer = AutoTokenizer.from_pretrained(ATTACKER_MODEL, cache_dir=CACHE_DIR)

target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, cache_dir=CACHE_DIR).to(device)
target_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, cache_dir=CACHE_DIR)

# 加载测试数据
df = pd.read_csv(CSV_FILENAME)
kgw_samples = df[df['algorithm'] == 'KGW'].head(50)

print(f"加载了 {len(kgw_samples)} 个KGW水印样本")

# 不同强度的DIPPER改写
diversity_levels = [
    {"name": "低强度", "lex": 20, "order": 0, "temp": 1.0},
    {"name": "中强度", "lex": 40, "order": 20, "temp": 1.2},
    {"name": "高强度", "lex": 60, "order": 40, "temp": 1.5}
]

results = []

for level_config in diversity_levels:
    level_name = level_config["name"]
    lex_diversity = level_config["lex"]
    order_diversity = level_config["order"]
    temperature = level_config["temp"]

    print(f"\n{'='*60}")
    print(f"改写强度: {level_name} (lex={lex_diversity}, order={order_diversity})")
    print(f"{'='*60}")

    detection_before = 0
    detection_after = 0
    semantic_scores = []
    ppl_before_list = []
    ppl_after_list = []

    for idx, row in tqdm(kgw_samples.iterrows(), total=len(kgw_samples), desc=f"{level_name}改写"):
        original_text = row['text']

        # 检测原始水印
        z_score_before = detect_kgw_watermark(original_text, target_tokenizer, vocab_size)
        if z_score_before > 4.0:
            detection_before += 1

        # DIPPER改写（用T5模拟，添加多样性提示）
        # 真实DIPPER会控制词汇和句式多样性
        prompt = f"paraphrase with diversity {lex_diversity}: {original_text}"
        input_ids = para_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = paraphraser.generate(
                input_ids,
                max_length=150,
                num_beams=4,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

        paraphrased_text = para_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 检测改写后的水印
        z_score_after = detect_kgw_watermark(paraphrased_text, target_tokenizer, vocab_size)
        if z_score_after > 4.0:
            detection_after += 1

        # 计算语义相似度（简化版：用Jaccard相似度）
        tokens_orig = set(original_text.lower().split())
        tokens_para = set(paraphrased_text.lower().split())
        if len(tokens_orig | tokens_para) > 0:
            semantic_sim = len(tokens_orig & tokens_para) / len(tokens_orig | tokens_para)
        else:
            semantic_sim = 0.0
        semantic_scores.append(semantic_sim)

        # 计算困惑度
        ppl_before = calculate_perplexity(original_text, target_model, target_tokenizer)
        ppl_after = calculate_perplexity(paraphrased_text, target_model, target_tokenizer)
        ppl_before_list.append(ppl_before)
        ppl_after_list.append(ppl_after)

    # 统计结果
    tpr_before = detection_before / len(kgw_samples)
    tpr_after = detection_after / len(kgw_samples)
    attack_success = (detection_before - detection_after) / max(detection_before, 1)
    avg_semantic = np.mean(semantic_scores)
    avg_ppl_before = np.mean(ppl_before_list)
    avg_ppl_after = np.mean(ppl_after_list)

    print(f"\n{level_name}改写结果:")
    print(f"  改写前检测率: {tpr_before:.2%}")
    print(f"  改写后检测率: {tpr_after:.2%}")
    print(f"  攻击成功率: {attack_success:.2%}")
    print(f"  平均语义保持度: {avg_semantic:.3f}")
    print(f"  困惑度: {avg_ppl_before:.2f} → {avg_ppl_after:.2f}")

    results.append({
        "level": level_name,
        "lex_diversity": lex_diversity,
        "order_diversity": order_diversity,
        "tpr_before": tpr_before,
        "tpr_after": tpr_after,
        "attack_success": attack_success,
        "semantic_similarity": avg_semantic,
        "ppl_before": avg_ppl_before,
        "ppl_after": avg_ppl_after
    })

# 可视化结果
results_df = pd.DataFrame(results)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('DIPPER深度改写攻击效果', fontsize=16, fontweight='bold')

# 1. 检测率对比
ax1 = axes[0, 0]
x = np.arange(len(results_df))
width = 0.35
ax1.bar(x - width/2, results_df['tpr_before'], width, label='改写前', color='#2ecc71', alpha=0.8)
ax1.bar(x + width/2, results_df['tpr_after'], width, label='改写后', color='#e74c3c', alpha=0.8)
ax1.set_xlabel('改写强度', fontsize=12)
ax1.set_ylabel('检测率', fontsize=12)
ax1.set_title('水印检测率变化', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['level'])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. 攻击成功率
ax2 = axes[0, 1]
ax2.plot(results_df['level'], results_df['attack_success'], marker='o', linewidth=2,
         markersize=10, color='#e74c3c')
ax2.fill_between(range(len(results_df)), results_df['attack_success'], alpha=0.3, color='#e74c3c')
ax2.set_xlabel('改写强度', fontsize=12)
ax2.set_ylabel('攻击成功率', fontsize=12)
ax2.set_title('DIPPER攻击成功率', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

# 3. 语义保持度
ax3 = axes[1, 0]
ax3.bar(results_df['level'], results_df['semantic_similarity'], color='#3498db', alpha=0.8)
ax3.axhline(y=0.5, color='red', linestyle='--', label='可接受阈值')
ax3.set_xlabel('改写强度', fontsize=12)
ax3.set_ylabel('语义相似度 (Jaccard)', fontsize=12)
ax3.set_title('语义保持度', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 1)

# 4. 困惑度变化
ax4 = axes[1, 1]
ax4.plot(results_df['level'], results_df['ppl_before'], marker='s', linewidth=2,
         markersize=8, label='改写前', color='#2ecc71')
ax4.plot(results_df['level'], results_df['ppl_after'], marker='o', linewidth=2,
         markersize=8, label='改写后', color='#e74c3c')
ax4.set_xlabel('改写强度', fontsize=12)
ax4.set_ylabel('困惑度 (PPL)', fontsize=12)
ax4.set_title('文本质量变化', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('attack_18_dipper_paraphrase.png', dpi=300, bbox_inches='tight')
print(f"\n✅ 结果图已保存: attack_18_dipper_paraphrase.png")

# 打印论文对比
print("\n" + "="*80)
print("论文复现对比 (Krishna et al., NeurIPS 2023):")
print("="*80)
print(f"{'指标':<25} {'本实验':<20} {'论文报告':<20}")
print("-"*80)
print(f"{'KGW检测率(改写前)':<25} {results[0]['tpr_before']:.1%:<20} {'70.3%':<20}")
print(f"{'KGW检测率(中强度改写后)':<25} {results[1]['tpr_after']:.1%:<20} {'4.6%':<20}")
print(f"{'攻击成功率(中强度)':<25} {results[1]['attack_success']:.1%:<20} {'93.5%':<20}")
print("="*80)
print("✅ Test 18 完成")
