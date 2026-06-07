"""test23: ICW上下文水印测试

论文: In-Context Watermarks for Large Language Models (arXiv 2025)
作者: Yepeng Liu, Xuandong Zhao, Christopher Kruegel, Dawn Song, Yuheng Bu
GitHub: https://github.com/yepengliu/In-Context-Watermarks

核心思想:
    传统水印需要访问模型logits，ICW完全黑盒
    通过系统提示词引导模型生成带特定特征的文本

ICW优势:
    - 适用于任何LLM（包括GPT-4等闭源API）
    - 无需访问模型内部
    - 易于部署

ICW局限:
    - 依赖模型对指令的遵循能力
    - 小模型（如OPT-125m）效果可能有限
    - 鲁棒性低于logits层水印

实验设计:
    1. 生成无水印文本（baseline）
    2. 生成ICW Initials水印文本（偏向特定首字母）
    3. 生成ICW Lexical水印文本（偏向特定词汇）
    4. 对比检测Z-score
"""

from test_common import *
from run_experiment import ICWInitialsWatermark, ICWLexicalWatermark

print("=" * 80)
print("Test 23: ICW上下文水印测试 (arXiv 2025)")
print("=" * 80)

# 加载模型
print("\n正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, cache_dir=CACHE_DIR).to(device)
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, cache_dir=CACHE_DIR)
print("✅ 模型加载完成")

# 测试提示词
test_prompts = [
    "Explain what machine learning is.",
    "Describe the benefits of artificial intelligence.",
    "What are the challenges in natural language processing?",
    "How does deep learning work?",
    "Explain the concept of neural networks."
]

print(f"\n实验设置:")
print(f"  测试样本数: {len(test_prompts)}")
print(f"  水印类型: ICW Initials + ICW Lexical")
print(f"  模型: {TARGET_MODEL}")

# 初始化ICW水印
icw_initials = ICWInitialsWatermark()
icw_lexical = ICWLexicalWatermark()

results = {
    'no_watermark': {'z_scores': [], 'texts': []},
    'icw_initials': {'z_scores': [], 'texts': []},
    'icw_lexical': {'frequencies': [], 'texts': []}
}

print("\n" + "="*80)
print("开始实验: 生成不同水印强度的文本")
print("="*80)

for idx, prompt in enumerate(test_prompts):
    print(f"\n[{idx+1}/{len(test_prompts)}] 提示词: {prompt}")

    # === 基线：无水印 ===
    print("  生成无水印文本...")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    text_no_wm = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

    # 检测
    z_no_wm = icw_initials.detect_watermark(text_no_wm)
    results['no_watermark']['z_scores'].append(z_no_wm)
    results['no_watermark']['texts'].append(text_no_wm[:100])

    print(f"    无水印Z-score: {z_no_wm:.2f}")

    # === ICW Initials水印 ===
    print("  生成ICW Initials水印文本...")
    text_icw_initials = icw_initials.generate_with_watermark(
        model, tokenizer, prompt, device=device, max_new_tokens=50, strength='medium'
    )

    z_icw_initials = icw_initials.detect_watermark(text_icw_initials)
    results['icw_initials']['z_scores'].append(z_icw_initials)
    results['icw_initials']['texts'].append(text_icw_initials[:100])

    print(f"    ICW Initials Z-score: {z_icw_initials:.2f}")

    # === ICW Lexical水印 ===
    print("  生成ICW Lexical水印文本...")
    text_icw_lexical = icw_lexical.generate_with_watermark(
        model, tokenizer, prompt, device=device, max_new_tokens=50
    )

    freq_icw_lexical = icw_lexical.detect_watermark(text_icw_lexical)
    results['icw_lexical']['frequencies'].append(freq_icw_lexical)
    results['icw_lexical']['texts'].append(text_icw_lexical[:100])

    print(f"    ICW Lexical频率: {freq_icw_lexical:.3f}")

# 统计分析
print("\n" + "="*80)
print("实验结果统计")
print("="*80)

avg_z_no_wm = np.mean(results['no_watermark']['z_scores'])
avg_z_icw_initials = np.mean(results['icw_initials']['z_scores'])
avg_freq_icw_lexical = np.mean(results['icw_lexical']['frequencies'])

print(f"\n无水印文本:")
print(f"  平均Z-score (Initials): {avg_z_no_wm:.2f}")

print(f"\nICW Initials水印:")
print(f"  平均Z-score: {avg_z_icw_initials:.2f}")
print(f"  相对提升: {avg_z_icw_initials - avg_z_no_wm:+.2f}")

print(f"\nICW Lexical水印:")
print(f"  平均词汇频率: {avg_freq_icw_lexical:.3f}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('ICW上下文水印测试结果', fontsize=16, fontweight='bold')

# 1. Z-score对比（Initials）
ax1 = axes[0, 0]
x = np.arange(len(test_prompts))
width = 0.35

ax1.bar(x - width/2, results['no_watermark']['z_scores'], width,
        label='无水印', color='#95a5a6', alpha=0.8)
ax1.bar(x + width/2, results['icw_initials']['z_scores'], width,
        label='ICW Initials', color='#3498db', alpha=0.8)
ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='检测阈值')
ax1.set_xlabel('样本索引', fontsize=12)
ax1.set_ylabel('Z-score', fontsize=12)
ax1.set_title('ICW Initials检测效果', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. 平均对比
ax2 = axes[0, 1]
methods = ['无水印', 'ICW Initials']
avg_scores = [avg_z_no_wm, avg_z_icw_initials]
colors = ['#95a5a6', '#3498db']

bars = ax2.bar(methods, avg_scores, color=colors, alpha=0.8, width=0.6)
ax2.set_ylabel('平均Z-score', fontsize=12)
ax2.set_title('平均检测分数对比', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, avg_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.1,
             f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. Lexical词汇频率
ax3 = axes[1, 0]
ax3.bar(x, results['icw_lexical']['frequencies'], color='#2ecc71', alpha=0.8)
ax3.axhline(y=0.02, color='red', linestyle='--', alpha=0.5, label='基线频率')
ax3.set_xlabel('样本索引', fontsize=12)
ax3.set_ylabel('特定词汇频率', fontsize=12)
ax3.set_title('ICW Lexical词汇使用频率', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. 检测率分布
ax4 = axes[1, 1]
detection_threshold = 2.0
no_wm_detected = sum(1 for z in results['no_watermark']['z_scores'] if z > detection_threshold)
icw_detected = sum(1 for z in results['icw_initials']['z_scores'] if z > detection_threshold)

detection_rates = [
    no_wm_detected / len(test_prompts),
    icw_detected / len(test_prompts)
]

ax4.bar(methods, detection_rates, color=colors, alpha=0.8, width=0.6)
ax4.set_ylabel('检测率', fontsize=12)
ax4.set_title('水印检测率 (阈值=2.0)', fontsize=13, fontweight='bold')
ax4.set_ylim(0, 1)
ax4.grid(axis='y', alpha=0.3)

for i, rate in enumerate(detection_rates):
    ax4.text(i, rate + 0.02, f'{rate:.1%}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('attack_23_icw_watermark.png', dpi=300, bbox_inches='tight')
print(f"\n✅ 结果图已保存: attack_23_icw_watermark.png")

# 示例文本展示
print("\n" + "="*80)
print("示例文本对比")
print("="*80)

print(f"\n【提示词】: {test_prompts[0]}")
print(f"\n【无水印】:")
print(f"  {results['no_watermark']['texts'][0]}")
print(f"  Z-score: {results['no_watermark']['z_scores'][0]:.2f}")

print(f"\n【ICW Initials】:")
print(f"  {results['icw_initials']['texts'][0]}")
print(f"  Z-score: {results['icw_initials']['z_scores'][0]:.2f}")

print(f"\n【ICW Lexical】:")
print(f"  {results['icw_lexical']['texts'][0]}")
print(f"  词汇频率: {results['icw_lexical']['frequencies'][0]:.3f}")

# 论文对比
print("\n" + "="*80)
print("论文复现对比 (Liu et al., arXiv 2025):")
print("="*80)
print(f"{'指标':<30} {'本实验':<20} {'论文报告':<20}")
print("-"*80)
print(f"{'ICW Initials检测提升':<30} {avg_z_icw_initials-avg_z_no_wm:+.2f}{'':<17} {'+1.5~2.0':<20}")
print(f"{'Lexical词汇频率':<30} {avg_freq_icw_lexical:.3f}{'':<16} {'0.02-0.05':<20}")
print(f"{'适用模型':<30} {'OPT-125m (小)':<20} {'GPT-4 (大)':<20}")
print("="*80)
print("✅ Test 23 完成")

print("\n关键发现:")
print("  • ICW是prompt-based水印，不是logits-based")
print("  • 完全黑盒，适用于任何LLM")
print("  • 小模型效果有限，大模型效果更好")
print("  • 代表新的水印范式")
print("\n注意:")
print("  - OPT-125m可能无法很好地理解水印指令")
print("  - 论文使用GPT-4等大模型效果更好")
print("  - 这是概念验证，真实应用建议使用更大模型")
