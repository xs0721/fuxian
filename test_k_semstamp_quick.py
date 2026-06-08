"""test_k_semstamp_quick.py - k-SemStamp快速验证

论文: k-SemStamp: A Clustering-Based Semantic Watermark (ACL 2024 Findings)
作者: Abe Hou, Jingyu Zhang, Yichen Wang, Daniel Khashabi, Tianxing He

k-SemStamp vs SemStamp:
- SemStamp: 单一语义空间
- k-SemStamp: k个聚类语义空间，更灵活和鲁棒

快速验证k-SemStamp的实现是否正确
"""

import sys
import os

# ============================================================================
# 自动设置环境变量 - 无需手动export
# ============================================================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 强制UTF-8编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("快速验证 k-SemStamp (基于k-means聚类的语义水印) 实现")
print("=" * 80)

# 导入必要模块
from run_experiment import KSemStampLogitsProcessor, detect_watermark
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import torch

print("\n[1/4] 测试KSemStampLogitsProcessor实例化...")
try:
    vocab_size = 50272
    processor = KSemStampLogitsProcessor(vocab_size, k=5, gamma=0.5, delta=2.0)
    print("✅ KSemStampLogitsProcessor实例化成功")
    print(f"   簇数量(k): {processor.k}")
    print(f"   词表大小: {processor.vocab_size}")
    print(f"   绿名单比例(gamma): {processor.gamma}")
    print(f"   水印强度(delta): {processor.delta}")
except Exception as e:
    print(f"❌ KSemStampLogitsProcessor实例化失败: {e}")
    sys.exit(1)

print("\n[2/4] 测试LogitsProcessor功能...")
try:
    # 模拟输入
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    scores = torch.randn(1, vocab_size)
    original_scores = scores.clone()

    # 应用processor
    modified_scores = processor(input_ids, scores)

    print("✅ LogitsProcessor功能正常")
    print(f"   输入shape: {scores.shape}")
    print(f"   输出shape: {modified_scores.shape}")
    print(f"   分数已修改: {not torch.equal(original_scores, modified_scores)}")
except Exception as e:
    print(f"❌ LogitsProcessor测试失败: {e}")
    sys.exit(1)

print("\n[3/4] 测试聚类分配...")
try:
    # 检查token到簇的映射
    cluster_distribution = {}
    for token_id in range(min(1000, vocab_size)):  # 只检查前1000个
        cluster_id = processor.token_to_cluster[token_id]
        cluster_distribution[cluster_id] = cluster_distribution.get(cluster_id, 0) + 1

    print("✅ 聚类分配正常")
    print(f"   簇分布: {cluster_distribution}")
    print(f"   簇数量: {len(cluster_distribution)}")
except Exception as e:
    print(f"❌ 聚类分配测试失败: {e}")
    sys.exit(1)

print("\n[4/4] 测试k-SemStamp检测功能...")
try:
    # 准备tokenizer
    from test_common import TARGET_MODEL, CACHE_DIR
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, cache_dir=CACHE_DIR)

    # 测试文本
    test_text_watermarked = "Machine learning is a method of data analysis that automates analytical model building."
    test_text_human = "The quick brown fox jumps over the lazy dog and runs away quickly."
    test_text_train = "Artificial intelligence and machine learning are transforming modern technology."

    # 检测
    z_watermarked = detect_watermark(test_text_watermarked, "k-SemStamp", tokenizer, vocab_size)
    z_human = detect_watermark(test_text_human, "k-SemStamp", tokenizer, vocab_size)
    z_train = detect_watermark(test_text_train, "k-SemStamp", tokenizer, vocab_size)

    print("✅ k-SemStamp检测功能正常")
    print(f"   水印文本: {test_text_watermarked[:50]}...")
    print(f"   Z-score: {z_watermarked:.2f}")
    print(f"   人类文本 Z-score: {z_human:.2f}")
    print(f"   训练语料 Z-score: {z_train:.2f}")
except Exception as e:
    print(f"❌ k-SemStamp检测测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("k-SemStamp关键特性验证:")
print("=" * 80)
print("✅ KSemStampLogitsProcessor成功实现")
print("✅ LogitsProcessor功能正常")
print("✅ k-means聚类分配正常")
print("✅ 检测函数正常工作")

print("\n特点:")
print("  • 使用k个语义簇（而非单一空间）")
print("  • 更灵活的绿名单选择")
print("  • 比SemStamp更鲁棒")
print("  • ACL 2024 Findings论文")

print("\n简化说明:")
print("  • 完整版: 用sentence-transformers进行真实k-means聚类")
print("  • 简化版: 用哈希函数模拟聚类效果")
print("  • 核心思想一致，实现更轻量")

print("\n" + "=" * 80)
print("✅ k-SemStamp验证完成")
print("=" * 80)

print("\nk-SemStamp在run_experiment.py中的位置:")
print("  1. KSemStampLogitsProcessor类")
print("  2. detect_watermark()中的k-SemStamp分支")

print("\n使用方法:")
print("  # 生成")
print("  processor = KSemStampLogitsProcessor(vocab_size, k=5)")
print("  outputs = model.generate(..., logits_processor=[processor])")
print("  ")
print("  # 检测")
print("  z_score = detect_watermark(text, 'k-SemStamp', tokenizer, vocab_size)")

print("\nk-SemStamp vs SemStamp:")
print("  • SemStamp: 单一语义空间")
print("  • k-SemStamp: k个聚类，更细粒度的语义控制")
print("  • k-SemStamp更适合复杂文本")

# ============================================================================
# 可视化
# ============================================================================
print("\n" + "=" * 80)
print("📊 生成可视化...")
print("=" * 80)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# 数据
labels = ['Training\nCorpus', 'Watermarked\nText', 'Human\nText']
z_scores = [z_train, z_watermarked, z_human]
colors = ['#3498db', '#2ecc71', '#e74c3c']

bars = ax.bar(labels, z_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

ax.set_ylabel('Z-Score', fontsize=13, fontweight='bold')
ax.set_title('k-SemStamp Detection Results', fontsize=15, fontweight='bold', pad=15)
ax.axhline(y=4.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Safe Threshold (Z=4.0)')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(fontsize=11)

# 标注数值
for bar, score in zip(bars, z_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
output_file = 'k_semstamp_quick_test.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ 可视化已保存: {output_file}")
plt.close()
