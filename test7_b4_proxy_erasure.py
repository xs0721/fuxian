"""测试7: B4 黑盒代理模型盲擦除攻击 (NAACL 2025)"""
from test_common import *
from test_common import _get_local_model_path  # 显式导入私有函数
import time

print_test_header("B4 黑盒代理模型盲擦除攻击 (NAACL 2025)")

# test7 特殊处理：需要用 OPT 作为检测器（Gemma2 有 CUDA 兼容性问题）
print(">>> test7: 重新加载 OPT-1.3b 作为检测器（替换 Gemma）...")
if target_model is not None:
    del target_model
    del detector_tokenizer
    torch.cuda.empty_cache()

opt_detector_path = _get_local_model_path("facebook/opt-1.3b", CACHE_DIR)
print(f"    → 加载 OPT-1.3b: {opt_detector_path}")
detector_tokenizer = AutoTokenizer.from_pretrained(opt_detector_path, cache_dir=CACHE_DIR)
target_model = AutoModelForCausalLM.from_pretrained(opt_detector_path, cache_dir=CACHE_DIR).to(device)
print(f"    ✅ OPT 检测器加载完成")

print(">>> 加载 B4 所需 OPT-1.3b 模型（与检测器共享权重）...")
# B4 的三个角色直接复用检测器模型，节省显存
b4_model = target_model
b4_tokenizer = detector_tokenizer
paraphrase_model = target_model
amateur_model = target_model
origin_model = target_model
b4_amateur_tokenizer = detector_tokenizer
b4_tokenizer.padding_side = "left"
print(">>> B4 模型配置完成（共享检测器权重）")

b4_results = []
sample_b4_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).sample(n=min(20, len(df)), random_state=42)

for algo in algorithms:
    if algo == "Natural":
        continue
    original_texts = sample_b4_df[f"Text_{algo}"].tolist()
    print(f"\n>>> 处理算法: {algo}, 样本数: {len(original_texts)}")

    # === 步骤1: 计算攻击前Z-score ===
    print(f"  [1/3] 计算攻击前Z-score... ", end='', flush=True)
    for text in original_texts:
        z_before = detect_watermark(text, algo, detector_tokenizer, vocab_size)
        b4_results.append({"Algorithm": algo, "State": "1_Before B4", "Z_Score": z_before})
    print(f"✓ 完成")

    # === 步骤2: B4攻击（逐样本显示进度） ===
    print(f"  [2/3] B4攻击 (逐样本处理):")
    attacked_texts = []
    attack_start_time = time.time()

    try:
        for sample_idx, text in enumerate(original_texts):
            sample_start = time.time()
            progress_pct = (sample_idx + 1) / len(original_texts) * 100

            print(f"    [{sample_idx+1:2d}/{len(original_texts):2d}] ({progress_pct:5.1f}%) ", end='', flush=True)

            try:
                result = b4_proxy_erasure_attack(
                    texts=[text],
                    paraphrase_model=paraphrase_model,
                    amateur_model=amateur_model,
                    origin_model=origin_model,
                    tokenizer=b4_tokenizer,
                    amateur_tokenizer=b4_amateur_tokenizer,
                    prompt_id=4, num_beams=4, max_new_tokens=200, coef=1.0, batch_size=1)

                attacked_texts.extend(result)

                elapsed = time.time() - sample_start
                total_elapsed = time.time() - attack_start_time
                avg_time = total_elapsed / (sample_idx + 1)
                remaining = avg_time * (len(original_texts) - sample_idx - 1)

                print(f"✓ {elapsed:5.1f}s | 平均:{avg_time:5.1f}s | 剩余:{remaining/60:.1f}min", flush=True)

                # 每5个样本清理一次显存
                if (sample_idx + 1) % 5 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"✗ 失败: {str(e)[:40]}", flush=True)
                continue

        total_time = time.time() - attack_start_time
        print(f"  ✓ B4攻击完成! 总耗时:{total_time/60:.1f}分钟, 成功:{len(attacked_texts)}/{len(original_texts)}")

        # === 步骤3: 计算攻击后Z-score ===
        print(f"  [3/3] 计算攻击后Z-score... ", end='', flush=True)
        for text in attacked_texts:
            z_after = detect_watermark(text, algo, detector_tokenizer, vocab_size)
            b4_results.append({"Algorithm": algo, "State": "2_After B4", "Z_Score": z_after})
        print(f"✓ 完成")
    except Exception as e:
        import traceback
        print(f"  B4 攻击失败跳过: {e}")
        traceback.print_exc()

b4_df = pd.DataFrame(b4_results)
if not b4_df.empty:
    summary_table_b4 = b4_df.groupby(['Algorithm', 'State'])['Z_Score'].mean().unstack('State').round(3)
    summary_table_b4.columns = [col.split('_')[1] for col in summary_table_b4.columns]

    print("\n=== [数据表] B4 黑盒代理盲擦除攻击 Z-Score 汇总 ===")
    print(summary_table_b4.to_string())

    b4_df['State'] = b4_df['State'].apply(lambda x: x.split('_')[1])
    fig7 = plt.figure(figsize=(14, 6))
    gs7 = fig7.add_gridspec(1, 2, width_ratios=[2, 1.2])
    ax7_1 = fig7.add_subplot(gs7[0])
    ax7_2 = fig7.add_subplot(gs7[1])

    sns.boxplot(x="Algorithm", y="Z_Score", hue="State", data=b4_df, ax=ax7_1, width=0.6, showfliers=False)
    sns.stripplot(x="Algorithm", y="Z_Score", hue="State", data=b4_df, ax=ax7_1, dodge=True, color='black', alpha=0.3)
    ax7_1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
    ax7_1.set_title('Test 7: B4 Proxy-guided Blind Erasure Attack (NAACL 2025)', fontsize=14, fontweight='bold')
    ax7_1.set_ylabel('Z-Score', fontsize=13)
    ax7_1.legend(loc='upper right')

    table7_data = summary_table_b4.reset_index()
    table7_data.rename(columns={'Algorithm': 'Algo\n(Z-Score)'}, inplace=True)
    ax7_2.axis('off')
    ax7_2.set_title('Data Summary (Metric: Z-Score)', fontsize=13, fontweight='bold', pad=10)
    table7 = ax7_2.table(cellText=table7_data.values, colLabels=table7_data.columns, loc='center', cellLoc='center')
    table7.auto_set_font_size(False)
    table7.set_fontsize(10)
    table7.scale(1, 2.5)

    plt.tight_layout()
    plt.savefig("attack_7_b4_proxy_erasure.png", dpi=300, bbox_inches='tight')
    print(">>> 图表已保存: attack_7_b4_proxy_erasure.png")
    plt.show()
    plt.close()

print("=== 测试7完成 ===\n")
