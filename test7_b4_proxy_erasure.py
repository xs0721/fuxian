"""测试7: B4 黑盒代理模型盲擦除攻击 (NAACL 2025)"""
from test_common import *

print_test_header("B4 黑盒代理模型盲擦除攻击 (NAACL 2025)")

# 卸载 OPT-125m 模型权重释放 VRAM（B4 仅需 tokenizer）
if target_model is not None:
    target_model.cpu()
torch.cuda.empty_cache()

print(">>> 加载 B4 所需 Gemma-2-2B 模型 (共享权重节省 VRAM)...")
try:
    b4_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_DIR, torch_dtype=torch.bfloat16).to(device)
    b4_model.eval()
    # B4 三个角色共用同一份权重，通过独立 KV-cache 实现分工
    paraphrase_model = b4_model
    amateur_model = b4_model
    origin_model = b4_model
    b4_tokenizer = AutoTokenizer.from_pretrained(GEMMA_DIR)
    b4_amateur_tokenizer = b4_tokenizer
    b4_tokenizer.padding_side = "left"
    print(">>> B4 模型加载完成")
except Exception as e:
    print(f"B4 模型加载失败: {e}")
    exit(1)

b4_results = []
sample_b4_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).sample(n=min(20, len(df)), random_state=42)

for algo in algorithms:
    if algo == "Natural":
        continue
    original_texts = sample_b4_df[f"Text_{algo}"].tolist()
    print(f"\n>>> 处理算法: {algo}, 样本数: {len(original_texts)}")

    for text in original_texts:
        z_before = detect_watermark(text, algo, detector_tokenizer, vocab_size)
        b4_results.append({"Algorithm": algo, "State": "1_Before B4", "Z_Score": z_before})

    try:
        attacked_texts = b4_proxy_erasure_attack(
            texts=original_texts, paraphrase_model=paraphrase_model,
            amateur_model=amateur_model, origin_model=origin_model,
            tokenizer=b4_tokenizer, amateur_tokenizer=b4_amateur_tokenizer,
            prompt_id=4, num_beams=4, max_new_tokens=200, coef=1.0, batch_size=1)
        print(f"  B4 攻击完成, 结果数: {len(attacked_texts)}")

        for text in attacked_texts:
            z_after = detect_watermark(text, algo, detector_tokenizer, vocab_size)
            b4_results.append({"Algorithm": algo, "State": "2_After B4", "Z_Score": z_after})
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
    plt.close()

print("=== 测试7完成 ===\n")
