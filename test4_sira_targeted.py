"""测试4: SIRA 三阶段自信息重写攻击 (Self-Information Rewrite Attack)"""
from test_common import *

print_test_header("SIRA 三阶段自信息重写攻击 (Self-Information Rewrite Attack)")

load_attacker()  # SIRA 需要 T5 做 Stage1 改写 + Stage3 填充

sira_results = []
# 使用 percentile 阈值列表 (参考 SIRA 默认 P30)
thresholds = [10, 30, 50]  # P10=保留10%低自信息token, P30=保留30%, P50=保留50%
sample_sira_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).sample(n=min(15, len(df)), random_state=42)

for idx, row in tqdm(sample_sira_df.iterrows(), total=len(sample_sira_df), desc="SIRA 三阶段攻击"):
    for algo in algorithms:
        if algo == "Natural":
            continue
        original_text = row[f"Text_{algo}"]
        z_before = detect_watermark(original_text, algo, detector_tokenizer, vocab_size)
        sira_results.append({"Algorithm": algo, "Stage": "0_Before SIRA", "Z_Score": z_before})

        # ---- Stage 1: 生成参考文本 (改写) ----
        ref_text = sira_generate_reference(original_text)

        # ---- Stage 2+3: 不同阈值下空白化+填充 ----
        for pct in thresholds:
            blank_text = sira_masking(original_text, target_model, detector_tokenizer,
                                      threshold_percentile=pct, device=device)
            attacked_text = sira_t5_infilling(blank_text, reference_text=ref_text)
            z_after = detect_watermark(attacked_text, algo, detector_tokenizer, vocab_size)
            sira_results.append({
                "Algorithm": algo,
                "Stage": f"SIRA P{pct}",
                "Z_Score": z_after
            })


sira_df = pd.DataFrame(sira_results)

# 使用通用绘图函数（将 Stage 重命名为 State 以兼容通用函数）
sira_df_plot = sira_df.copy()
sira_df_plot.rename(columns={'Stage': 'State'}, inplace=True)

if not sira_df_plot.empty:
    plot_attack_results(
        df=sira_df_plot,
        test_name="SIRA 3-Stage Self-Information Rewrite Attack",
        test_number=4,
        output_filename="attack_4_sira_targeted.png",
        metric="Z_Score",
        threshold=4.0
    )

print("=== 测试4完成 ===\n")
