"""测试2: LLM Rewrite 与 CWRA 跨语言纵深打击"""
from test_common import *

print_test_header("LLM Rewrite 与 CWRA 跨语言纵深打击")

sample_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).sample(n=min(20, len(df)), random_state=42)
attack_results_complex = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="高级攻击进度"):
    for algo in algorithms:
        original_text = row[f"Text_{algo}"]
        attack_results_complex.append({"Algorithm": algo, "State": "1_Before Attack", "Z_Score": detect_watermark(original_text, algo, detector_tokenizer, vocab_size)})
        t5_attacked_text = llm_paraphrase_attack(original_text)
        attack_results_complex.append({"Algorithm": algo, "State": "2_After T5 Rewrite", "Z_Score": detect_watermark(t5_attacked_text, algo, detector_tokenizer, vocab_size)})
        cwra_attacked_text = cwra_translation_attack(original_text)
        attack_results_complex.append({"Algorithm": algo, "State": "3_After CWRA", "Z_Score": detect_watermark(cwra_attacked_text, algo, detector_tokenizer, vocab_size)})


results_df = pd.DataFrame(attack_results_complex)

if not results_df.empty:
    # 使用通用绘图函数
    plot_attack_results(
        df=results_df,
        test_name="Vulnerability to T5 Paraphrasing & CWRA",
        test_number=2,
        output_filename="attack_2_complex_rewrite.png",
        metric="Z_Score",
        threshold=4.0
    )

print("=== 测试2完成 ===\n")
