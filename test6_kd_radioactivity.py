"""测试6: 知识蒸馏(KD)的水印放射性与 WN 中和攻击"""
from test_common import *

print_test_header("知识蒸馏(KD)的水印放射性与 WN 中和攻击")

def get_kgw_green_mask(prefix_token, vocab_size, gamma=0.5, secret_key=15485863):
    torch.manual_seed(secret_key * prefix_token)
    return (torch.rand(vocab_size) < gamma)

def simulate_kd_radioactivity(teacher_text, tokenizer, vocab_size, gamma=0.5, secret_key=15485863):
    tokens = tokenizer.encode(teacher_text, return_tensors="pt")[0].tolist()
    if len(tokens) <= 1: return teacher_text

    student_tokens = [tokens[0]]
    for i in range(1, len(tokens)):
        # 修复：跳过超出 vocab_size 的 token
        if tokens[i] >= vocab_size:
            student_tokens.append(tokens[i])
            continue

        green_mask = get_kgw_green_mask(tokens[i-1], vocab_size, gamma, secret_key)
        is_green = green_mask[tokens[i]].item()

        if random.random() < 0.85:
            student_tokens.append(tokens[i])
        else:
            if is_green and random.random() < 0.55:
                student_tokens.append(tokens[i])
            elif not is_green and random.random() < 0.40:
                student_tokens.append(tokens[i])
            else:
                student_tokens.append((tokens[i] + random.randint(1, 200)) % vocab_size)

    return tokenizer.decode(student_tokens, skip_special_tokens=True)

def simulate_wn_neutralization(teacher_text, tokenizer, vocab_size, gamma=0.5, secret_key=15485863, delta=3.0):
    tokens = tokenizer.encode(teacher_text, return_tensors="pt")[0].tolist()
    if len(tokens) <= 1: return teacher_text

    stolen_greenlist = {}
    for i in range(1, len(tokens)):
        prev = tokens[i-1]
        # 修复：跳过超出 vocab_size 的 token
        if prev >= vocab_size:
            continue
        if prev not in stolen_greenlist:
            actual_green = get_kgw_green_mask(prev, vocab_size, gamma, secret_key)
            inferred = set()
            for tid in range(vocab_size):
                if actual_green[tid].item():
                    if random.random() < 0.90: inferred.add(tid)
                else:
                    if random.random() < 0.05: inferred.add(tid)
            stolen_greenlist[prev] = inferred

    neutralized_tokens = [tokens[0]]
    for i in range(1, len(tokens)):
        prev_token = tokens[i-1]
        inferred_green = stolen_greenlist.get(prev_token, set())
        current = tokens[i]

        # 修复：跳过超出 vocab_size 的 token
        if current >= vocab_size or prev_token >= vocab_size:
            neutralized_tokens.append(current)
            continue

        if current in inferred_green:
            if random.random() < gamma:
                neutralized_tokens.append(current)
            else:
                actual_green = get_kgw_green_mask(prev_token, vocab_size, gamma, secret_key)
                candidate = (current + random.randint(1, 500)) % vocab_size
                attempts = 0
                while actual_green[candidate].item() and attempts < 200:
                    candidate = (candidate + 1) % vocab_size
                    attempts += 1
                neutralized_tokens.append(candidate)
        else:
            if random.random() < 0.85:
                neutralized_tokens.append(current)
            else:
                neutralized_tokens.append((current + random.randint(1, 200)) % vocab_size)

    return tokenizer.decode(neutralized_tokens, skip_special_tokens=True)

radioactivity_results = []
sample_kd_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).sample(n=min(20, len(df)), random_state=42)

for idx, row in tqdm(sample_kd_df.iterrows(), total=len(sample_kd_df), desc="知识蒸馏与WN中和"):
    for algo in algorithms:
        if algo == "Natural": continue

        teacher_text = row[f"Text_{algo}"]
        z_teacher = detect_watermark(teacher_text, algo, detector_tokenizer, vocab_size)

        student_radioactive = simulate_kd_radioactivity(teacher_text, detector_tokenizer, vocab_size)
        z_radioactive = detect_watermark(student_radioactive, algo, detector_tokenizer, vocab_size)

        student_wn = simulate_wn_neutralization(teacher_text, detector_tokenizer, vocab_size)
        z_wn = detect_watermark(student_wn, algo, detector_tokenizer, vocab_size)

        radioactivity_results.append({"Algorithm": algo, "State": "1_Teacher Model", "Z_Score": z_teacher})
        radioactivity_results.append({"Algorithm": algo, "State": "2_Student (Radioactive)", "Z_Score": z_radioactive})
        radioactivity_results.append({"Algorithm": algo, "State": "3_Student (WN Neutralized)", "Z_Score": z_wn})


kd_df = pd.DataFrame(radioactivity_results)

if not kd_df.empty:
    # 使用通用绘图函数
    plot_attack_results(
        df=kd_df,
        test_name="KD Radioactivity vs. Watermark Neutralization (WN)\n(Pan et al., ACL 2025)",
        test_number=6,
        output_filename="attack_6_kd_radioactivity_wn.png",
        metric="Z_Score",
        threshold=4.0
    )

print("=== 测试6完成 ===\n")
