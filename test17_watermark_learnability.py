"""测试17: 水印可学习性 — Logit知识蒸馏训练学生模型复现水印分布 (ICLR 2024)"""
from test_common import *
import math
import copy
from torch.optim import AdamW

print_test_header("水印可学习性 (Learnability) — Logit蒸馏使Student复现Teacher水印分布")

load_detector()

# ── KGW LogitsProcessor ────────────────────────────
class KGWLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, hash_key=15485863):
        self.vocab_size = vocab_size; self.gamma = gamma
        self.delta = delta; self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            g = torch.Generator(device='cpu')
            g.manual_seed(self.hash_key * input_ids[b, -1].item())
            greenlist_size = int(self.vocab_size * self.gamma)
            perm = torch.randperm(self.vocab_size, generator=g)
            scores[b, perm[:greenlist_size].to(scores.device)] += self.delta
        return scores


# ── 水印Logit蒸馏核心 ──────────────────────────────
def _watermark_logits(model, input_ids, attention_mask, kgw_processor, device):
    """对输入文本的每个位置施加KGW水印, 返回水印化后的logits

    对齐 train_logit_distill.py WatermarkLogitsDistillTrainer.compute_loss
    的 watermark_logits() 调用 (分布蒸馏模式).
    """
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.clone()

    # 逐位置施加水印 (KGW: 基于前一token哈希的绿名单+delta)
    for pos in range(1, logits.shape[1]):
        prev_token = input_ids[0, pos - 1].item()
        g = torch.Generator(device='cpu')
        g.manual_seed(15485863 * prev_token)
        greenlist_size = int(vocab_size * 0.5)
        perm = torch.randperm(vocab_size, generator=g)
        greenlist = perm[:greenlist_size]
        logits[0, pos, greenlist.to(device)] += 2.0

    return logits


def _train_distill_step(student, teacher, input_ids, attention_mask,
                         kgw_processor, optimizer, device):
    """单步Logit蒸馏: 最小化 KL(Student || Teacher_watermarked)

    对齐 WatermarkLogitsDistillTrainer.compute_loss 分布模式.
    loss = KL(log_softmax(student_logits), log_softmax(watermarked_logits)) / seq_len
    """
    student.train()
    optimizer.zero_grad()

    student_logits = student(input_ids, attention_mask=attention_mask).logits
    watermarked_logits = _watermark_logits(teacher, input_ids, attention_mask,
                                            kgw_processor, device)

    loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.log_softmax(watermarked_logits, dim=-1),
        reduction='batchmean', log_target=True,
    ) / input_ids.shape[1]

    loss.backward()
    optimizer.step()
    return loss.item()


# ── 主流程 ─────────────────────────────────────────
print("  >>> 阶段1: Logit蒸馏训练 (Student学习Teacher+KGW的水印分布)")

GAMMA = 0.5; DELTA = 2.0; HASH_KEY = 15485863
kgw_processor = KGWLogitsProcessor(vocab_size, gamma=GAMMA, delta=DELTA, hash_key=HASH_KEY)

# 训练数据: 从benchmark取文本作为训练prompts
train_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).head(30)
train_texts = [str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))[:120]
               for _, row in train_df.iterrows()]

# Student = Teacher的深拷贝 (论文中Student从Teacher权重初始化)
student_model = copy.deepcopy(target_model)
student_model.to(device)
optimizer = AdamW(student_model.parameters(), lr=1e-5)

losses = []
for epoch in range(3):
    epoch_loss = 0.0
    for text in tqdm(train_texts, desc=f"  Epoch {epoch+1}/3", leave=False):
        enc = detector_tokenizer(text, return_tensors='pt', truncation=True,
                                  max_length=128).to(device)
        if enc.input_ids.shape[1] < 5:
            continue
        loss = _train_distill_step(
            student_model, target_model, enc.input_ids, enc.attention_mask,
            kgw_processor, optimizer, device,
        )
        epoch_loss += loss
    avg_loss = epoch_loss / max(len(train_texts), 1)
    losses.append(avg_loss)
    print(f"  Epoch {epoch+1}/3  Avg KL Loss: {avg_loss:.6f}")

# ── 阶段2: 测试 ───────────────────────────────────
print("\n  >>> 阶段2: 对比 Teacher+KGW vs Student(无水印) vs Student(无训练)")

test_df = df.dropna(subset=[f"Text_{algorithms[0]}"]).iloc[35:45]
learn_results = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="  评估"):
    prompt = str(row.get("Text_Natural", row[f"Text_{algorithms[0]}"]))[:60]
    inputs = detector_tokenizer(prompt, return_tensors="pt").to(device)

    # 1. Teacher + KGW 水印 (上界: 真实水印)
    torch.manual_seed(42)
    with torch.no_grad():
        out_t = target_model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            logits_processor=LogitsProcessorList([kgw_processor]),
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    text_t = detector_tokenizer.decode(out_t[0], skip_special_tokens=True)
    z_t = detect_watermark(text_t, "KGW", detector_tokenizer, vocab_size)

    # 2. Student (蒸馏后, 无水印处理器!) ← 核心测试
    torch.manual_seed(42)
    with torch.no_grad():
        out_s = student_model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
            pad_token_id=detector_tokenizer.eos_token_id,
        )
    text_s = detector_tokenizer.decode(out_s[0], skip_special_tokens=True)
    z_s = detect_watermark(text_s, "KGW", detector_tokenizer, vocab_size)

    # 3. 未训练的原始模型 (下界: 无水印)
    z_clean = detect_watermark(
        str(row.get(f"Text_{algorithms[0]}", prompt)), "KGW", detector_tokenizer, vocab_size)

    learn_results.append({"Method": "Teacher + KGW", "Z_Score": z_t})
    learn_results.append({"Method": "Student (distilled, no WM)", "Z_Score": z_s})
    learn_results.append({"Method": "Clean (no WM)", "Z_Score": z_clean})

# ── 汇总 ──────────────────────────────────────────
df_l = pd.DataFrame(learn_results)
summary = df_l.groupby('Method')['Z_Score'].agg(['mean', 'std']).round(3)

print(f"\n=== [数据表] 水印可学习性 — Logit蒸馏实验结果 ===")
print("  对齐: train_logit_distill.py (ICLR 2024)")
print("  核心主张: 可检测性 → 可学习性")
print("  Student通过KL散度学习Teacher+KGW的logits分布, 然后无水印生成\n")
print(summary.to_string())

# ── 绘图 ──────────────────────────────────────────
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.2])
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

order = ["Teacher + KGW", "Student (distilled, no WM)", "Clean (no WM)"]
pal = {"Teacher + KGW": "#2c7bb6", "Student (distilled, no WM)": "#d7191c",
       "Clean (no WM)": "#999999"}
sns.boxplot(x="Method", y="Z_Score", data=df_l, ax=ax1,
            order=order, palette=pal, width=0.5, showfliers=False)
sns.stripplot(x="Method", y="Z_Score", data=df_l, ax=ax1,
              order=order, color='black', alpha=0.3, size=4)
ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2,
            label='Detection Threshold (z=4.0)')
ax1.set_title('Test 17: Learnability of LLM Watermarks (ICLR 2024)\n'
              'Logit Distillation: Student learns watermark distribution without key',
              fontsize=11, fontweight='bold', pad=15)
ax1.set_ylabel('Z-Score', fontsize=12); ax1.set_xlabel('')
ax1.legend(loc='upper right'); ax1.tick_params(axis='x', rotation=8)

# 训练损失曲线
ax2.plot(range(1, len(losses) + 1), losses, marker='o', color='#9467bd', linewidth=2)
ax2.set_title('Distillation Loss', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('KL Divergence')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("attack_17_watermark_learnability.png", dpi=300, bbox_inches='tight')
print(">>> 图表已保存: attack_17_watermark_learnability.png")
plt.show(); plt.close()

# 清理
del student_model
torch.cuda.empty_cache()
print("=== 测试17完成 ===\n")
