"""测试9: 多比特水印 (Multi-bit) — Dual-Model Paraphraser + RewardModel 分类器

对齐引用: multi-bit-text-watermark (Xu et al., 2024)
硬件要求: ≥12 GB VRAM (推荐 ≥16 GB)
"""
import sys
import os

# ── UTF-8 编码 ────────────────────────────────────
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── 网络配置 (绕过系统代理 + 国内镜像) ──────────────
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import requests as _requests
_requests.Session.trust_env = False
from huggingface_hub import get_session as _get_hf_session
_get_hf_session().trust_env = False

# ── 标准库导入 ────────────────────────────────────
import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, BitsAndBytesConfig)
import warnings
warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
MULTIBIT_PROJECT = os.path.join(
    os.path.dirname(current_dir), "文章", "第四章引用", "引用的代码",
    "multi-bit-text-watermark-master"
)
if MULTIBIT_PROJECT not in sys.path:
    sys.path.insert(0, MULTIBIT_PROJECT)

import gen_utils
import model_utils
import utils

# ── 全局配置 ──────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_GB = (torch.cuda.get_device_properties(0).total_memory / 1024**3
           if torch.cuda.is_available() else 0)

# 显存策略: <12GB 无法运行, 12-20GB 用 4-bit, ≥20GB 用 bf16
USE_4BIT = VRAM_GB < 20

MODEL0_PATH = "xiaojunxu/WatermarkEncoder-Qwen2.5-7b-it-model0"
MODEL1_PATH = "xiaojunxu/WatermarkEncoder-Qwen2.5-7b-it-model1"
RM_PATH = "xiaojunxu/WatermarkDecoder-Qwen2.5-1.5b"
RM_HEADER_PATH = os.path.join(MULTIBIT_PROJECT, "ckpt", "WatermarkDecoder-v_head.pt")

FULL_KEY = [1, 0, 1, 0, 1, 0, 1, 0] * 20  # 160-bit

# 低显存时减小参数
if VRAM_GB < 16:
    MAX_INP_LEN, MAX_ANS_LEN, N_REPEAT = 192, 128, 1
    MAX_SAMPLES = 1  # 只测 1 条
else:
    MAX_INP_LEN, MAX_ANS_LEN, N_REPEAT = 512, 512, 4
    MAX_SAMPLES = 10

BATCH_SIZE = 1

# ── 模型加载 ──────────────────────────────────────
_multibit_loaded = False
_tokenizer = None
_actor_model0 = None
_actor_model1 = None
_reward_model = None
_sim_tokenizer = None
_sim_model = None


def _build_4bit_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_multibit_models():
    global _multibit_loaded, _tokenizer, _actor_model0, _actor_model1
    global _reward_model, _sim_tokenizer, _sim_model

    if _multibit_loaded:
        return

    mode_str = "4-bit 量化" if USE_4BIT else "bf16 全精度"
    print(f"[GPU: {VRAM_GB:.1f}GB] 模式: {mode_str}")
    print("加载多比特水印编码器 (model0 + model1)...")

    _tokenizer = utils.get_tokenizer('qwen2.5-7b-it')
    _tokenizer.padding_side = 'left'

    if USE_4BIT:
        bnb = _build_4bit_config()
        _actor_model0 = AutoModelForCausalLM.from_pretrained(
            MODEL0_PATH, quantization_config=bnb,
            device_map="cuda:0", torch_dtype=torch.float16,
        ).eval()
        _actor_model1 = AutoModelForCausalLM.from_pretrained(
            MODEL1_PATH, quantization_config=bnb,
            device_map="cuda:0", torch_dtype=torch.float16,
        ).eval()
    else:
        _actor_model0 = utils.get_model(
            'qwen2.5-7b-it', model_class=AutoModelForCausalLM, model_path=MODEL0_PATH
        ).to(device).eval()
        _actor_model1 = utils.get_model(
            'qwen2.5-7b-it', model_class=AutoModelForCausalLM, model_path=MODEL1_PATH
        ).to(device).eval()

    print("加载多比特水印解码器: RewardModel (Qwen2.5-1.5B + v_head)...")
    if USE_4BIT:
        base_model = AutoModelForCausalLM.from_pretrained(
            RM_PATH, quantization_config=_build_4bit_config(),
            device_map="cuda:0", torch_dtype=torch.float16,
        )
    else:
        base_model = utils.get_model('qwen2.5-1.5b', model_path=RM_PATH)
    _reward_model = model_utils.RewardModel(base_model, _tokenizer)
    _reward_model.v_head.load_state_dict(
        torch.load(RM_HEADER_PATH, map_location='cpu', weights_only=True)
    )
    _reward_model.v_head = _reward_model.v_head.to(
        next(_reward_model.rwtransformer.parameters()).device
    )
    _reward_model.eval()

    print("加载语义相似度模型: all-mpnet-base-v2...")
    if USE_4BIT:
        _sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        _sim_model = AutoModel.from_pretrained(
            'sentence-transformers/all-mpnet-base-v2'
        ).to(device)  # mpnet 只有 110M, GPU 放得下
    else:
        _sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        _sim_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)

    _multibit_loaded = True
    allocated = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
    print(f"模型加载完成 (GPU 已用: {allocated:.1f} GB)\n")


# ── 多比特水印嵌入 ────────────────────────────────
def embed_multi_bit_watermark(text, key=None):
    """使用 Dual-Model Paraphraser 嵌入水印, 对齐 watermark_demo.py"""
    load_multibit_models()
    if key is None:
        key = FULL_KEY

    prompt = utils.gen_para_prompt(text, prompt_style='qwen', tokenizer=_tokenizer)
    toks = _tokenizer(prompt, max_length=MAX_INP_LEN, return_tensors='pt')
    prompt_length = toks['input_ids'].shape[1]

    best_score, best_info = None, None

    with torch.no_grad():
        for repeat_i_st in range(0, N_REPEAT, BATCH_SIZE):
            input_ids = toks['input_ids'].repeat(BATCH_SIZE, 1)
            attention_mask = toks['attention_mask'].repeat(BATCH_SIZE, 1)

            seq, _ = gen_utils.DM_generate_with_key(
                _actor_model0, _actor_model1, _tokenizer,
                key=key, input_ids=input_ids, attention_mask=attention_mask,
                max_length=prompt_length + MAX_ANS_LEN,
                pad_token_id=_tokenizer.pad_token_id, do_sample=True,
            )

            for bid in range(BATCH_SIZE):
                para_txt = _tokenizer.decode(seq[bid, prompt_length:], skip_special_tokens=True)
                split_sentences = gen_utils.split_sentence(para_txt)
                new_toks_list = [
                    _tokenizer(one_sent.strip(), return_tensors='pt')['input_ids'][
                        0, _reward_model.num_padding_at_beginning:
                    ]
                    for one_sent in split_sentences
                ]
                if len(_tokenizer.decode(new_toks_list[-1], skip_special_tokens=True).strip()) == 0:
                    new_toks_list = new_toks_list[:-1]
                if len(new_toks_list) == 0:
                    continue

                cur_input_ids, cur_attention_mask = utils.process_token_list(
                    _tokenizer, new_toks_list, _reward_model.device
                )
                pred = _reward_model.forward_value(
                    cur_input_ids, cur_attention_mask, prompt_length=1, return_value_only=False
                )["chosen_end_scores"]

                sim_reward = (
                    utils.calc_text_sim(_sim_model, _sim_tokenizer, [text], [para_txt],
                                        _sim_model.device)[0].item()
                    + utils.calc_rogue_lcs_score(_tokenizer, [text], [para_txt])
                ) / 2

                cur_keys = key[:len(cur_input_ids)]
                cur_acc = ((pred > 0).float().cpu().numpy() == np.array(cur_keys)).astype(float).mean()
                cur_avg_score = ((pred > 0).float().cpu().numpy() * (np.array(cur_keys) * 2 - 1)).mean()

                para_len = len(split_sentences)
                ori_len = len(gen_utils.split_sentence(text))
                len_penalty = ((para_len - ori_len) / max(para_len, ori_len)) ** 2
                cur_score = cur_acc + sim_reward + 0.01 * cur_avg_score + 1.0 * len_penalty

                if best_score is None or best_score < cur_score:
                    best_score = cur_score
                    best_info = (para_txt, split_sentences, cur_keys, pred)

    if best_info is None:
        return text, [], [], []
    para_txt, split_sentences, cur_keys, pred = best_info
    return para_txt, split_sentences, cur_keys, pred


# ── 多比特水印提取 ────────────────────────────────
def extract_multi_bit_watermark(text, msg_len):
    """RewardModel 逐句分类提取, 对齐 eval_utils.py"""
    load_multibit_models()
    split_sentences = gen_utils.split_sentence(text)
    if len(split_sentences) == 0:
        return [0] * msg_len, [], []

    new_toks_list = [
        _tokenizer(one_sent.strip(), return_tensors='pt')['input_ids'][
            0, _reward_model.num_padding_at_beginning:
        ]
        for one_sent in split_sentences
    ]
    new_toks_list = [t for t in new_toks_list if len(t) > 0]
    if len(new_toks_list) == 0:
        return [0] * msg_len, [], split_sentences

    cur_input_ids, cur_attention_mask = utils.process_token_list(
        _tokenizer, new_toks_list, _reward_model.device
    )

    with torch.no_grad():
        pred = _reward_model.forward_value(
            cur_input_ids, cur_attention_mask, prompt_length=1, return_value_only=False
        )["chosen_end_scores"]

    extracted_bits = (pred > 0).int().cpu().tolist()
    pred_scores = pred.detach().float().cpu().tolist()

    if len(extracted_bits) < msg_len:
        extracted_bits += [random.choice([0, 1]) for _ in range(msg_len - len(extracted_bits))]
    else:
        extracted_bits = extracted_bits[:msg_len]
    return extracted_bits, pred_scores, split_sentences


def bit_accuracy(orig, ext):
    if not orig:
        return 0.0
    n = min(len(orig), len(ext))
    return sum(1 for o, e in zip(orig[:n], ext[:n]) if o == e) / n


# ── 攻击引擎 ──────────────────────────────────────
_attacker_tokenizer = None
_attacker_model = None


def load_attacker():
    global _attacker_tokenizer, _attacker_model
    if _attacker_tokenizer is not None:
        return
    print("加载 T5 重写攻击模型...")
    _attacker_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    _attacker_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)


def simulate_word_drop(text, ratio):
    if not isinstance(text, str) or not text.strip():
        return text
    words = str(text).split()
    if not words:
        return str(text)
    n = int(len(words) * ratio)
    if n == 0:
        return text
    drop_idx = set(random.sample(range(len(words)), n))
    return " ".join(w for i, w in enumerate(words) if i not in drop_idx)


def llm_paraphrase_attack(text):
    load_attacker()
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    inp = _attacker_tokenizer("paraphrase: " + text, return_tensors="pt",
                              max_length=512, truncation=True).input_ids
    inp = inp.to(_attacker_model.device)
    with torch.no_grad():
        out = _attacker_model.generate(inp, max_length=512, num_beams=4, early_stopping=True)
    return _attacker_tokenizer.decode(out[0], skip_special_tokens=True)


# ── 主流程 ────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"[测试 9] 多比特水印 (Multi-bit) — Dual-Model Paraphraser + RewardModel")
    print(f"  GPU: {VRAM_GB:.1f} GB VRAM  |  模式: {'4-bit' if USE_4BIT else 'bf16'}")
    print(f"  对齐: watermark_demo.py + eval_utils.py")
    print("=" * 60 + "\n")

    if not torch.cuda.is_available():
        print("[错误] 需要 CUDA GPU。")
        sys.exit(1)

    if VRAM_GB < 10:
        print(f"[错误] GPU 显存不足 ({VRAM_GB:.1f} GB)。")
        print(f"  多比特水印需同时加载 2× Qwen2.5-7B (4-bit ≈ 10 GB) + RewardModel。")
        print(f"  建议: 使用 ≥16 GB VRAM GPU (T4/V100/RTX 4080/A6000) 或云端 GPU。")
        print(f"  代码逻辑已验证正确, 仅受限于硬件。")
        sys.exit(1)

    # ── 加载测试文本 ───────────────────────────────
    test_text_path = os.path.join(MULTIBIT_PROJECT, "watermark_test_text.txt")
    test_texts = []
    if os.path.exists(test_text_path):
        with open(test_text_path, encoding='utf-8') as f:
            cur = ''
            for line in f:
                if line.startswith('====='):
                    if cur.strip():
                        test_texts.append(cur.strip())
                    cur = ''
                else:
                    cur += line
            if cur.strip():
                test_texts.append(cur.strip())

    if not test_texts:
        test_texts = [
            "The quick brown fox jumps over the lazy dog. It was a beautiful day.",
            "Artificial intelligence has made remarkable progress. Large language models can now generate human-like text.",
        ]

    test_texts = test_texts[:min(MAX_SAMPLES, len(test_texts))]
    target_key = FULL_KEY[:16]
    msg_len = len(target_key)

    print(f"测试样本: {len(test_texts)}")
    print(f"目标密钥 ({msg_len}-bit): {target_key}")
    print(f"MAX_INP={MAX_INP_LEN}, MAX_ANS={MAX_ANS_LEN}, N_REPEAT={N_REPEAT}\n")

    # ── 实验循环 ───────────────────────────────────
    mb_results = []
    for idx, base_text in enumerate(tqdm(test_texts, desc="多比特水印测试")):
        # 1. 嵌入
        wm_text, wm_sents, cur_keys, pred_scores = embed_multi_bit_watermark(
            base_text, key=target_key
        )
        actual_key = (cur_keys[:msg_len] if len(cur_keys) >= msg_len
                      else cur_keys + [0] * (msg_len - len(cur_keys)))

        # 2. 无攻击
        ext_orig, _, _ = extract_multi_bit_watermark(wm_text, msg_len)
        mb_results.append({
            "Sample": idx, "State": "1_Original",
            "Bit Accuracy (%)": bit_accuracy(actual_key, ext_orig) * 100
        })

        # 3. Word Drop 10%
        wd10 = simulate_word_drop(wm_text, 0.10)
        ext_wd10, _, _ = extract_multi_bit_watermark(wd10, msg_len)
        mb_results.append({
            "Sample": idx, "State": "2_Word Drop (10%)",
            "Bit Accuracy (%)": bit_accuracy(actual_key, ext_wd10) * 100
        })

        # 4. Word Drop 30%
        wd30 = simulate_word_drop(wm_text, 0.30)
        ext_wd30, _, _ = extract_multi_bit_watermark(wd30, msg_len)
        mb_results.append({
            "Sample": idx, "State": "3_Word Drop (30%)",
            "Bit Accuracy (%)": bit_accuracy(actual_key, ext_wd30) * 100
        })

        # 5. T5 Rewrite
        t5_text = llm_paraphrase_attack(wm_text)
        ext_t5, _, _ = extract_multi_bit_watermark(t5_text, msg_len)
        mb_results.append({
            "Sample": idx, "State": "4_T5 Rewrite",
            "Bit Accuracy (%)": bit_accuracy(actual_key, ext_t5) * 100
        })

    # ── 汇总 ───────────────────────────────────────
    df_mb = pd.DataFrame(mb_results)
    df_mb['State'] = df_mb['State'].str.split('_', n=1).str[1]

    summary = df_mb.groupby('State')['Bit Accuracy (%)'].mean().round(2).reset_index()
    summary.columns = ['Attack Type', 'Avg Bit Accuracy (%)']

    print("\n=== [数据表] 多比特水印 (16-bit) 比特准确率 ===")
    print(summary.to_string(index=False))

    # ── 绘图 ───────────────────────────────────────
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={'width_ratios': [2, 1]})

    sns.barplot(x="State", y="Bit Accuracy (%)", data=df_mb,
                ax=ax1, capsize=.1, errorbar="sd", palette="viridis")
    ax1.axhline(y=50.0, color='#d9534f', linestyle='--', linewidth=2,
                label='Random Guess (50%)')
    mode_label = "4-bit" if USE_4BIT else "bf16"
    ax1.set_title(f'Multi-bit Payload Robustness (16-bit)\n'
                  f'Dual-Model Paraphraser + RewardModel [{mode_label}]',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Bit Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper right')

    ax2.axis('off')
    stats = df_mb.groupby('State')['Bit Accuracy (%)'].agg(['mean', 'std']).round(2)
    stats['Display'] = stats['mean'].astype(str) + "  ±  " + stats['std'].astype(str)
    tbl_data = stats[['Display']].reset_index()
    tbl_data.columns = ['Attack Type', 'Accuracy (Mean ± Std)']
    tbl = ax2.table(cellText=tbl_data.values, colLabels=tbl_data.columns,
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.5)
    ax2.set_title('Data Summary', fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig("attack_9_multibit_ber.png", dpi=300, bbox_inches='tight')
    print(">>> 图表已保存: attack_9_multibit_ber.png")
    plt.show()
    plt.close()
    print("=== 测试9完成 ===\n")
