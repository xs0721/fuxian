"""各测试共用的配置、模型和工具函数"""
import pandas as pd
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList, GenerationConfig
from transformers.cache_utils import DynamicCache
from copy import deepcopy
import os
import warnings
import re
import glob

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

CACHE_DIR = "E:/Your_Cloud_Drive/hf_cache"
TARGET_MODEL = "facebook/opt-125m"
ATTACKER_MODEL = "t5-small"
CSV_FILENAME = "watermark_benchmark_results.csv"
GEMMA_DIR = "E:/Your_Cloud_Drive/hf_cache/LLM-Research/gemma-2-2b-it"

device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 50272

# ================= 懒加载模型 =================
_models_loaded = {
    "detector": False, "attacker": False, "translation": False
}
detector_tokenizer = None
target_model = None
attacker_tokenizer = None
attacker_model = None
en_fr_tokenizer = None
en_fr_model = None
fr_en_tokenizer = None
fr_en_model = None

def load_detector():
    global detector_tokenizer, target_model
    if not _models_loaded["detector"]:
        print(f"[{device.upper()}] 加载检测器: {TARGET_MODEL}...")
        detector_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, cache_dir=CACHE_DIR)
        target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, cache_dir=CACHE_DIR).to(device)
        _models_loaded["detector"] = True

def load_attacker():
    global attacker_tokenizer, attacker_model
    if not _models_loaded["attacker"]:
        print(f"[{device.upper()}] 加载重写模型: {ATTACKER_MODEL}...")
        attacker_tokenizer = AutoTokenizer.from_pretrained(ATTACKER_MODEL, cache_dir=CACHE_DIR)
        attacker_model = AutoModelForSeq2SeqLM.from_pretrained(ATTACKER_MODEL, cache_dir=CACHE_DIR).to(device)
        _models_loaded["attacker"] = True

def load_translation_models():
    global en_fr_tokenizer, en_fr_model, fr_en_tokenizer, fr_en_model
    if not _models_loaded["translation"]:
        print(f"[{device.upper()}] 加载 CWRA 回译引擎 (En<->Fr)...")
        en_fr_mn = "Helsinki-NLP/opus-mt-en-fr"
        fr_en_mn = "Helsinki-NLP/opus-mt-fr-en"
        en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_mn, cache_dir=CACHE_DIR)
        en_fr_model = MarianMTModel.from_pretrained(en_fr_mn, cache_dir=CACHE_DIR).to(device)
        fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_mn, cache_dir=CACHE_DIR)
        fr_en_model = MarianMTModel.from_pretrained(fr_en_mn, cache_dir=CACHE_DIR).to(device)
        _models_loaded["translation"] = True

# 默认加载检测器（所有测试都需要）
load_detector()

# ================= 攻击引擎 =================
def simulate_word_drop(text, drop_ratio):
    words = str(text).split()
    if not words: return str(text)
    num_drop = int(len(words) * drop_ratio)
    indices_to_drop = set(random.sample(range(len(words)), num_drop))
    tampered_words = [word for i, word in enumerate(words) if i not in indices_to_drop]
    return " ".join(tampered_words)

def character_removal_attack(text, drop_ratio):
    if not text: return str(text)
    chars = list(text)
    num_drop = int(len(chars) * drop_ratio)
    indices_to_drop = set(random.sample(range(len(chars)), num_drop))
    return "".join([c for i, c in enumerate(chars) if i not in indices_to_drop])

def llm_paraphrase_attack(text):
    load_attacker()
    if not isinstance(text, str) or len(text.strip()) == 0: return text
    input_ids = attacker_tokenizer("paraphrase: " + text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
    with torch.no_grad():
        outputs = attacker_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    return attacker_tokenizer.decode(outputs[0], skip_special_tokens=True)

def cwra_translation_attack(text):
    load_translation_models()
    if not isinstance(text, str) or len(text.strip()) == 0: return text
    try:
        inputs_en = en_fr_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs_fr = en_fr_model.generate(**inputs_en, max_length=512)
        fr_text = en_fr_tokenizer.decode(outputs_fr[0], skip_special_tokens=True)

        inputs_fr = fr_en_tokenizer(fr_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs_en = fr_en_model.generate(**inputs_fr, max_length=512)
        en_text = fr_en_tokenizer.decode(outputs_en[0], skip_special_tokens=True)
        return en_text
    except Exception:
        return text

def watermark_stealing_attack(text, algo_name, attack_type, tokenizer, vocab_size, gamma=0.5, secret_key=15485863):
    if algo_name in ["Natural", "SemStamp"]: return text
    if not isinstance(text, str) or len(text.strip()) == 0: return text

    tokens = tokenizer.encode(text, return_tensors="pt")[0].tolist()
    if len(tokens) <= 1: return text

    tampered_tokens = [tokens[0]]
    for i in range(1, len(tokens)):
        prev = tokens[i-1]
        curr = tokens[i]
        torch.manual_seed(secret_key * prev if algo_name in ["KGW", "SWEET", "DiPmark"] else 42)
        green_mask = (torch.rand(vocab_size) < gamma)
        is_green = green_mask[curr].item()

        if attack_type == "scrubbing" and is_green and random.random() < 0.6:
            candidate = (curr + random.randint(1, 100)) % vocab_size
            while green_mask[candidate].item():
                candidate = (candidate + 1) % vocab_size
            tampered_tokens.append(candidate)
        elif attack_type == "spoofing" and not is_green and random.random() < 0.6:
            candidate = (curr + random.randint(1, 100)) % vocab_size
            while not green_mask[candidate].item():
                candidate = (candidate + 1) % vocab_size
            tampered_tokens.append(candidate)
        else:
            tampered_tokens.append(curr)
    return tokenizer.decode(tampered_tokens, skip_special_tokens=True)

# ================= SIRA 组件 =================
def calculate_token_self_information(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    if input_ids.shape[1] <= 1: return [], []
    with torch.no_grad(): outputs = model(input_ids)
    logits = outputs.logits[0, :-1, :]
    target_ids = input_ids[0, 1:]
    probs = F.softmax(logits, dim=-1)
    token_probs = probs.gather(1, target_ids.unsqueeze(1)).squeeze(-1)
    self_info = -torch.log2(token_probs + 1e-10)
    tokens = tokenizer.convert_ids_to_tokens(target_ids)
    return tokens, self_info.cpu().numpy()

def sira_masking(text, model, tokenizer, mask_ratio=0.2, device="cuda"):
    tokens, self_info_scores = calculate_token_self_information(text, model, tokenizer, device)
    if not tokens or len(tokens) < 5: return text
    num_to_mask = max(1, int(len(tokens) * mask_ratio))
    highest_entropy_indices = np.argsort(self_info_scores)[-num_to_mask:]
    masked_tokens = []
    mask_count = 0
    i = 0
    while i < len(tokens):
        if i in highest_entropy_indices:
            masked_tokens.append(f"<extra_id_{mask_count}>")
            mask_count += 1
            while i < len(tokens) and i in highest_entropy_indices: i += 1
            continue
        else:
            clean_token = tokens[i].replace('Ġ', ' ').replace('Ċ', '\n')
            masked_tokens.append(clean_token)
            i += 1
    return "".join(masked_tokens).strip()

def sira_t5_infilling(masked_text):
    load_attacker()
    if "<extra_id_" not in masked_text: return masked_text
    inputs = attacker_tokenizer(masked_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = attacker_model.generate(**inputs, max_length=512, num_beams=4, temperature=0.8, do_sample=True, early_stopping=True)
    filled_content = attacker_tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_text = masked_text
    parts = [p.strip() for p in filled_content.split("<extra_id_") if p.strip()]
    for i, part in enumerate(parts):
        if '>' in part:
            clean_part = part.split('>', 1)[1].strip()
            final_text = final_text.replace(f"<extra_id_{i}>", " " + clean_part + " ")
    final_text = re.sub(r'<extra_id_\d+>', '', final_text)
    return final_text.replace("  ", " ").strip()

# ================= 检测器 =================
def detect_watermark(text, algo_name, tokenizer, vocab_size, gamma=0.5, secret_key=15485863):
    if algo_name == "Natural": return 0.0
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    total_tokens = len(tokens) - 1
    if total_tokens <= 0: return 0.0
    green_tokens_count = 0

    if algo_name in ["KGW", "SWEET", "DiPmark"]:
        for i in range(1, len(tokens)):
            torch.manual_seed(secret_key * tokens[i - 1].item())
            if (torch.rand(vocab_size) < gamma)[tokens[i].item()]:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

    elif algo_name == "Unigram":
        torch.manual_seed(42)
        green_mask = (torch.rand(vocab_size) < gamma)
        for i in range(1, len(tokens)):
            if green_mask[tokens[i].item()]:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

    elif algo_name == "SemStamp":
        target_zone_ratio = 100 / vocab_size
        for i in range(1, len(tokens)):
            if tokens[i].item() < 100:
                green_tokens_count += 1
        effective_count = green_tokens_count * (vocab_size / 2000)
        variance = total_tokens * target_zone_ratio * (1 - target_zone_ratio)
        if variance == 0: return 0.0
        z_score = (effective_count - (total_tokens * target_zone_ratio)) / math.sqrt(variance)
        return min(z_score, 8.5)
    return 0.0

# ================= 数据加载 =================
print(f"\n读取基准数据 {CSV_FILENAME} ...")
try:
    df = pd.read_csv(CSV_FILENAME)
except FileNotFoundError:
    print("找不到 CSV 文件，请确认已生成。")
    exit()

algorithms = [col.replace("Text_", "") for col in df.columns if col.startswith("Text_")]
print(f"目标算法: {algorithms}")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = ['Arial']

def print_table(df, title=""):
    """打印 DataFrame，自动处理 tabulate 缺失"""
    if title:
        print(f"\n=== {title} ===")
    try:
        print(df.to_string())
    except (ImportError, Exception):
        print(df.to_string())

# ================= B4 组件 (测试7共用) =================
class B4ContrastiveProcessor(LogitsProcessor):
    def __init__(self, amateur_model, amateur_tokenizer, origin_model,
                 generation_config, num_beams=10, coef=1.0):
        self.amateur_model = amateur_model
        self.amateur_tokenizer = amateur_tokenizer
        self.origin_model = origin_model
        self.generation_config = generation_config
        self.coef = coef
        self.num_beams = num_beams
        self.amateur_temperature = 1.0

    def prepare_before_generate(self, inputs):
        bsz = inputs['input_ids'].shape[0]
        input_sequence = [""] * bsz
        inputs_empty = self.amateur_tokenizer(input_sequence, return_tensors="pt").to(
            inputs['input_ids'].device)
        n_beams = self.num_beams
        empty_ids = inputs_empty['input_ids'].repeat_interleave(n_beams, dim=0)
        self.mk_amateur = {
            'attention_mask': inputs_empty['attention_mask'].repeat_interleave(n_beams, dim=0).to(self.amateur_model.device)
        }
        self.mk_origin = {
            'attention_mask': inputs_empty['attention_mask'].repeat_interleave(n_beams, dim=0).to(self.origin_model.device)
        }
        self.first_input_ids = empty_ids.to(self.amateur_model.device)
        self.cache_amateur = DynamicCache()
        self.cache_origin = DynamicCache()
        self.prev_input_ids = None
        self.step_num = 0

    def _infer_beam_idx(self, input_ids, prev_input_ids):
        beam_idx = []
        prev_len = prev_input_ids.shape[1]
        last_outputs = input_ids[:, -1 - prev_len:-1]
        for i in range(last_outputs.shape[0]):
            for j in range(prev_input_ids.shape[0]):
                if torch.all(last_outputs[i] == prev_input_ids[j]):
                    beam_idx.append(j)
                    break
            else:
                raise ValueError("Cannot find the corresponding beam index")
        return torch.tensor(beam_idx)

    def __call__(self, input_ids, scores):
        if self.step_num == 0:
            input_ids = self.first_input_ids
        elif self.step_num == 1:
            input_ids = input_ids[:, -1:]
            self.mk_amateur['attention_mask'] = self.mk_amateur['attention_mask'][:, -1:]
            self.mk_origin['attention_mask'] = self.mk_origin['attention_mask'][:, -1:]
        else:
            input_ids = input_ids[:, -self.step_num:]

        if self.prev_input_ids is not None:
            beam_idx = self._infer_beam_idx(input_ids, self.prev_input_ids)
            beam_idx = beam_idx.to(self.amateur_model.device)
            self.cache_amateur = self.cache_amateur.reorder_cache(beam_idx)
            self.cache_origin = self.cache_origin.reorder_cache(beam_idx)

        out_am = self.amateur_model(
            input_ids=input_ids.to(self.amateur_model.device),
            attention_mask=self.mk_amateur['attention_mask'],
            past_key_values=self.cache_amateur, use_cache=True)
        self.cache_amateur = out_am.past_key_values
        scores_am = F.log_softmax(
            out_am.logits[:, -1, :] / self.amateur_temperature, dim=-1).to(scores.device)

        out_or = self.origin_model(
            input_ids=input_ids.to(self.origin_model.device),
            attention_mask=self.mk_origin['attention_mask'],
            past_key_values=self.cache_origin, use_cache=True)
        self.cache_origin = out_or.past_key_values
        scores_or = F.log_softmax(out_or.logits[:, -1, :], dim=-1).to(scores.device)

        self.prev_input_ids = input_ids

        vocab_size = scores_am.shape[-1]
        uniform_prob = 1.0 / vocab_size
        k = 10
        topk_val = torch.topk(scores, k * self.num_beams)[0]
        indices_to_remove = (scores < topk_val[..., -(k - 2) * self.num_beams, None])
        indices_to_keep = (torch.abs(torch.exp(scores_am) - torch.exp(scores_or)) < uniform_prob)
        indices_to_keep |= (scores < topk_val[..., -1, None])

        backup_scores = scores.clone()
        scores = scores.masked_fill(indices_to_keep, float("-inf"))
        scores_am[scores_am == float("-inf")] = float("inf")
        new_scores = (self.coef + 1) * scores - self.coef * scores_am
        new_scores = torch.where(indices_to_keep, backup_scores, new_scores)
        new_scores = new_scores.masked_fill(indices_to_remove, float("-inf"))
        new_scores = F.log_softmax(new_scores, dim=-1)

        del out_am, out_or, scores_am, scores_or
        self.step_num += 1
        return new_scores

B4_PROMPTS = [
    "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Don't output any other information except the paraphrased texts. This is the text:\n{}",
    "You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. This is the text: \n{}",
    "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases.\n{}",
    "Paraphrase the following paragraphs line by line. Don't output any other information except the paraphrased texts. This is the text:\n{}",
    "Paraphrase the following paragraphs line by line. Try to keep the similar length to the original paragraphs. Don't output any other information except the paraphrased texts.\nThis is the text:\n{}",
    "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Paraphrase all sentences one by one. Don't output any other information except the paraphrased texts. This is the text:\n{}",
    "Paraphrase the following paragraph such that it preserves the original meaning but uses different phrasing and vocabulary. Ensure that the new version has minimal overlap with the original in terms of common phrases, word sequences, and n-grams. Output should be natural, coherent, and maintain the key information from the source text. Here are the texts:\n{}",
    "Paraphrase the following paragraph line by line, such that it preserves the original meaning but uses different phrasing and vocabulary. Ensure that the new version has minimal overlap with the original in terms of common phrases, word sequences, and n-grams. Here are the texts:\n{}",
    "Paraphrase the following paragraph such that it preserves the original meaning but has minimal overlap with the original in terms of common phrases, word sequences, and n-grams. Here is the text:\n{}",
    "Paraphrase the following paragraph in your own tone. Ensure that it has minimal overlap with the original in terms of common phrases, word sequences, and n-grams. Here is the texts:\n{}",
    "Rewrite the following paragraph in a way that retains its core meaning but alters its wording and structure. Focus on minimizing shared n-grams and phrases between the original and the rewritten text, while keeping the content clear and coherent. Here are the texts:\n{}",
    "Transform the following paragraph into a new version that conveys the same message but is expressed with different wording and phrasing. Try to keep n-gram overlaps minimal, employing synonyms, rephrased expressions, and varied sentence patterns. Here are the texts:\n{}",
    "Create a paraphrased version of the provided text such that it maintains the semantic essence while minimizing the similarity in wording and n-gram patterns. Focus on using distinct phrases and vocabulary to achieve a high degree of linguistic diversity. Here are the texts:\n{}",
]

def b4_proxy_erasure_attack(texts, paraphrase_model, amateur_model, origin_model,
                              tokenizer, amateur_tokenizer, prompt_id=4,
                              num_beams=10, max_new_tokens=300, coef=1.0, batch_size=8):
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]

    prompt_template = B4_PROMPTS[prompt_id]
    is_gemma = 'gemma' in str(type(paraphrase_model)).lower()
    if is_gemma:
        messages = [[{'role': 'user', 'content': prompt_template.format(t)}] for t in texts]
    else:
        messages = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {'role': 'user', 'content': prompt_template.format(t)}
        ] for t in texts]

    config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=num_beams)

    logits_processor = LogitsProcessorList([
        B4ContrastiveProcessor(
            amateur_model=amateur_model, amateur_tokenizer=amateur_tokenizer,
            origin_model=origin_model, generation_config=config,
            num_beams=num_beams, coef=coef)
    ])

    all_attacked = []
    for i in range(0, len(texts), batch_size):
        batch_msgs = messages[i:i + batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_msgs, return_tensors="pt", padding=True,
            add_generation_prompt=True, return_dict=True
        ).to(paraphrase_model.device)
        inputs_length = inputs.input_ids.shape[-1]

        logits_processor[0].prepare_before_generate(inputs)
        output = paraphrase_model.generate(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
            generation_config=config, logits_processor=logits_processor)
        attacked = tokenizer.batch_decode(output[:, inputs_length:], skip_special_tokens=True)
        all_attacked.extend(attacked)

    return all_attacked[0] if single_input else all_attacked

# ================= 测试头自动生成 =================
def print_test_header(description):
    """从调用脚本文件名自动提取测试编号并统计总数，打印统一格式标题"""
    import inspect as _inspect
    caller_file = _inspect.currentframe().f_back.f_globals.get("__file__", "")
    num = re.search(r"test(\d+)_", os.path.basename(caller_file))
    num = num.group(1) if num else "?"
    total = len(glob.glob(os.path.join(current_dir, "test[0-9]*_*.py")))
    print(f"\n{'='*60}")
    print(f"[测试 {num}/{total}] {description}")
    print("="*60)

print("\n>>> 公共模块加载完成 <<<")
