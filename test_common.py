"""各测试共用的配置、模型和工具函数"""
import sys
import os
import hashlib

# 强制 UTF-8 编码, 解决 Windows 终端 GBK 显示乱码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

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


# ================= 共享水印处理器 =================
class KGWLogitsProcessor(LogitsProcessor):
    """KGW 标准水印: randperm固定大小绿名单 + 独立Generator"""
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


class SmoothedWatermarkLogitsProcessor(LogitsProcessor):
    """平滑水印 — 连续均匀绿度 U(0,1) 替换硬二元掩码, 消除断层"""
    def __init__(self, vocab_size, delta=3.5, hash_key=15485863):
        self.vocab_size = vocab_size; self.delta = delta; self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            g = torch.Generator(device='cpu')
            g.manual_seed(self.hash_key * input_ids[b, -1].item())
            continuous_greenness = torch.rand(self.vocab_size, generator=g)
            scores[b] += self.delta * continuous_greenness.to(scores.device)
        return scores


class PubliclyDetectableProcessor(LogitsProcessor):
    """公开可检测水印 — 私钥生成 + 公钥检测 (非对称密钥架构)"""
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, secret_key=15485863,
                 public_salt=9876543):
        self.vocab_size = vocab_size; self.gamma = gamma
        self.delta = delta; self.secret_key = secret_key
        self.public_salt = public_salt

    def _get_greenlist(self, prev_token, key):
        h = hashlib.sha256(f"{key}_{prev_token}".encode()).digest()
        seed = int.from_bytes(h[:4], 'big') % (2**31 - 1)
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        greenlist_size = int(self.vocab_size * self.gamma)
        return torch.randperm(self.vocab_size, generator=g)[:greenlist_size]

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            greenlist = self._get_greenlist(input_ids[b, -1].item(), self.secret_key)
            scores[b, greenlist.to(scores.device)] += self.delta
        return scores


class _SimpleWatermarkNet(torch.nn.Module):
    """小型神经网络: token上下文 → 绿度分数 (UPV 私钥划分器)"""
    def __init__(self, context_len=1, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(context_len, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1), torch.nn.Sigmoid())
        g = torch.Generator(); g.manual_seed(42)
        for p in self.net.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5, generator=g)

    def forward(self, context_tokens):
        return self.net(context_tokens.float().unsqueeze(0) / 50272.0).squeeze()


class UnforgeableLogitsProcessor(LogitsProcessor):
    """不可伪造水印 — 神经网络权重=私钥, 攻击者无法逆向绿名单划分"""
    def __init__(self, vocab_size, gamma=0.5, delta=2.0):
        self.vocab_size = vocab_size; self.gamma = gamma; self.delta = delta
        self.net = _SimpleWatermarkNet(context_len=1)

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            prev = torch.tensor([input_ids[b, -1].item() % 50272], dtype=torch.float32)
            greenness = self.net(prev)
            effective_gamma = self.gamma * (0.5 + 0.5 * float(greenness))
            greenlist_size = max(1, int(self.vocab_size * effective_gamma))
            hidden_repr = int(float(greenness) * 1e6) + input_ids[b, -1].item()
            g = torch.Generator(device='cpu')
            g.manual_seed(hidden_repr % (2**31 - 1))
            perm = torch.randperm(self.vocab_size, generator=g)
            scores[b, perm[:greenlist_size].to(scores.device)] += self.delta
        return scores


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
        detector_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, cache_dir=CACHE_DIR, trust_remote_code=True)
        target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, cache_dir=CACHE_DIR, trust_remote_code=True).to(device)
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

# ---- 同义词替换 (来自 ASW markllm_editor.SynonymSubstitution) ----
_SYNONYM_DICT = {
    "good": ["great", "excellent", "fine", "superb", "nice", "decent", "wonderful"],
    "bad": ["poor", "terrible", "awful", "dreadful", "lousy", "inferior", "unpleasant"],
    "big": ["large", "huge", "massive", "enormous", "vast", "gigantic", "immense"],
    "small": ["tiny", "little", "miniature", "compact", "petite", "slight", "modest"],
    "happy": ["glad", "pleased", "cheerful", "joyful", "delighted", "content", "thrilled"],
    "sad": ["unhappy", "sorrowful", "gloomy", "miserable", "depressed", "melancholy", "dismal"],
    "fast": ["quick", "rapid", "swift", "speedy", "brisk", "accelerated", "hasty"],
    "slow": ["sluggish", "gradual", "leisurely", "delayed", "prolonged", "tardy", "unhurried"],
    "important": ["significant", "crucial", "vital", "essential", "critical", "paramount", "key"],
    "difficult": ["hard", "challenging", "tough", "complex", "complicated", "arduous", "demanding"],
    "easy": ["simple", "straightforward", "effortless", "uncomplicated", "facile", "light", "smooth"],
    "strong": ["powerful", "robust", "sturdy", "forceful", "mighty", "vigorous", "potent"],
    "weak": ["feeble", "fragile", "frail", "vulnerable", "brittle", "faint", "powerless"],
    "old": ["ancient", "aged", "elderly", "antique", "vintage", "outdated", "mature"],
    "new": ["fresh", "novel", "modern", "recent", "innovative", "original", "current"],
    "beautiful": ["attractive", "pretty", "lovely", "gorgeous", "stunning", "elegant", "handsome"],
    "ugly": ["unattractive", "hideous", "unpleasant", "unsightly", "grotesque", "plain", "homely"],
    "rich": ["wealthy", "affluent", "prosperous", "opulent", "well-off", "abundant", "lavish"],
    "poor": ["impoverished", "needy", "destitute", "underprivileged", "broke", "scarce", "meager"],
    "smart": ["intelligent", "clever", "bright", "brilliant", "sharp", "wise", "astute"],
    "stupid": ["foolish", "dumb", "ignorant", "silly", "unwise", "absurd", "ridiculous"],
    "brave": ["courageous", "bold", "fearless", "valiant", "heroic", "daring", "gallant"],
    "angry": ["furious", "irate", "enraged", "wrathful", "indignant", "annoyed", "irritated"],
    "calm": ["peaceful", "serene", "tranquil", "placid", "relaxed", "composed", "still"],
    "kind": ["benevolent", "compassionate", "gentle", "considerate", "generous", "caring", "sympathetic"],
    "cruel": ["brutal", "ruthless", "merciless", "heartless", "savage", "vicious", "inhumane"],
    "brave": ["courageous", "bold", "fearless", "valiant", "heroic", "daring", "gallant"],
    "honest": ["truthful", "sincere", "frank", "candid", "genuine", "upright", "trustworthy"],
    "brave": ["courageous", "bold", "fearless", "valiant", "heroic", "daring", "gallant"],
    "interesting": ["fascinating", "engaging", "intriguing", "compelling", "captivating", "absorbing"],
    "boring": ["dull", "tedious", "monotonous", "uninteresting", "dreary", "tiresome", "mundane"],
    "happy": ["glad", "pleased", "cheerful", "joyful", "delighted", "content", "thrilled"],
    "sad": ["unhappy", "sorrowful", "gloomy", "miserable", "depressed", "melancholy", "dismal"],
    "certain": ["sure", "definite", "positive", "confident", "assured", "guaranteed", "inevitable"],
    "possible": ["feasible", "achievable", "attainable", "viable", "practicable", "conceivable"],
    "common": ["ordinary", "usual", "frequent", "typical", "regular", "standard", "everyday"],
    "rare": ["uncommon", "scarce", "unusual", "infrequent", "exceptional", "unique", "extraordinary"],
    "careful": ["cautious", "prudent", "vigilant", "wary", "alert", "attentive", "meticulous"],
    "careless": ["reckless", "negligent", "heedless", "rash", "inattentive", "thoughtless", "sloppy"],
    "clear": ["obvious", "evident", "plain", "apparent", "transparent", "distinct", "lucid"],
    "confused": ["bewildered", "perplexed", "puzzled", "baffled", "muddled", "disoriented", "dazed"],
    "deep": ["profound", "abyssal", "bottomless", "fathomless", "intense", "serious", "extreme"],
    "shallow": ["superficial", "surface-level", "skin-deep", "cursory", "lightweight", "trivial"],
    "early": ["premature", "initial", "beginning", "first", "advance", "prior", "preceding"],
    "late": ["tardy", "delayed", "overdue", "belated", "behind", "postponed", "deferred"],
    "empty": ["vacant", "hollow", "blank", "void", "bare", "deserted", "unoccupied"],
    "full": ["filled", "packed", "crowded", "stuffed", "loaded", "saturated", "complete"],
    "gentle": ["mild", "soft", "tender", "light", "moderate", "delicate", "subtle"],
    "rough": ["coarse", "uneven", "rugged", "harsh", "abrasive", "bumpy", "crude"],
    "healthy": ["well", "fit", "robust", "sound", "wholesome", "vigorous", "hearty"],
    "sick": ["ill", "unwell", "ailing", "diseased", "suffering", "indisposed", "unhealthy"],
    "narrow": ["tight", "slim", "slender", "restricted", "limited", "confined", "thin"],
    "wide": ["broad", "extensive", "spacious", "vast", "expansive", "sweeping", "ample"],
    "noisy": ["loud", "boisterous", "clamorous", "rowdy", "vociferous", "raucous", "deafening"],
    "quiet": ["silent", "still", "hushed", "peaceful", "muted", "subdued", "tranquil"],
    "polite": ["courteous", "respectful", "civil", "gracious", "well-mannered", "diplomatic", "refined"],
    "rude": ["impolite", "discourteous", "disrespectful", "insolent", "abrupt", "crass", "boorish"],
    "safe": ["secure", "protected", "shielded", "guarded", "harmless", "reliable", "dependable"],
    "dangerous": ["risky", "hazardous", "perilous", "unsafe", "treacherous", "threatening", "precarious"],
    "same": ["identical", "equivalent", "alike", "uniform", "indistinguishable", "matching"],
    "different": ["distinct", "diverse", "various", "disparate", "dissimilar", "contrasting", "varied"],
    "true": ["accurate", "correct", "valid", "genuine", "authentic", "factual", "legitimate"],
    "false": ["incorrect", "wrong", "untrue", "erroneous", "fake", "bogus", "invalid"],
    "young": ["youthful", "juvenile", "adolescent", "immature", "fresh", "green", "budding"],
    "create": ["generate", "produce", "make", "build", "construct", "develop", "establish"],
    "destroy": ["ruin", "demolish", "wreck", "annihilate", "devastate", "obliterate", "shatter"],
    "improve": ["enhance", "upgrade", "refine", "boost", "advance", "optimize", "strengthen"],
    "reduce": ["decrease", "diminish", "lower", "cut", "lessen", "shrink", "curtail"],
    "increase": ["raise", "elevate", "expand", "grow", "augment", "amplify", "escalate"],
    "begin": ["start", "commence", "initiate", "launch", "inaugurate", "embark", "open"],
    "finish": ["complete", "conclude", "end", "finalize", "terminate", "accomplish", "wrap up"],
    "help": ["assist", "aid", "support", "facilitate", "benefit", "serve", "back"],
    "stop": ["halt", "cease", "quit", "discontinue", "suspend", "terminate", "pause"],
    "show": ["display", "exhibit", "demonstrate", "reveal", "present", "indicate", "manifest"],
    "hide": ["conceal", "cover", "mask", "disguise", "obscure", "camouflage", "veil"],
    "think": ["believe", "consider", "suppose", "assume", "reckon", "ponder", "reflect"],
    "know": ["understand", "comprehend", "grasp", "recognize", "realize", "perceive", "appreciate"],
    "want": ["desire", "wish", "crave", "long for", "yearn", "covet", "seek"],
    "need": ["require", "necessitate", "demand", "call for", "entail", "warrant", "mandate"],
    "give": ["provide", "offer", "supply", "deliver", "grant", "bestow", "donate"],
    "take": ["seize", "grab", "acquire", "obtain", "capture", "collect", "extract"],
    "find": ["discover", "locate", "detect", "uncover", "identify", "spot", "trace"],
    "lose": ["misplace", "forfeit", "surrender", "yield", "relinquish", "drop", "suffer loss"],
    "change": ["alter", "modify", "transform", "convert", "adjust", "revise", "reshape"],
    "keep": ["retain", "maintain", "preserve", "hold", "guard", "safeguard", "sustain"],
    "believe": ["trust", "accept", "credit", "deem", "hold", "maintain", "presume"],
    "explain": ["clarify", "elucidate", "describe", "illustrate", "interpret", "define", "expound"],
    "understand": ["comprehend", "grasp", "apprehend", "fathom", "discern", "perceive"],
    "remember": ["recall", "recollect", "reminisce", "retrieve", "recognize", "bear in mind"],
    "forget": ["overlook", "neglect", "omit", "disregard", "ignore", "abandon", "dismiss"],
    "allow": ["permit", "authorize", "enable", "let", "sanction", "approve", "empower"],
    "prevent": ["stop", "block", "hinder", "impede", "prohibit", "restrict", "thwart"],
    "include": ["contain", "incorporate", "encompass", "comprise", "cover", "embrace", "involve"],
    "exclude": ["omit", "eliminate", "remove", "reject", "bar", "ban", "expel"],
    "achieve": ["attain", "accomplish", "realize", "reach", "fulfill", "complete", "execute"],
    "fail": ["flop", "collapse", "founder", "miscarry", "fall short", "come up short"],
    "agree": ["concur", "consent", "accede", "approve", "endorse", "align", "harmonize"],
    "disagree": ["differ", "dissent", "dispute", "object", "oppose", "contradict", "conflict"],
    "love": ["adore", "cherish", "treasure", "worship", "devote", "revere", "idolize"],
    "hate": ["detest", "despise", "loathe", "abhor", "resent", "scorn", "disdain"],
    "speak": ["talk", "say", "utter", "articulate", "express", "voice", "declare"],
    "write": ["compose", "draft", "pen", "jot", "record", "scribble", "inscribe"],
    "read": ["peruse", "scan", "study", "browse", "skim", "review", "examine"],
    "learn": ["study", "acquire", "master", "grasp", "absorb", "assimilate", "educate"],
    "teach": ["instruct", "educate", "train", "guide", "coach", "mentor", "enlighten"],
    "ask": ["inquire", "question", "query", "request", "solicit", "interrogate", "probe"],
    "answer": ["reply", "respond", "retort", "rejoin", "acknowledge", "address", "counter"],
    "buy": ["purchase", "acquire", "procure", "obtain", "shop for", "invest in", "secure"],
    "sell": ["vend", "market", "trade", "peddle", "hawk", "auction", "retail"],
    "build": ["construct", "erect", "assemble", "fabricate", "forge", "raise", "establish"],
    "break": ["shatter", "fracture", "crack", "smash", "split", "crush", "rupture"],
    "open": ["unlock", "unseal", "unwrap", "unfold", "reveal", "expose", "access"],
    "close": ["shut", "seal", "lock", "fasten", "secure", "conclude", "end"],
    "win": ["triumph", "prevail", "succeed", "conquer", "vanquish", "overcome", "dominate"],
    "lose": ["suffer defeat", "fall short", "be beaten", "yield", "capitulate", "succumb"],
    "lead": ["guide", "direct", "steer", "pilot", "command", "head", "govern"],
    "follow": ["pursue", "trail", "shadow", "track", "trace", "observe", "obey"],
    "grow": ["expand", "develop", "flourish", "thrive", "prosper", "mature", "bloom"],
    "die": ["perish", "expire", "pass away", "decease", "depart", "fade", "vanish"],
    "like": ["enjoy", "appreciate", "fancy", "favor", "relish", "admire", "prefer"],
    "fear": ["dread", "terror", "anxiety", "worry", "panic", "alarm", "fright"],
}

def synonym_substitution_attack(text, ratio=0.3):
    """同义词替换攻击 (来自 ASW markllm_editor.SynonymSubstitution).
    ratio: 替换比例, 0.0~1.0"""
    if not isinstance(text, str) or not text.strip():
        return text
    words = text.split()
    num_words = len(words)
    if num_words == 0:
        return text

    # 找出可替换的词索引
    replaceable = []
    word_lower_map = {}
    for i, w in enumerate(words):
        clean = w.strip(".,;:!?\"'()[]{}").lower()
        if clean in _SYNONYM_DICT:
            replaceable.append(i)
            word_lower_map[i] = clean

    if not replaceable:
        return text

    num_to_replace = max(1, int(min(ratio, len(replaceable) / num_words) * num_words))
    num_to_replace = min(num_to_replace, len(replaceable))
    indices = random.sample(replaceable, num_to_replace)

    new_words = words[:]
    for i in indices:
        clean_lower = word_lower_map[i]
        synonyms = _SYNONYM_DICT[clean_lower]
        replacement = random.choice(synonyms)
        # 保留原词的大写/首字母大写
        orig = words[i]
        if orig.isupper():
            replacement = replacement.upper()
        elif orig[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        new_words[i] = replacement

    return " ".join(new_words)

# ---- 复制粘贴攻击 (来自 ASW markllm_editor.CopyPasteAttack) ----
def copy_paste_attack(text, reference_text=None, ratio=0.2):
    """从无 watermark 的参考文本中随机复制句子替换 watermark 文本中的句子.
    (来自 ASW markllm_editor.CopyPasteAttack)"""
    if not isinstance(text, str) or not text.strip():
        return text
    if not reference_text:
        return text

    wm_sents = re.split(r'(?<=[.!?])\s+', text)
    ref_sents = re.split(r'(?<=[.!?])\s+', reference_text)

    if len(wm_sents) <= 1 or len(ref_sents) == 0:
        return text

    change_count = max(1, int(len(wm_sents) * ratio))
    change_count = min(change_count, len(ref_sents))
    replace_ids = random.sample(range(len(wm_sents)), min(change_count, len(wm_sents)))
    replace_texts = random.sample(ref_sents, change_count)

    for idx, rep_sent in zip(replace_ids, replace_texts):
        wm_sents[idx] = rep_sent

    return " ".join(wm_sents)

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
        g = torch.Generator(device='cpu')
        g.manual_seed(secret_key * prev if algo_name in ["KGW", "SWEET", "DiPmark"] else 42)
        greenlist_size = int(vocab_size * gamma)
        vocab_permutation = torch.randperm(vocab_size, generator=g)
        greenlist = set(vocab_permutation[:greenlist_size].tolist())
        is_green = curr in greenlist

        if attack_type == "scrubbing" and is_green and random.random() < 0.6:
            candidate = (curr + random.randint(1, 100)) % vocab_size
            while candidate in greenlist:
                candidate = (candidate + 1) % vocab_size
            tampered_tokens.append(candidate)
        elif attack_type == "spoofing" and not is_green and random.random() < 0.6:
            candidate = (curr + random.randint(1, 100)) % vocab_size
            while candidate not in greenlist:
                candidate = (candidate + 1) % vocab_size
            tampered_tokens.append(candidate)
        else:
            tampered_tokens.append(curr)
    return tokenizer.decode(tampered_tokens, skip_special_tokens=True)

# ================= SIRA 组件 (对齐 Self-information-Rewrite-Attack 三阶段流水线) =================
def calculate_token_self_information(text, model, tokenizer, device):
    """Stage 2 核心: 计算每个 token 的自信息 (surprisal). 参考 SIRA SelfInformationCalculator"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    if input_ids.shape[1] <= 1: return [], []
    with torch.no_grad(): outputs = model(input_ids)
    logits = outputs.logits[0, :-1, :]
    target_ids = input_ids[0, 1:]
    probs = torch.softmax(logits, dim=-1)  # 对齐参考: 使用自然对数
    token_probs = probs.gather(1, target_ids.unsqueeze(1)).squeeze(-1)
    self_info = -torch.log(token_probs + 1e-10)
    tokens = tokenizer.convert_ids_to_tokens(target_ids)
    return tokens, self_info.cpu().numpy()

def sira_generate_reference(text):
    """Stage 1: 改写水印文本生成"参考文本".
    对齐 SIRA attack_onestep.fill_parapharse_prompt.
    使用 T5 的 paraphrase 前缀作为轻量替代."""
    load_attacker()
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    input_ids = attacker_tokenizer("paraphrase: " + text, return_tensors="pt",
                                    max_length=512, truncation=True).input_ids.to(device)
    with torch.no_grad():
        outputs = attacker_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    return attacker_tokenizer.decode(outputs[0], skip_special_tokens=True)

def sira_masking(text, model, tokenizer, threshold_percentile=30, device="cuda"):
    """Stage 2: 自信息驱动的空白化.
    对齐 SIRA SelfInformationCalculator.transform_tokens:
    - 低自信息 token (≤ P_threshold) → 保留, 作为上下文线索
    - 高自信息 token (> P_threshold) → 替换为 <extra_id_N> 等待填充
    连续低自信息块 → 合并, 连续高自信息块 → 合并为一个空白位
    threshold_percentile: 默认 30, 即保留自信息最低的 30% token"""
    tokens, self_info_scores = calculate_token_self_information(text, model, tokenizer, device)
    if not tokens or len(tokens) < 5:
        return text

    cutoff = np.percentile(self_info_scores, threshold_percentile)
    # low_si = 自信息 <= cutoff (保留), high_si = 自信息 > cutoff (遮蔽)
    mask_flags = self_info_scores > cutoff

    masked_tokens = []
    mask_count = 0
    i = 0
    while i < len(tokens):
        if mask_flags[i]:
            # 高自信息块: 合并连续高自信息 token 为一个空白位
            masked_tokens.append(f"<extra_id_{mask_count}>")
            mask_count += 1
            while i < len(tokens) and mask_flags[i]:
                i += 1
        else:
            # 低自信息块: 保留作为上下文 (保留原始 token, 不 strip Ġ/Ċ)
            masked_tokens.append(tokens[i])
            i += 1

    blank_text = "".join(masked_tokens).strip()
    return blank_text

def sira_attack_prompt(reference_text, blank_text):
    """Stage 3 攻击 prompt: 参考文本 + 空白化文本 → 引导 LLM 填充.
    对齐 SIRA fill_attack_prompt"""
    prompt = (
        "You will be shown one reference paragraph and one incomplete paragraph.\n"
        "Your task is to write a complete paragraph using incomplete paragraph.\n"
        "The complete paragraph should have similar length with reference paragraph.\n"
        "You need to include all the information in the reference.\n"
        "But do not take the expression and words in the reference paragraph.\n"
        "You should only answer the complete paragraph.\n"
        f"reference: {reference_text}\n"
        f"incomplete paragraph: {blank_text}\n"
    )
    return prompt

def sira_t5_infilling(masked_text, reference_text=None):
    """Stage 3: T5 文本填充.
    如果有 reference_text, 将其作为上下文拼接到输入中辅助 T5 填充空白位;
    否则回退到直接填充空白位."""
    load_attacker()
    if "<extra_id_" not in masked_text:
        return masked_text

    if reference_text:
        # T5 seq2seq 友好格式: 参考文本作为额外上下文
        input_str = f"paraphrase: {reference_text} context: {masked_text}"
    else:
        input_str = masked_text

    inputs = attacker_tokenizer(input_str, return_tensors="pt",
                                 max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = attacker_model.generate(**inputs, max_length=512, num_beams=4,
                                          temperature=0.8, do_sample=True, early_stopping=True)
    filled_content = attacker_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 从 T5 输出中提取并替换 <extra_id_N> 占位符
    final_text = masked_text
    parts = [p.strip() for p in filled_content.split("<extra_id_") if p.strip()]
    for i, part in enumerate(parts):
        if '>' in part:
            clean_part = part.split('>', 1)[1].strip()
            final_text = final_text.replace(f"<extra_id_{i}>", " " + clean_part + " ")
    final_text = re.sub(r'<extra_id_\d+>', '', final_text)
    return final_text.replace("  ", " ").strip()

# ================= 检测器 =================
def detect_sweet(text, model, tokenizer, vocab_size, gamma=0.5,
                 entropy_threshold=1.5, secret_key=15485863, device="cuda"):
    """SWEET 熵感知检测器 — 对齐 sweet.py SweetDetector._score_sequence

    核心差异 vs KGW:
      1. 用模型逐token计算熵 (一次前向传播)
      2. 排除低熵token (e ≤ entropy_threshold) — 不计入T, 不统计green_count
      3. z = (green_count - γ×T_scored) / sqrt(T_scored×γ×(1-γ))
      4. 若 scored_tokens<1 → 返回 -100 (视为人类生成)
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return -100.0
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    T = len(tokens) - 1
    if T <= 0:
        return -100.0

    # 逐token熵 (对齐 lm_eval/utils.py calculate_entropy + evaluator.py)
    with torch.no_grad():
        input_ids = tokens.unsqueeze(0).to(device)
        outputs = model(input_ids)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        ent = -torch.where(probs > 0, probs * probs.log(),
                           torch.tensor(0.0, device=device)).sum(dim=-1)
    entropy = ent.cpu().tolist()
    # 右移: 位置i的熵预测token_{i+1}, 前缀处填0
    entropy = [0.0] + entropy[:-1]

    green_count = 0
    scored_tokens = 0
    for i in range(1, len(tokens)):
        e = entropy[i]
        if e <= entropy_threshold:
            continue
        scored_tokens += 1

        g = torch.Generator(device='cpu')
        g.manual_seed(secret_key * tokens[i - 1].item())
        greenlist_size = int(vocab_size * gamma)
        vocab_permutation = torch.randperm(vocab_size, generator=g)
        greenlist = vocab_permutation[:greenlist_size]
        if tokens[i] in greenlist:
            green_count += 1

    if scored_tokens < 1:
        return -100.0

    expected = gamma * scored_tokens
    variance = scored_tokens * gamma * (1 - gamma)
    return (green_count - expected) / math.sqrt(variance) if variance > 0 else 0.0


def detect_watermark(text, algo_name, tokenizer, vocab_size, gamma=0.5,
                     secret_key=15485863, model=None, device="cuda"):
    """统一检测入口 — 按算法路由

    model 和 device 参数仅对 SWEET 必须 (需要模型计算每token熵).
    其他算法忽略这两个参数.
    """
    if algo_name == "Natural": return 0.0
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    total_tokens = len(tokens) - 1
    if total_tokens <= 0: return 0.0
    green_tokens_count = 0

    if algo_name == "SWEET":
        if model is None:
            load_detector()
            model = target_model
        return detect_sweet(text, model, tokenizer, vocab_size,
                            gamma=gamma, secret_key=secret_key, device=device)

    elif algo_name in ["KGW", "DiPmark"]:
        for i in range(1, len(tokens)):
            g = torch.Generator(device='cpu')
            g.manual_seed(secret_key * tokens[i - 1].item())
            greenlist_size = int(vocab_size * gamma)
            vocab_permutation = torch.randperm(vocab_size, generator=g)
            greenlist = vocab_permutation[:greenlist_size]
            if tokens[i] in greenlist:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

    elif algo_name == "Unigram":
        g = torch.Generator(device='cpu')
        g.manual_seed(42)
        greenlist_size = int(vocab_size * gamma)
        vocab_permutation = torch.randperm(vocab_size, generator=g)
        greenlist = set(vocab_permutation[:greenlist_size].tolist())
        for i in range(1, len(tokens)):
            if tokens[i].item() in greenlist:
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
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

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
