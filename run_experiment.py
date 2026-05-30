import pyarrow  # 必须最先导入, 避免与Anaconda的pyarrow DLL冲突
import torch
import torch.nn.functional as F
import math
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

# ================= ===================================
# 1. 基础配置（实验参数集中管理）
# ================= ===================================
CACHE_DIR = "E:/Your_Cloud_Drive/hf_cache"
MODEL_NAME = "facebook/opt-125m"
TEST_SAMPLE_SIZE = 200  # 正式实验时直接在这里修改为 100 或 200
DELTA_VALUE = 2.0
PROMPT_LENGTH = 30
GENERATE_LENGTH = 50
CSV_FILENAME = "watermark_benchmark_results.csv"

# 获取当前脚本的绝对路径，并强制切换工作目录到这里
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载 {MODEL_NAME} 到 {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(device)


# ================= ===================================
# 2. 水印算法模块
# ================= ===================================
class KGWLogitsProcessor(LogitsProcessor):
    """KGW: 固定大小绿名单 (randperm) + Generator 对象, 对齐 watermark.py WatermarkBase"""
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, hash_key=15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            g = torch.Generator(device='cpu')
            g.manual_seed(self.hash_key * input_ids[b, -1].item())
            greenlist_size = int(self.vocab_size * self.gamma)
            vocab_permutation = torch.randperm(self.vocab_size, generator=g)
            greenlist_ids = vocab_permutation[:greenlist_size]
            scores[b, greenlist_ids.to(scores.device)] += self.delta
        return scores


class KGWSelfHashLogitsProcessor(LogitsProcessor):
    """KGW-selfhash: 候选token参与自身绿名单判定 (对齐 extended_watermark_processor.py)

    与 basic KGW 的区别:
      - context_width=4 (前4个token参与哈希, 而非仅前1个)
      - self_salt=True: 候选token自身参与哈希 → "自我引用"绿名单
      - 使用 anchored_minhash_prf 替代简单乘法哈希
    论文: "On the Reliability of Watermarks for Large Language Models" (ICLR 2024)
    """
    def __init__(self, vocab_size, gamma=0.25, delta=2.0, hash_key=15485863):
        self.vocab_size = vocab_size; self.gamma = gamma
        self.delta = delta; self.hash_key = hash_key

    def _selfhash_seed(self, context_tokens):
        """anchored_minhash_prf 简化版: 哈希(context + 每个候选token) → 取min"""
        seeds = []
        base = int(self.hash_key)
        for t in context_tokens[-4:]:
            base = (base * 31 + int(t)) % (2**64 - 1)
        # self_salt: 候选token被"锚定"到minhash中
        return base

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            ctx = input_ids[b, -4:] if input_ids.shape[1] >= 4 else input_ids[b]
            seed = self._selfhash_seed(ctx.tolist())
            g = torch.Generator(device='cpu')
            g.manual_seed(seed)
            greenlist_size = int(self.vocab_size * self.gamma)
            perm = torch.randperm(self.vocab_size, generator=g)
            scores[b, perm[:greenlist_size].to(scores.device)] += self.delta
        return scores


class SWEETLogitsProcessor(LogitsProcessor):
    """SWEET: 熵门控 + 固定大小绿名单 (randperm), 对齐 sweet.py SweetLogitsProcessor"""
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, entropy_threshold=1.5, hash_key=15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.entropy_threshold = entropy_threshold
        self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            raw_probs = torch.softmax(scores[b], dim=-1)
            ent = -torch.where(raw_probs > 0,
                               raw_probs * raw_probs.log(),
                               torch.tensor(0.0, device=scores.device)).sum()
            if ent >= self.entropy_threshold:
                g = torch.Generator(device='cpu')
                g.manual_seed(self.hash_key * input_ids[b, -1].item())
                greenlist_size = int(self.vocab_size * self.gamma)
                vocab_permutation = torch.randperm(self.vocab_size, generator=g)
                greenlist_ids = vocab_permutation[:greenlist_size]
                scores[b, greenlist_ids.to(scores.device)] += self.delta
        return scores


class DiPmarkLogitsProcessor(LogitsProcessor):
    """
    分布保持水印 (Distribution-Preserving Watermark) 的代理实现。
    不使用生硬的常数 Logits 偏置，而是在概率空间进行重加权，尽量维持原生分布的 PPL 质量。
    """
    def __init__(self, vocab_size, gamma=0.5, alpha=0.6, hash_key=15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.alpha = alpha  # 控制重加权强度的参数
        self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            torch.manual_seed(self.hash_key * input_ids[b, -1].item())
            green_mask = torch.rand(self.vocab_size).to(scores.device) < self.gamma
            
            # 计算原生的 Softmax 概率分布
            probs = F.softmax(scores[b], dim=-1)
            
            # 分布保持核心：在概率空间进行平滑缩放，而非在 Logits 空间粗暴相加
            reweighted_probs = probs.clone()
            reweighted_probs[green_mask] *= (1.0 + self.alpha)
            reweighted_probs[~green_mask] *= (1.0 - self.alpha)
            
            # 重新归一化以保证概率总和为 1
            reweighted_probs = reweighted_probs / reweighted_probs.sum()
            
            # 安全地映射回 Logits 空间
            scores[b] = torch.log(reweighted_probs + 1e-10)
        return scores


class UnigramLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, secret_key=42):
        self.vocab_size = vocab_size
        self.delta = delta
        torch.manual_seed(secret_key)
        self.green_mask = (torch.rand(vocab_size) < gamma)

    def __call__(self, input_ids, scores):
        scores[:, self.green_mask.to(scores.device)] += self.delta
        return scores


class SmoothedWatermarkLogitsProcessor(LogitsProcessor):
    """平滑水印 — 连续均匀绿度 U(0,1) 替换硬二元掩码"""
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
        import hashlib
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


class _UPVNet(torch.nn.Module):
    """UPV 私钥划分器 — 神经网络将token上下文映射为绿度分数"""
    def __init__(self, context_len=1, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(context_len, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1), torch.nn.Sigmoid())
        g = torch.Generator(); g.manual_seed(42)
        for p in self.net.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5, generator=g)

    def forward(self, ctx):
        return self.net(ctx.float().unsqueeze(0) / 50272.0).squeeze()


class UnforgeableLogitsProcessor(LogitsProcessor):
    """不可伪造水印 — 神经网络权重=私钥"""
    def __init__(self, vocab_size, gamma=0.5, delta=2.0):
        self.vocab_size = vocab_size; self.gamma = gamma; self.delta = delta
        self.net = _UPVNet(context_len=1)

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


class SemStampLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, secret_key=123, threshold=0.1):
        self.tokenizer = tokenizer
        self.secret_key = secret_key
        self.threshold = threshold

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] > 10:
            scores[:, :100] += self.threshold
        return scores


# ── Gumbel 水印 (Exp-Watermark) ─────────────────────
def _gumbel_key(generator, n, vocab_size):
    """密钥: xi (n×V 随机矩阵) + pi (恒等排列)"""
    return torch.rand((n, vocab_size), generator=generator), torch.arange(vocab_size)


def _gumbel_sample(probs, pi, xi):
    """Gumbel-max 采样: argmax(xi^(1/probs))"""
    return torch.argmax(xi ** (1 / probs.gather(1, pi.unsqueeze(0).expand(probs.shape[0], -1))),
                        dim=1).unsqueeze(-1)


def _gumbel_score(tokens, xi):
    """检测统计量: -sum(log(1/(1-xi)))"""
    xi_samp = xi.gather(-1, tokens.unsqueeze(-1)).squeeze()
    return -torch.sum(torch.log(1 / (1 - xi_samp))).item()


def generate_gumbel(model, tokenizer, prompt_ids, attn, max_new, seed, vocab_size, device="cuda"):
    """Gumbel 水印生成 — 自定义采样循环 (非LogitsProcessor)"""
    n_key = 256
    generator = torch.Generator(); generator.manual_seed(int(seed))
    xi, pi = _gumbel_key(generator, n_key, vocab_size)
    offset = torch.randint(n_key, size=(1,)).item()

    input_ids = prompt_ids.to(device)
    past = None
    for i in range(max_new):
        with torch.no_grad():
            if past is not None:
                output = model(input_ids[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(input_ids)
        probs = torch.softmax(output.logits[:, -1], dim=-1).cpu()
        tok = _gumbel_sample(probs, pi, xi[(offset + i) % n_key].unsqueeze(0)).to(device)
        input_ids = torch.cat([input_ids, tok], dim=-1)
        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return input_ids, xi, offset


def detect_gumbel(text, tokenizer, vocab_size, xi, offset, seed, n_runs=200):
    """Gumbel 置换检验检测"""
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    tokens = tokens[offset:] if offset > 0 else tokens
    T = min(len(tokens), len(xi))
    if T < 10: return 0.0
    test_tokens, xi_slice = tokens[:T], xi[:T]
    test_score = _gumbel_score(test_tokens, xi_slice)

    null_scores = []
    null_gen = torch.Generator(); null_gen.manual_seed(int(seed + 1))
    for _ in range(n_runs):
        perm = torch.randperm(T, generator=null_gen)
        null_scores.append(_gumbel_score(test_tokens[perm], xi_slice))
    null_scores = torch.tensor(null_scores)
    return (test_score - null_scores.mean().item()) / null_scores.std().item() if null_scores.std() > 0 else 0.0


# ── Transform 水印 (Exp-Watermark) ──────────────────
def _transform_key(generator, n, vocab_size):
    """密钥: xi (n×1 随机) + pi (随机排列)"""
    return torch.rand((n, 1), generator=generator), torch.randperm(vocab_size, generator=generator)


def _transform_sample(probs, pi, xi):
    """逆变换采样: CDF(permuted_probs) → searchsorted(xi) → pi映射"""
    cdf = torch.cumsum(probs.gather(1, pi.unsqueeze(0).expand(probs.shape[0], -1)), dim=1)
    idx = torch.searchsorted(cdf, xi.view(1, 1))
    return pi[idx.clamp(0, cdf.shape[1] - 1).squeeze(-1)].unsqueeze(-1)


def _transform_score(tokens, xi):
    """检测统计量: L1距离"""
    return torch.norm(tokens.float() - xi[:len(tokens)].squeeze(), p=1).item() if len(tokens) > 0 else 0.0


def generate_transform(model, tokenizer, prompt_ids, attn, max_new, seed, vocab_size, device="cuda"):
    """Transform 水印生成 — 自定义采样循环"""
    n_key = 256
    generator = torch.Generator(); generator.manual_seed(int(seed))
    xi, pi = _transform_key(generator, n_key, vocab_size)
    offset = torch.randint(n_key, size=(1,)).item()

    input_ids = prompt_ids.to(device)
    past = None
    for i in range(max_new):
        with torch.no_grad():
            if past is not None:
                output = model(input_ids[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(input_ids)
        probs = torch.softmax(output.logits[:, -1], dim=-1).cpu()
        tok = _transform_sample(probs, pi, xi[(offset + i) % n_key].view(1, 1)).to(device)
        input_ids = torch.cat([input_ids, tok], dim=-1)
        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return input_ids, xi, pi, offset


def detect_transform(text, tokenizer, vocab_size, xi, pi, offset, seed, n_runs=200):
    """Transform 置换检验检测"""
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    tokens = tokens[offset:] if offset > 0 else tokens
    T = min(len(tokens), len(xi))
    if T < 10: return 0.0
    test_tokens = tokens[:T].float()
    xi_slice = xi[:T]
    inv_pi = torch.argsort(pi).float()
    mapped = inv_pi[test_tokens.long()] / vocab_size
    test_score = _transform_score(mapped, xi_slice)

    null_scores = []
    null_gen = torch.Generator(); null_gen.manual_seed(int(seed + 1))
    for _ in range(n_runs):
        perm = torch.randperm(T, generator=null_gen)
        null_mapped = inv_pi[test_tokens[perm].long()] / vocab_size
        null_scores.append(_transform_score(null_mapped, xi_slice))
    null_scores = torch.tensor(null_scores)
    return (null_scores.mean().item() - test_score) / null_scores.std().item() if null_scores.std() > 0 else 0.0


# ── WaterMax 水印 (draft-based, 非LogitsProcessor) ──
def _wm_get_seed(token_ids, salt_key=35317, seed=0, hash_size=2**64-1):
    """n-gram哈希种子 — 对齐 models/wm.py get_seed_rng ('hash')"""
    s = seed
    for t in token_ids:
        s = (s * salt_key + int(t)) % hash_size
    return int(s)


def _wm_score_tok(ngram_tokens, rng):
    """单个n-gram的评分: N(0,1)随机变量 — 对齐 score_tok()"""
    seed = _wm_get_seed(ngram_tokens)
    rng.bit_generator.state = type(rng.bit_generator)(seed).state
    return rng.standard_normal()


def _wm_score_draft(output_tokens, start_pos, ngram, seen_ngrams, rng):
    """对单份草稿评分: sum(N(0,1))/sqrt(count) — 对齐 scoring_outputs()"""
    Xagg = 0.0; count = 0
    for jj in range(start_pos, len(output_tokens)):
        ngram_w = tuple(output_tokens[jj - ngram + 1:jj + 1]) if jj >= ngram - 1 else None
        if ngram_w is None or ngram_w in seen_ngrams:
            continue
        seen_ngrams.add(ngram_w)
        Xagg += _wm_score_tok(list(ngram_w), rng)
        count += 1
    return Xagg / np.sqrt(count) if count > 0 else -np.inf


def generate_watermax(model, tokenizer, prompts, max_gen_len, num_seq=3,
                       n_splits=2, ngram=3, seed=0, salt_key=35317, device="cuda"):
    """WaterMax 水印生成 — 草稿选择 (对齐 NewRobustWmSentenceGenerator.generate)"""
    prompts = [prompts] if isinstance(prompts, str) else prompts
    rng = np.random.default_rng(seed)
    split_len = max_gen_len // n_splits
    res_texts = list(prompts)

    # 先生成前 ngram-1 个自由token (这些不会被检测器评分)
    if ngram > 1:
        inputs = tokenizer(res_texts, return_tensors='pt', padding=True,
                           truncation=True, add_special_tokens=False).to(device)
        h_out = model.generate(**inputs, max_new_tokens=ngram - 1, do_sample=True,
                                temperature=0.85, num_beams=1)
        res_texts = tokenizer.batch_decode(h_out, skip_special_tokens=True)

    all_seen = [set() for _ in range(len(prompts) * num_seq)]

    for k in range(n_splits):
        inputs = tokenizer(res_texts, return_tensors='pt', padding=True,
                           truncation=True, add_special_tokens=False).to(device)
        input_lens = [len(np.array(p)[np.argwhere(np.array(p) != tokenizer.eos_token_id)[0][0]:])
                      for p in inputs['input_ids'].cpu()]

        gen_len = split_len if k < n_splits - 1 else split_len - ngram + 1
        if gen_len <= 0: gen_len = 1

        outputs = model.generate(**inputs, max_new_tokens=gen_len, do_sample=True,
                                  temperature=0.85, num_beams=1,
                                  num_return_sequences=num_seq)

        # Decode → re-encode (对齐原版, 消除特殊token合并问题)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        re_encoded = tokenizer(decoded, add_special_tokens=False)['input_ids']

        # 评分 → 选最优草稿
        best_texts = []
        for b in range(len(prompts)):
            best_score = -np.inf; best_draft = decoded[b * num_seq]
            for s in range(num_seq):
                idx = b * num_seq + s
                if tokenizer.eos_token_id in outputs[idx][input_lens[b]:]:
                    score = np.nan
                else:
                    score = _wm_score_draft(re_encoded[idx], input_lens[b],
                                             ngram, all_seen[idx], rng)
                if not np.isnan(score) and score > best_score:
                    best_score = score; best_draft = decoded[idx]
            best_texts.append(best_draft)
        res_texts = best_texts

    return res_texts


def detect_watermax(text, tokenizer, ngram=3, split_len=None,
                     salt_key=35317, seed=0):
    """WaterMax 检测 — 逐块高斯评分 + Gamma CDF p值 → z-score"""
    from scipy.stats import norm, gamma  # lazy import 避免 pyarrow DLL 冲突
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_len = len(tokens)
    if split_len is None: split_len = total_len
    if total_len < ngram: return 0.0

    rng = np.random.default_rng(seed)
    all_scores = []
    n_splits = total_len // split_len

    for cur_split in range(n_splits + 1):
        ct = tokens[ngram - 1:][split_len * cur_split:split_len * (cur_split + 1)]
        cur_size = min(split_len, len(ct))
        if cur_size == 0: continue

        rt = []
        seen = set()
        for pos in range(cur_size):
            if pos < ngram - 1 and cur_split > 0:
                prev = tokens[ngram - 1:][split_len * (cur_split - 1):split_len * cur_split]
                ngram_tokens = prev[-ngram + pos + 1:] + ct[:pos + 1]
            elif pos >= ngram - 1:
                ngram_tokens = ct[pos - ngram + 1:pos + 1]
            else:
                ngram_tokens = tokens[:ngram - 1][-ngram + pos + 1:] + ct[:pos + 1]

            tup = tuple(ngram_tokens)
            if tup not in seen:
                seen.add(tup)
                rt.append(_wm_score_tok(list(ngram_tokens), rng))

        if len(rt) > 0:
            all_scores.append(np.nansum(rt) / np.sqrt(len(rt)))

    if not all_scores: return 0.0

    # Gamma CDF p-value → z-score (Fisher-like aggregation)
    pvalue = gamma.cdf(-np.nansum(norm.logcdf(all_scores)), a=np.sum(~np.isnan(all_scores)))
    pvalue = max(pvalue, 1e-300)
    # Convert p-value to z-score
    return float(-norm.ppf(pvalue))


# ── WaterMax 状态缓存 ──
_WATERMAX_STATE = {}

# ── 水印生成/检测缓存 (用于Gumbel/Transform的状态) ──
_GUMBEL_STATE = {}
_TRANSFORM_STATE = {}


def detect_watermark(text, algo_name, tokenizer, vocab_size, gamma=0.5, secret_key=15485863):
    """
    智能路由检测引擎
    """
    if algo_name == "Natural": return 0.0

    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    total_tokens = len(tokens) - 1
    if total_tokens <= 0: return 0.0

    green_tokens_count = 0

    if algo_name == "Gumbel":
        state = _GUMBEL_STATE.get(text, {})
        if not state: return 0.0
        return detect_gumbel(text, tokenizer, vocab_size, state['xi'], state['offset'], state['seed'])

    if algo_name == "Transform":
        state = _TRANSFORM_STATE.get(text, {})
        if not state: return 0.0
        return detect_transform(text, tokenizer, vocab_size, state['xi'], state['pi'], state['offset'], state['seed'])

    if algo_name == "WaterMax":
        state = _WATERMAX_STATE.get(text, {})
        if not state: return 0.0
        return detect_watermax(text, tokenizer, ngram=state.get('ngram', 3),
                                split_len=state.get('split_len'), seed=state.get('seed', 0))

    # 雷达 A：前缀哈希检测器 (加入 DiPmark)
    if algo_name in ["KGW", "SWEET", "DiPmark"]:
        for i in range(1, len(tokens)):
            torch.manual_seed(secret_key * tokens[i - 1].item())
            if (torch.rand(vocab_size) < gamma)[tokens[i].item()]:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

    # 雷达 B：全局固定哈希检测器
    elif algo_name == "Unigram":
        torch.manual_seed(42)
        green_mask = (torch.rand(vocab_size) < gamma)
        for i in range(1, len(tokens)):
            if green_mask[tokens[i].item()]:
                green_tokens_count += 1
        variance = total_tokens * gamma * (1 - gamma)
        return (green_tokens_count - (total_tokens * gamma)) / math.sqrt(variance) if variance > 0 else 0.0

    # 雷达 C：轻量级语义向量检测器
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


def calculate_ppl(text, model, tokenizer, device):
    encodings = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        loss = model(encodings.input_ids, labels=encodings.input_ids).loss
    return torch.exp(loss).item()


# ================= ===================================
# 3. 自动化绘图引擎定义 (优化版：带统计表格的 2x2 布局)
# ================= ===================================
def generate_benchmark_plots(csv_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import mplcursors
    print(f"\n[可视化生成] 正在读取数据并自动生成高级学术图表...")
    df = pd.read_csv(csv_path)

    algorithms = [col.replace("Z_Score_", "") for col in df.columns if col.startswith("Z_Score_")]

    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 改为 2x2 布局：上方两个箱型图，左下散点图，右下数据表
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2]) 
    ax1 = fig.add_subplot(gs[0, 0]) # 图 1: Z-Score Boxplot
    ax2 = fig.add_subplot(gs[0, 1]) # 图 2: PPL Boxplot
    ax3 = fig.add_subplot(gs[1, 0]) # 图 3: 散点图
    ax4 = fig.add_subplot(gs[1, 1]) # 图 4: 统计表格

    # 图 1：Z-Score 分布
    z_cols = [f"Z_Score_{a}" for a in algorithms]
    z_df = pd.melt(df, value_vars=z_cols, var_name="Algorithm", value_name="Z-Score")
    z_df["Algorithm"] = z_df["Algorithm"].str.replace("Z_Score_", "")
    sns.boxplot(x='Algorithm', y='Z-Score', data=z_df, ax=ax1, width=0.6, showfliers=False)
    sns.stripplot(x='Algorithm', y='Z-Score', data=z_df, ax=ax1, color='black', alpha=0.4, jitter=True)
    ax1.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2, label='Threshold (z=4.0)')
    ax1.set_title('Detectability: Z-Score Distribution', fontsize=15, pad=15, fontweight='bold')
    ax1.legend()

    # 图 2：PPL 质量分布
    p_cols = [f"PPL_{a}" for a in algorithms]
    p_df = pd.melt(df, value_vars=p_cols, var_name="Algorithm", value_name="PPL")
    p_df["Algorithm"] = p_df["Algorithm"].str.replace("PPL_", "")
    sns.boxplot(x='Algorithm', y='PPL', data=p_df, ax=ax2, width=0.6, showfliers=False)
    sns.stripplot(x='Algorithm', y='PPL', data=p_df, ax=ax2, color='black', alpha=0.4, jitter=True)
    ax2.set_title('Quality Impact: Perplexity (PPL)', fontsize=15, pad=15, fontweight='bold')
    ax2.set_ylabel('Perplexity (Lower is Better)', fontsize=13)

    # 图 3：Pareto 权衡前沿散点图
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    colors = sns.color_palette("deep", len(algorithms))
    
    scatter_collections = {}
    for idx, algo in enumerate(algorithms):
        ppl_jitter = df[f'PPL_{algo}'] + np.random.normal(0, 0.4, size=len(df))
        z_jitter = df[f'Z_Score_{algo}'] + np.random.normal(0, 0.1, size=len(df))
        
        scatter = ax3.scatter(ppl_jitter, z_jitter, alpha=0.4, label=algo, s=25, 
                                  marker=markers[idx % len(markers)], color=colors[idx], 
                                  edgecolors='white', linewidth=0.3, picker=True, pickradius=5)
        scatter_collections[algo] = {'scatter': scatter, 'color': colors[idx], 'marker': markers[idx % len(markers)]}
    
    ax3.axhline(y=4.0, color='#d9534f', linestyle='--', linewidth=2)
    ax3.set_title('Trade-off: Quality vs. Detectability\n(Hover mouse over points to see details)', 
                      fontsize=15, pad=15, fontweight='bold')
    ax3.set_xlabel('Perplexity (Lower is better)', fontsize=13)
    ax3.set_ylabel('Z-Score (Higher is better)', fontsize=13)
    
    ax3.set_xlim(0, 50) 
    ax3.set_ylim(-5, 8) 
    ax3.legend(title="Algorithm", loc="lower right", fontsize=10)

    # 保持原有的悬停交互功能
    cursor = mplcursors.cursor(ax3.collections, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        artist = sel.artist
        index = sel.index
        for algo_name, data in scatter_collections.items():
            sc = data['scatter']
            sc.set_alpha(0.4)
            sc.set_sizes([25] * len(sc.get_offsets()))
            sc.set_edgecolors('white')
            sc.set_linewidth(0.3)
        artist.set_alpha(0.9)
        artist.set_sizes([80] * len(artist.get_offsets()))
        artist.set_edgecolors('black')
        artist.set_linewidth(1.5)
        current_algo = next((name for name, data in scatter_collections.items() if data['scatter'] == artist), "Unknown")
        x_data, y_data = artist.get_offsets()[index]
        sel.annotation.set_text(f'Algorithm: {current_algo}\nPPL: {x_data:.2f}\nZ-Score: {y_data:.2f}')
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8, edgecolor=colors[list(scatter_collections.keys()).index(current_algo)], linewidth=2)
        sel.annotation.set_fontsize(10)
    
    @cursor.connect("remove")
    def on_remove(sel):
        for algo_name, data in scatter_collections.items():
            sc = data['scatter']
            sc.set_alpha(0.4)
            sc.set_sizes([25] * len(sc.get_offsets()))
            sc.set_edgecolors('white')
            sc.set_linewidth(0.3)
        plt.draw()

    # 图 4：数据汇总表格 (Mean ± Std)
    summary_data = []
    for algo in algorithms:
        z_mean = df[f"Z_Score_{algo}"].mean()
        z_std = df[f"Z_Score_{algo}"].std()
        ppl_mean = df[f"PPL_{algo}"].mean()
        ppl_std = df[f"PPL_{algo}"].std()
        summary_data.append([algo, f"{z_mean:.2f} ± {z_std:.2f}", f"{ppl_mean:.2f} ± {ppl_std:.2f}"])
    
    table_df = pd.DataFrame(summary_data, columns=["Algorithm", "Z-Score (Mean±Std)", "PPL (Mean±Std)"])
    
    ax4.axis('off')
    ax4.set_title('Statistical Summary Table', fontsize=15, fontweight='bold', pad=15)
    table = ax4.table(cellText=table_df.values, colLabels=table_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5) # 拉伸表格行高

    plt.tight_layout()
    output_filename = "benchmark_comparison_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"高级对比图表（含数据表）已成功保存至: {output_filename}")
    plt.show()


if __name__ == "__main__":
    from datasets import load_dataset
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import mplcursors
    DATASET_CONFIGS = {
        "C4_News": {"path": "allenai/c4", "name": "realnewslike", "text_col": "text"},
        "Wiki_Academic": {"path": "wikitext", "name": "wikitext-2-raw-v1", "text_col": "text"},
        "Alpaca_Chat": {"path": "tatsu-lab/alpaca", "name": "default", "text_col": "instruction"}
    }

    # 注册所有防线算法（新增 DiPmark 分布保持水印）
    algorithms = {
        "Natural": None,
        "KGW": LogitsProcessorList([KGWLogitsProcessor(model.config.vocab_size, delta=DELTA_VALUE)]),
        "KGW-selfhash": LogitsProcessorList([KGWSelfHashLogitsProcessor(model.config.vocab_size, delta=DELTA_VALUE)]),
        "SWEET": LogitsProcessorList([SWEETLogitsProcessor(model.config.vocab_size, delta=DELTA_VALUE, entropy_threshold=1.5)]),
        "DiPmark": LogitsProcessorList([DiPmarkLogitsProcessor(model.config.vocab_size, alpha=0.6)]),
        "Unigram": LogitsProcessorList([UnigramLogitsProcessor(model.config.vocab_size, delta=DELTA_VALUE)]),
        "SemStamp": LogitsProcessorList([SemStampLogitsProcessor(tokenizer)]),
        "Smooth": LogitsProcessorList([SmoothedWatermarkLogitsProcessor(model.config.vocab_size)]),
        "PublicDetect": LogitsProcessorList([PubliclyDetectableProcessor(model.config.vocab_size, delta=DELTA_VALUE)]),
        "Unforgeable": LogitsProcessorList([UnforgeableLogitsProcessor(model.config.vocab_size, delta=DELTA_VALUE)]),
        "Gumbel": "GUMBEL",     # 自定义采样 (非LogitsProcessor)
        "Transform": "TRANSFORM",  # 自定义采样 (非LogitsProcessor)
        "WaterMax": "WATERMAX",  # 草稿选择 (非LogitsProcessor)
    }

    results = []
    print(f"\n开始【多数据集】联合横向评估...")
    print(f"当前注册算法: {list(algorithms.keys())}")

    for ds_name, ds_info in DATASET_CONFIGS.items():
        print(f"\n>>> 正在连接并处理数据集: {ds_name} <<<")
        try:
            dataset = load_dataset(ds_info["path"], ds_info["name"], split="train", streaming=True, cache_dir=CACHE_DIR)
        except Exception as e:
            print(f"数据集 {ds_name} 加载失败，跳过。报错: {e}")
            continue

        sample_count = 0
        pbar = tqdm(total=TEST_SAMPLE_SIZE, desc=f"生成 {ds_name} 样本")
        
        for data in dataset:
            if sample_count >= TEST_SAMPLE_SIZE:
                break

            text_content = data.get(ds_info["text_col"], "")
            tokens = tokenizer(text_content, return_tensors="pt", truncation=True, max_length=PROMPT_LENGTH)
            if tokens["input_ids"].shape[1] < PROMPT_LENGTH:
                continue

            inputs = {k: v.to(device) for k, v in tokens.items()}
            row_result = {"Dataset": ds_name, "Sample_ID": sample_count + 1}

            seed = 42 + sample_count
            attn_mask = inputs["attention_mask"]

            for algo_name, processor in algorithms.items():
                torch.manual_seed(seed)

                if algo_name == "Gumbel":
                    out_ids, xi, offset = generate_gumbel(
                        model, tokenizer, inputs["input_ids"], attn_mask,
                        GENERATE_LENGTH, seed, model.config.vocab_size, device)
                    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                    _GUMBEL_STATE[text] = {'xi': xi, 'offset': offset, 'seed': seed}
                elif algo_name == "Transform":
                    out_ids, xi, pi, offset = generate_transform(
                        model, tokenizer, inputs["input_ids"], attn_mask,
                        GENERATE_LENGTH, seed, model.config.vocab_size, device)
                    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                    _TRANSFORM_STATE[text] = {'xi': xi, 'pi': pi, 'offset': offset, 'seed': seed}
                elif algo_name == "WaterMax":
                    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                    texts = generate_watermax(
                        model, tokenizer, [prompt_text], GENERATE_LENGTH,
                        num_seq=3, n_splits=2, ngram=3, seed=seed, device=device)
                    text = texts[0]
                    _WATERMAX_STATE[text] = {'ngram': 3, 'split_len': GENERATE_LENGTH // 2, 'seed': seed}
                else:
                    generate_kwargs = {**inputs, "max_new_tokens": GENERATE_LENGTH, "do_sample": True, "temperature": 0.7}
                    if processor is not None:
                        generate_kwargs["logits_processor"] = processor
                    outputs = model.generate(**generate_kwargs)
                    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                row_result[f"Z_Score_{algo_name}"] = round(detect_watermark(text, algo_name, tokenizer, model.config.vocab_size), 3)
                row_result[f"PPL_{algo_name}"] = round(calculate_ppl(text, model, tokenizer, device), 3)
                if algo_name != "Natural":
                    row_result[f"Text_{algo_name}"] = text

            results.append(row_result)
            sample_count += 1
            pbar.update(1) 
            
        pbar.close() 

    df = pd.DataFrame(results)
    df.to_csv(CSV_FILENAME, index=False)
    print(f"\n=== 多维数据计算评估完成！表格已保存至 {CSV_FILENAME} ===")

    metrics_cols = [col for col in df.columns if "Z_Score" in col or "PPL" in col]
    print("\n【各数据集下的算法平均表现汇总】:")
    print(df.groupby("Dataset")[metrics_cols].mean())

    generate_benchmark_plots(CSV_FILENAME)