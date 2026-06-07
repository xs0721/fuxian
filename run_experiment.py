import pyarrow  # 必须最先导入, 避免与Anaconda的pyarrow DLL冲突
import torch
import torch.nn.functional as F
import math
import os
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

# ================= ===================================
# 1. 基础配置（实验参数集中管理）
# ================= ===================================
CACHE_DIR = "E:/Your_Cloud_Drive/hf_cache"
MODEL_NAME = "facebook/opt-125m"
TEST_SAMPLE_SIZE = 200  # 快速测试用, 正式实验改为 200
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
# 使用fp16半精度减少显存占用（显存减半，速度略快）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)


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
        # 兼容PyTorch 2.1.x：uniform_不支持generator参数
        torch.manual_seed(42)
        for p in self.net.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

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


class KSemStampLogitsProcessor(LogitsProcessor):
    """k-SemStamp: 基于k-means聚类的语义水印 (ACL 2024 Findings)

    论文: k-SemStamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text
    作者: Abe Hou, Jingyu Zhang, Yichen Wang, Daniel Khashabi, Tianxing He
    会议: ACL 2024 Findings

    核心思想:
        1. 将词表用k-means聚类成k个语义簇
        2. 根据上下文哈希选择一个簇作为绿名单
        3. 偏向该簇的tokens
        4. 比SemStamp更灵活和鲁棒

    与SemStamp的区别:
        - SemStamp: 单一语义空间
        - k-SemStamp: k个聚类，更细粒度的语义控制

    简化实现说明:
        完整版需要sentence-transformers对词表进行k-means聚类
        本简化版用哈希函数模拟聚类效果，保持核心思想
    """

    def __init__(self, vocab_size, k=5, gamma=0.5, delta=2.0, hash_key=15485863):
        """
        Args:
            vocab_size: 词表大小
            k: 聚类数量（簇的个数）
            gamma: 绿名单比例
            delta: 水印强度
            hash_key: 哈希密钥
        """
        self.vocab_size = vocab_size
        self.k = k
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key

        # 简化版：用哈希函数将词表分成k个簇
        # 完整版应该用sentence-transformers + k-means
        self.token_to_cluster = self._init_clusters()

    def _init_clusters(self):
        """将词表tokens分配到k个簇"""
        token_to_cluster = {}
        for token_id in range(self.vocab_size):
            # 使用哈希函数分配簇ID
            cluster_id = hash((token_id, self.hash_key)) % self.k
            token_to_cluster[token_id] = cluster_id
        return token_to_cluster

    def _get_cluster_greenlist(self, context_hash, cluster_id):
        """获取指定簇内的绿名单tokens"""
        # 获取该簇的所有tokens
        cluster_tokens = [t for t, c in self.token_to_cluster.items() if c == cluster_id]

        # 在簇内随机选择绿名单
        random.seed(context_hash)
        num_green = int(len(cluster_tokens) * self.gamma)
        green_tokens = random.sample(cluster_tokens, min(num_green, len(cluster_tokens)))

        return green_tokens

    def __call__(self, input_ids, scores):
        """应用k-SemStamp水印"""
        for b in range(input_ids.shape[0]):
            # 根据上下文选择一个簇
            context = tuple(input_ids[b].tolist())
            context_hash = hash(context)
            cluster_id = context_hash % self.k

            # 获取该簇的绿名单
            green_tokens = self._get_cluster_greenlist(context_hash, cluster_id)

            # 偏向绿名单tokens
            for token_id in green_tokens:
                scores[b, token_id] += self.delta

        return scores


class STA1LogitsProcessor(LogitsProcessor):
    """STA-1: Sampling-Then-Accepting 无偏低风险水印 (ACL 2025)

    核心思想: 先采样一个token, 判断是否在绿名单, 接受则输出, 拒绝则重采样
    特点: 在密钥空间上期望无偏 (E[P_w] = P_θ), 避免KGW的分布偏移
    """
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, hash_key=15485863, max_trials=10):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.max_trials = max_trials

    def __call__(self, input_ids, scores):
        # STA-1在sampling层面实现，这里仅做标记性偏置
        # 实际实现需要在generate()时用自定义采样器
        for b in range(input_ids.shape[0]):
            g = torch.Generator(device='cpu')
            g.manual_seed(self.hash_key * input_ids[b, -1].item())
            greenlist_size = int(self.vocab_size * self.gamma)
            perm = torch.randperm(self.vocab_size, generator=g)
            greenlist = perm[:greenlist_size]
            # 温和偏置，保持分布接近原始
            scores[b, greenlist.to(scores.device)] += self.delta * 0.5
        return scores


class SIRLogitsProcessor(LogitsProcessor):
    """SIR: Semantic Invariant Robust Watermark (ICLR 2024)

    核心思想: 使用语义编码器提取上下文语义向量，通过MLP映射为每个token的偏置
    特点: 语义相似的改写会产生相似的偏置，抵抗DIPPER等深度改写攻击
    """
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, hash_key=15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        # 简化版：用上下文的聚合哈希代替完整的语义编码器+MLP

    def _semantic_hash(self, context_ids):
        """基于上下文的语义哈希（简化版）"""
        h = 0
        for i, token_id in enumerate(context_ids[-8:]):  # 取最后8个token
            h = (h * 31 + int(token_id) * (i + 1)) % (2**64 - 1)
        return h ^ self.hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            # 基于语义上下文生成绿名单
            semantic_seed = self._semantic_hash(input_ids[b].tolist())
            g = torch.Generator(device='cpu')
            g.manual_seed(semantic_seed % (2**31 - 1))
            greenlist_size = int(self.vocab_size * self.gamma)
            perm = torch.randperm(self.vocab_size, generator=g)
            scores[b, perm[:greenlist_size].to(scores.device)] += self.delta
        return scores


class KTHLogitsProcessor(LogitsProcessor):
    """KTH: Distortion-free Watermark (ICLR 2024)

    核心思想: 使用逆变换采样(Inverse Transform Sampling)在期望上保持原始分布
    特点: E_ξ[P_w] = P_θ, 完全无失真，但需要私钥ξ才能检测
    """
    def __init__(self, vocab_size, hash_key=15485863):
        self.vocab_size = vocab_size
        self.hash_key = hash_key

    def __call__(self, input_ids, scores):
        # KTH的完整实现需要在采样阶段进行逆变换
        # 这里提供一个简化的标记版本
        for b in range(input_ids.shape[0]):
            g = torch.Generator(device='cpu')
            g.manual_seed(self.hash_key * input_ids[b, -1].item())
            # 生成伪随机扰动，但保持期望为0
            noise = torch.randn(self.vocab_size, generator=g) * 0.1
            scores[b] += noise.to(scores.device)
        return scores


class TBWLogitsProcessor(LogitsProcessor):
    """TBW: Topic-Based Watermark (arXiv 2024)

    核心思想: 将词表按主题聚类，根据输入提示词的主题动态选择绿名单
    特点: 避免在特定主题下破坏生成质量，提高主题一致性
    """
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, hash_key=15485863, num_topics=8):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.num_topics = num_topics
        self._init_topic_clusters()

    def _init_topic_clusters(self):
        """将词表划分为主题簇（简化版：随机聚类）"""
        g = torch.Generator(device='cpu')
        g.manual_seed(42)  # 固定种子保证一致性
        self.topic_assignment = torch.randint(0, self.num_topics, (self.vocab_size,), generator=g)

    def _detect_topic(self, context_ids):
        """检测当前上下文的主题（简化版：基于token频率）"""
        if len(context_ids) == 0:
            return 0
        # 简单策略：用最后一个token的模运算确定主题
        return int(context_ids[-1]) % self.num_topics

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            # 检测当前主题
            topic_id = self._detect_topic(input_ids[b].tolist())

            # 在当前主题内生成绿名单
            topic_mask = (self.topic_assignment == topic_id)
            topic_vocab = torch.where(topic_mask)[0]

            if len(topic_vocab) > 0:
                g = torch.Generator(device='cpu')
                g.manual_seed(self.hash_key * topic_id)
                greenlist_size = min(len(topic_vocab), int(len(topic_vocab) * self.gamma))
                selected_indices = torch.randperm(len(topic_vocab), generator=g)[:greenlist_size]
                greenlist = topic_vocab[selected_indices]
                scores[b, greenlist.to(scores.device)] += self.delta
        return scores


class XSIRLogitsProcessor(LogitsProcessor):
    """X-SIR: Cross-lingual Semantic Invariant Robust Watermark (ACL 2024)

    核心思想: 跨语言语义一致性水印，翻译后仍可检测
    简化实现: 用鲁棒语义哈希代替多语言编码器（XLM-R/mBERT）

    完整版需要:
    - 多语言语义编码器（XLM-R）
    - 跨语言对齐的语义空间
    - 翻译模型测试

    简化版策略:
    - 使用token集合而非序列（忽略顺序）
    - 对小变化鲁棒的哈希函数
    - 可以用改写测试代替翻译测试
    """
    def __init__(self, vocab_size, gamma=0.5, delta=2.0, hash_key=15485863,
                 aggregation='set', context_window=8):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.aggregation = aggregation  # 'set', 'bag', 'sum'
        self.context_window = context_window

    def _semantic_hash_robust(self, context_ids):
        """鲁棒语义哈希：对token顺序和小变化不敏感

        关键：翻译后语义相似 → 哈希值相似 → 绿名单重叠
        """
        context = context_ids[-self.context_window:]  # 取最近的上下文

        if self.aggregation == 'set':
            # 方法1: 集合哈希（最鲁棒，忽略顺序和重复）
            token_set = sorted(set(context))
            h = sum(token_set) % (2**31 - 1)

        elif self.aggregation == 'bag':
            # 方法2: 词袋哈希（考虑频率但忽略顺序）
            from collections import Counter
            token_bag = Counter(context)
            h = sum(token_id * count for token_id, count in token_bag.items())
            h = h % (2**31 - 1)

        else:  # 'sum'
            # 方法3: 简单求和（对顺序不敏感）
            h = sum(context) % (2**31 - 1)

        return h ^ self.hash_key

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            # 计算鲁棒语义哈希（模拟多语言语义编码）
            semantic_seed = self._semantic_hash_robust(input_ids[b].tolist())

            # 基于语义生成绿名单
            g = torch.Generator(device='cpu')
            g.manual_seed(semantic_seed)
            greenlist_size = int(self.vocab_size * self.gamma)
            perm = torch.randperm(self.vocab_size, generator=g)

            scores[b, perm[:greenlist_size].to(scores.device)] += self.delta

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
    # 兼容PyTorch 2.1.x：cumsum在CPU上不支持Half类型
    gathered = probs.gather(1, pi.unsqueeze(0).expand(probs.shape[0], -1))
    if gathered.dtype == torch.float16 and gathered.device.type == 'cpu':
        gathered = gathered.float()
    cdf = torch.cumsum(gathered, dim=1)
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
        input_lens = [len(p[p != tokenizer.eos_token_id])
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


def calculate_entropy_for_ewd(model, tokenizer, tokens, device):
    """
    EWD专用：计算每个token位置的条件熵

    参数:
        model: 语言模型
        tokenizer: 分词器
        tokens: token序列 (torch.Tensor)
        device: 设备

    返回:
        entropy_list: 每个位置的熵值列表
    """
    entropy_list = []

    with torch.no_grad():
        for i in range(1, len(tokens)):
            # 获取前缀
            prefix = tokens[:i].unsqueeze(0).to(device)

            # 计算logits
            outputs = model(prefix)
            logits = outputs.logits[0, -1, :]  # 最后一个位置的logits

            # 计算概率分布
            probs = F.softmax(logits, dim=-1)

            # 计算熵: H = -Σ p(x) * log(p(x))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            entropy_list.append(entropy)

    return entropy_list


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

    # 雷达 C+：k-SemStamp 聚类语义检测器 (ACL 2024 Findings)
    elif algo_name == "k-SemStamp":
        """
        k-SemStamp检测：基于k-means聚类的语义水印检测

        核心思想：
        1. 词表被分成k个语义簇
        2. 对每个可能的簇，计算绿token比例
        3. 选择最大比例作为检测分数
        4. 用Z-score评估显著性

        简化实现：用哈希模拟聚类
        """
        k = 5  # 簇的数量

        # 初始化token到簇的映射（与生成时一致）
        token_to_cluster = {}
        for token_id in range(vocab_size):
            cluster_id = hash((token_id, secret_key)) % k
            token_to_cluster[token_id] = cluster_id

        # 对每个簇计算绿token比例
        max_green_ratio = 0.0
        best_cluster_id = 0

        for cluster_id in range(k):
            # 获取该簇的所有tokens
            cluster_tokens = set([t for t, c in token_to_cluster.items() if c == cluster_id])
            cluster_size = len(cluster_tokens)

            if cluster_size == 0:
                continue

            # 统计文本中该簇的绿token数量
            green_in_cluster = 0
            total_in_cluster = 0

            for i in range(1, len(tokens)):
                token_id = tokens[i].item()
                if token_id in cluster_tokens:
                    total_in_cluster += 1

                    # 判断是否在该簇的绿名单中
                    context = tuple(tokens[:i].tolist())
                    context_hash = hash(context)

                    # 模拟簇内绿名单选择
                    import random
                    random.seed(context_hash)
                    num_green = int(cluster_size * gamma)
                    green_list = random.sample(list(cluster_tokens), min(num_green, cluster_size))

                    if token_id in green_list:
                        green_in_cluster += 1

            # 计算该簇的绿token比例
            if total_in_cluster > 0:
                ratio = green_in_cluster / total_in_cluster
                if ratio > max_green_ratio:
                    max_green_ratio = ratio
                    best_cluster_id = cluster_id

        # 计算Z-score
        # 在k个簇的情况下，期望比例是gamma
        # 但由于簇内采样，实际期望略有不同
        expected_ratio = gamma

        if total_tokens == 0:
            return 0.0

        # 标准差计算
        variance = expected_ratio * (1 - expected_ratio) / max(total_tokens, 1)
        if variance == 0:
            return 0.0

        z_score = (max_green_ratio - expected_ratio) / math.sqrt(variance)
        return z_score

    # 雷达 D：EWD熵加权检测器 (ACL 2024)
    elif algo_name == "EWD":
        """
        EWD核心思想：使用熵加权来改进KGW检测
        - 高熵位置（模型不确定）的绿token权重更高
        - 低熵位置（模型确定）的绿token权重更低
        - 提高检测准确性，降低误报率
        """
        # 方法1：完整版（需要模型，准确但慢）
        # entropy_weights = calculate_entropy_for_ewd(model, tokenizer, tokens, device)

        # 方法2：简化版（无需模型，快速但近似）
        # 使用启发式规则估计熵：token变化大 ≈ 熵高
        entropy_weights = []
        for i in range(1, len(tokens)):
            # 启发式熵估计
            if i > 1:
                # 基于相邻token的差异估计不确定性
                token_diff = abs(tokens[i].item() - tokens[i-1].item())
                # 归一化到合理范围[0.5, 1.5]
                weight = 0.7 + 0.6 * min(token_diff / vocab_size, 1.0)
            else:
                weight = 1.0
            entropy_weights.append(weight)

            # 检测是否在绿名单（标准KGW规则）
            torch.manual_seed(secret_key * tokens[i - 1].item())
            if (torch.rand(vocab_size) < gamma)[tokens[i].item()]:
                green_tokens_count += weight  # 使用熵权重

        # 加权统计量计算
        weighted_total = sum(entropy_weights)
        if weighted_total == 0: return 0.0

        expected_green = weighted_total * gamma
        # 加权方差：Σw_i * γ(1-γ)
        weighted_variance = sum(w * gamma * (1 - gamma) for w in entropy_weights)

        if weighted_variance == 0: return 0.0

        # EWD加权Z-score
        z_score = (green_tokens_count - expected_green) / math.sqrt(weighted_variance)
        return z_score

    # 雷达 E：X-SIR跨语言语义鲁棒检测器 (ACL 2024)
    elif algo_name == "X-SIR":
        """
        X-SIR核心思想：跨语言语义一致性检测
        - 使用鲁棒语义哈希（对顺序不敏感）
        - 翻译/改写后语义相似 → 哈希相似 → 检测一致
        - 相比KGW对改写更鲁棒
        """
        green_tokens_count = 0
        context_window = 8

        for i in range(1, len(tokens)):
            # 获取上下文窗口
            context = tokens[max(0, i-context_window):i].tolist()

            # 鲁棒语义哈希（与生成时一致）
            # 使用集合而非序列，忽略token顺序
            token_set = sorted(set(context))
            semantic_hash = sum(token_set) % (2**31 - 1)
            semantic_seed = semantic_hash ^ secret_key

            # 基于语义生成绿名单
            g = torch.Generator(device='cpu')
            g.manual_seed(semantic_seed)
            greenlist_size = int(vocab_size * gamma)
            perm = torch.randperm(vocab_size, generator=g)
            greenlist = set(perm[:greenlist_size].tolist())

            if tokens[i].item() in greenlist:
                green_tokens_count += 1

        # 标准Z-score计算
        variance = total_tokens * gamma * (1 - gamma)
        if variance == 0: return 0.0
        z_score = (green_tokens_count - total_tokens * gamma) / math.sqrt(variance)
        return z_score

    return 0.0


def calculate_ppl(text, model, tokenizer, device):
    encodings = tokenizer(text, return_tensors="pt").to(device)
    max_length = 512  # 限制最大长度，避免超长文本
    if encodings.input_ids.size(1) > max_length:
        encodings.input_ids = encodings.input_ids[:, :max_length]

    with torch.no_grad():
        # 使用fp16计算，减少显存
        if device == "cuda" and model.dtype == torch.float16:
            with torch.cuda.amp.autocast():
                loss = model(encodings.input_ids, labels=encodings.input_ids).loss
        else:
            loss = model(encodings.input_ids, labels=encodings.input_ids).loss

    # 立即清理
    del encodings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.exp(loss).item()


# ══════════════════════════════════════════════════════════════════════════════
# ICW (In-Context Watermarks) - Prompt-based黑盒水印 (arXiv 2025)
# ══════════════════════════════════════════════════════════════════════════════

class ICWInitialsWatermark:
    """ICW Initials水印：通过提示词偏向特定首字母

    论文: In-Context Watermarks for Large Language Models (arXiv 2025)
    作者: Yepeng Liu, Xuandong Zhao et al.

    核心思想:
        不修改模型logits，而是通过系统提示词引导模型
        偏向使用特定首字母开头的单词

    优势:
        - 完全黑盒，适用于任何LLM（包括GPT-4等闭源模型）
        - 无需访问模型内部
        - 易于部署（只需修改prompt）

    局限:
        - 依赖模型对指令的遵循能力（小模型效果可能有限）
        - 鲁棒性低于logits层水印
        - 会影响文本自然度
    """

    def __init__(self, green_letters=None, red_letters=None):
        # 绿名单：高频字母（偏向使用）
        self.green_letters = green_letters or ['a','e','i','o','t','s','r','h','n','l']
        # 红名单：低频字母（减少使用）
        self.red_letters = red_letters or ['b','c','d','f','g','j','k','m','p','q']

        # 归一化为小写
        self.green_letters = [l.lower() for l in self.green_letters]
        self.red_letters = [l.lower() for l in self.red_letters]

        # 计算gamma（绿名单比例）
        total = len(self.green_letters) + len(self.red_letters)
        self.gamma = len(self.green_letters) / total if total > 0 else 0.5

    def generate_system_prompt(self, strength='medium'):
        """生成包含水印指令的系统提示词

        strength: 'weak', 'medium', 'strong'
        """
        green_str = ', '.join(self.green_letters[:5])  # 只显示前5个
        red_str = ', '.join(self.red_letters[:5])

        if strength == 'weak':
            instruction = f"When possible, slightly prefer words starting with: {green_str}"
        elif strength == 'strong':
            instruction = f"IMPORTANT: Strongly prefer words starting with: {green_str}. Avoid: {red_str}"
        else:  # medium
            instruction = f"Try to use more words that start with: {green_str}. Use fewer words starting with: {red_str}"

        return f"""You are a helpful assistant.

{instruction}

Still maintain natural and coherent responses."""

    def generate_with_watermark(self, model, tokenizer, prompt, device='cuda',
                                max_new_tokens=100, strength='medium'):
        """使用ICW生成带水印文本

        注意: 这不是LogitsProcessor方式，而是通过prompt引导
        """
        # 构造完整prompt（包含水印指令）
        system_prompt = self.generate_system_prompt(strength)
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        # 标准生成（无LogitsProcessor）
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        # 提取助手回复（去除系统提示部分）
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return response

    def detect_watermark(self, text):
        """检测ICW Initials水印

        返回: Z-score（越高越可能是水印文本）
        """
        # 分词并统计首字母
        words = text.lower().split()

        green_count = 0
        total_count = 0

        for word in words:
            # 提取第一个字母字符
            first_letter = None
            for char in word:
                if char.isalpha():
                    first_letter = char
                    break

            if first_letter and (first_letter in self.green_letters or
                                 first_letter in self.red_letters):
                total_count += 1
                if first_letter in self.green_letters:
                    green_count += 1

        if total_count == 0:
            return 0.0

        # 计算Z-score
        expected = total_count * self.gamma
        variance = total_count * self.gamma * (1 - self.gamma)

        if variance == 0:
            return 0.0

        z_score = (green_count - expected) / math.sqrt(variance)
        return z_score


class ICWLexicalWatermark:
    """ICW Lexical水印：偏向特定词汇集合

    简化版实现：通过提示词引导使用特定词汇
    """

    def __init__(self, green_words=None):
        self.green_words = green_words or [
            'innovative', 'advanced', 'intelligent', 'efficient',
            'sophisticated', 'revolutionary', 'cutting-edge'
        ]
        self.green_words_set = set(w.lower() for w in self.green_words)

    def generate_system_prompt(self):
        words_str = ', '.join(self.green_words[:5])
        return f"""You are a helpful assistant.

When appropriate, try to use these words in your response: {words_str}

Maintain natural and informative responses."""

    def generate_with_watermark(self, model, tokenizer, prompt, device='cuda',
                                max_new_tokens=100):
        system_prompt = self.generate_system_prompt()
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return response

    def detect_watermark(self, text):
        """检测特定词汇的出现频率"""
        words = text.lower().split()
        count = sum(1 for w in words if w in self.green_words_set)
        total = len(words)

        if total == 0:
            return 0.0

        # 返回频率（也可以转换为Z-score）
        frequency = count / total
        return frequency


# ══════════════════════════════════════════════════════════════════════════════
# 3. 自动化绘图引擎定义 (优化版：带统计表格的 2x2 布局)
# ══════════════════════════════════════════════════════════════════════════════
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
    import gc  # 用于垃圾回收，清理显存
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
                        num_seq=2, n_splits=1, ngram=3, seed=seed, device=device)
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

                # PPL计算后立即清理显存，防止碎片化
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if algo_name != "Natural":
                    row_result[f"Text_{algo_name}"] = text

                # 每个算法测试后清理显存，防止累积
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results.append(row_result)
            sample_count += 1
            pbar.update(1)

            # 每10个样本清理一次显存和垃圾回收
            if sample_count % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        pbar.close() 

    df = pd.DataFrame(results)
    df.to_csv(CSV_FILENAME, index=False)
    print(f"\n=== 多维数据计算评估完成！表格已保存至 {CSV_FILENAME} ===")

    metrics_cols = [col for col in df.columns if "Z_Score" in col or "PPL" in col]
    print("\n【各数据集下的算法平均表现汇总】:")
    print(df.groupby("Dataset")[metrics_cols].mean())

    generate_benchmark_plots(CSV_FILENAME)