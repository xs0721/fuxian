"""More faithful watermark implementations and upgrade scaffolding.

This module separates official-style sampling/detection logic from
``run_experiment.py`` so importing helpers does not load a language model.
STA-1 and KTH are implemented as custom samplers because a HuggingFace
``LogitsProcessor`` cannot express rejection sampling or inverse-transform
sampling by itself.

For SIR, X-SIR and k-SemStamp, the official papers require external semantic
encoders, trained watermark networks, or saved k-means centroids. The classes
below expose those hooks and provide deterministic local fallbacks for quick
verification. Use the asset-backed paths for paper-level experiments.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - optional dependency
    KMeans = None


_MOD31 = 2**31 - 1
_MOD53 = 2**53


def stable_hash_int(*parts: object, modulo: int = _MOD31) -> int:
    """Return a deterministic positive integer hash for arbitrary values."""
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "big") % modulo


def _seeded_generator(*parts: object) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(stable_hash_int(*parts))
    return gen


def _normal_p_to_z(pvalue: float) -> float:
    pvalue = max(min(float(pvalue), 1.0 - 1e-15), 1e-300)
    return float(-NormalDist().inv_cdf(pvalue))


def _z_from_count(count: float, total: int, gamma: float) -> float:
    if total <= 0:
        return 0.0
    variance = total * gamma * (1.0 - gamma)
    if variance <= 0:
        return 0.0
    return float((count - total * gamma) / math.sqrt(variance))


def greenlist_for_context(
    vocab_size: int,
    context_token: int,
    gamma: float = 0.5,
    key: int = 15485863,
) -> torch.Tensor:
    """KGW/STA-style fixed-size greenlist seeded by the previous token."""
    gen = _seeded_generator("greenlist", key, int(context_token))
    size = max(1, int(vocab_size * gamma))
    return torch.randperm(vocab_size, generator=gen)[:size]


def is_green_token(
    token_id: int,
    context_token: int,
    vocab_size: int,
    gamma: float = 0.5,
    key: int = 15485863,
) -> bool:
    green = greenlist_for_context(vocab_size, context_token, gamma, key)
    return bool((green == int(token_id)).any().item())


# ---------------------------------------------------------------------------
# STA-1: Sampling One Then Accepting
# ---------------------------------------------------------------------------


class STA1LogitsProcessor(LogitsProcessor):
    """Compatibility shim for STA-1.

    STA-1 is a custom sampling algorithm, not a logits-bias method. This
    processor intentionally leaves logits unchanged. Use ``generate_sta1`` for
    the official mechanism.
    """

    official_sampling_required = True

    def __init__(self, vocab_size: int, gamma: float = 0.5, hash_key: int = 15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.hash_key = hash_key

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return scores


@torch.no_grad()
def generate_sta1(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int,
    seed: int,
    vocab_size: int,
    gamma: float = 0.5,
    hash_key: int = 15485863,
    device: str = "cuda",
    sta_m: int = 1,
    entropy_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Generate text with STA-1 or STA-M.

    STA-1 samples one candidate from the original distribution. If it is green,
    it is accepted; otherwise a second sample from the original distribution is
    accepted unconditionally. STA-M repeats green-seeking only when an entropy
    threshold is supplied and the step is high entropy.
    """
    input_ids = prompt_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    sampler = _seeded_generator("sta", seed)
    past = None

    for step in range(max_new_tokens):
        if past is not None:
            output = model(input_ids[:, -1:], past_key_values=past, attention_mask=attention_mask)
        else:
            output = model(input_ids, attention_mask=attention_mask)

        logits = output.logits[:, -1, :].float().detach().cpu()
        probs = torch.softmax(logits, dim=-1)
        next_tokens: List[int] = []

        for batch_idx in range(input_ids.shape[0]):
            prev_token = int(input_ids[batch_idx, -1].item())
            row_probs = probs[batch_idx]
            entropy = float(-(row_probs * (row_probs + 1e-12).log()).sum().item())
            max_trials = 1
            if entropy_threshold is not None and entropy >= entropy_threshold:
                max_trials = max(1, int(sta_m))

            accepted: Optional[int] = None
            for trial in range(max_trials):
                cand = int(torch.multinomial(row_probs, 1, generator=sampler).item())
                if is_green_token(cand, prev_token, vocab_size, gamma, hash_key):
                    accepted = cand
                    break
                if trial == 0 and max_trials == 1:
                    accepted = int(torch.multinomial(row_probs, 1, generator=sampler).item())
                    break

            if accepted is None:
                accepted = int(torch.multinomial(row_probs, 1, generator=sampler).item())
            next_tokens.append(accepted)

        tok = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(-1)
        input_ids = torch.cat([input_ids, tok], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )
        past = output.past_key_values

    return input_ids


def detect_sta1_tokens(
    token_ids: Sequence[int],
    vocab_size: int,
    gamma: float = 0.5,
    hash_key: int = 15485863,
) -> float:
    """STA-1 uses the same black-box green-token z-test as KGW."""
    if len(token_ids) < 2:
        return 0.0
    green_count = 0
    for i in range(1, len(token_ids)):
        if is_green_token(token_ids[i], token_ids[i - 1], vocab_size, gamma, hash_key):
            green_count += 1
    return _z_from_count(green_count, len(token_ids) - 1, gamma)


# ---------------------------------------------------------------------------
# KTH/RDW: robust distortion-free inverse-transform sampling
# ---------------------------------------------------------------------------


@dataclass
class KTHDetectionState:
    vocab_size: int
    key_length: int = 256
    seed: int = 0
    offset: int = 0
    block_size: int = 35
    independent_permutations: bool = False


class KTHKey:
    """Lazy KTH key sequence of uniforms and permutations."""

    def __init__(
        self,
        state: KTHDetectionState,
    ):
        self.state = state
        self._perm_cache: Dict[int, torch.Tensor] = {}
        self._inv_cache: Dict[int, torch.Tensor] = {}

    def uniform(self, pos: int) -> float:
        raw = stable_hash_int("kth-u", self.state.seed, int(pos), modulo=_MOD53)
        return (raw + 0.5) / _MOD53

    def permutation(self, pos: int) -> torch.Tensor:
        cache_pos = int(pos) if self.state.independent_permutations else -1
        if cache_pos not in self._perm_cache:
            gen = _seeded_generator("kth-perm", self.state.seed, cache_pos)
            self._perm_cache[cache_pos] = torch.randperm(self.state.vocab_size, generator=gen)
        return self._perm_cache[cache_pos]

    def inverse_permutation(self, pos: int) -> torch.Tensor:
        cache_pos = int(pos) if self.state.independent_permutations else -1
        if cache_pos not in self._inv_cache:
            perm = self.permutation(pos)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(len(perm), dtype=perm.dtype)
            self._inv_cache[cache_pos] = inv
        return self._inv_cache[cache_pos]


class KTHLogitsProcessor(LogitsProcessor):
    """Compatibility shim for KTH/RDW.

    KTH is a distortion-free decoder. Use ``generate_kth_inverse`` rather than
    a logits processor to obtain the official inverse-transform mechanism.
    """

    official_sampling_required = True

    def __init__(self, vocab_size: int, hash_key: int = 15485863):
        self.vocab_size = vocab_size
        self.hash_key = hash_key

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return scores


def _inverse_transform_token(probs: torch.Tensor, key: KTHKey, pos: int) -> int:
    perm = key.permutation(pos)
    ordered_probs = probs[perm]
    cdf = torch.cumsum(ordered_probs, dim=-1)
    u = torch.tensor(key.uniform(pos), dtype=cdf.dtype)
    idx = int(torch.searchsorted(cdf, u).clamp(max=len(perm) - 1).item())
    return int(perm[idx].item())


@torch.no_grad()
def generate_kth_inverse(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int,
    seed: int,
    vocab_size: int,
    key_length: int = 256,
    block_size: int = 35,
    independent_permutations: bool = False,
    device: str = "cuda",
) -> Tuple[torch.Tensor, KTHDetectionState]:
    """Generate KTH/RDW watermarked text with inverse-transform sampling."""
    offset = stable_hash_int("kth-offset", seed, modulo=key_length)
    state = KTHDetectionState(
        vocab_size=vocab_size,
        key_length=key_length,
        seed=seed,
        offset=offset,
        block_size=block_size,
        independent_permutations=independent_permutations,
    )
    key = KTHKey(state)

    input_ids = prompt_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    past = None
    for step in range(max_new_tokens):
        if past is not None:
            output = model(input_ids[:, -1:], past_key_values=past, attention_mask=attention_mask)
        else:
            output = model(input_ids, attention_mask=attention_mask)

        logits = output.logits[:, -1, :].float().detach().cpu()
        probs = torch.softmax(logits, dim=-1)
        next_tokens = []
        for batch_idx in range(input_ids.shape[0]):
            pos = (offset + step) % key_length
            next_tokens.append(_inverse_transform_token(probs[batch_idx], key, pos))

        tok = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(-1)
        input_ids = torch.cat([input_ids, tok], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )
        past = output.past_key_values

    return input_ids, state


def _kth_block_cost(tokens: Sequence[int], key: KTHKey, start_pos: int) -> float:
    vocab_scale = max(key.state.vocab_size - 1, 1)
    cost = 0.0
    for local_idx, tok in enumerate(tokens):
        pos = (start_pos + local_idx) % key.state.key_length
        inv = key.inverse_permutation(pos)
        rank = float(inv[int(tok)].item()) / vocab_scale
        u = key.uniform(pos)
        cost += -((u - 0.5) * (rank - 0.5))
    return float(cost)


def kth_alignment_statistic(
    token_ids: Sequence[int],
    state: KTHDetectionState,
    max_windows: Optional[int] = None,
    max_offsets: Optional[int] = None,
) -> float:
    """Minimum alignment cost over text windows and key offsets."""
    if not token_ids:
        return 0.0
    key = KTHKey(state)
    k = min(max(1, state.block_size), len(token_ids))
    windows = range(0, len(token_ids) - k + 1)
    offsets = range(state.key_length)

    if max_windows is not None:
        windows = range(0, min(len(token_ids) - k + 1, max_windows))
    if max_offsets is not None:
        offsets = range(0, min(state.key_length, max_offsets))

    best = float("inf")
    for win_start in windows:
        block = token_ids[win_start : win_start + k]
        for offset in offsets:
            best = min(best, _kth_block_cost(block, key, offset))
    return best


def detect_kth_tokens(
    token_ids: Sequence[int],
    state: KTHDetectionState,
    n_runs: int = 100,
    max_windows: Optional[int] = 32,
    max_offsets: Optional[int] = 128,
) -> float:
    """Permutation-test KTH detector returning a z-score-like value."""
    if len(token_ids) < 4:
        return 0.0
    observed = kth_alignment_statistic(token_ids, state, max_windows, max_offsets)
    null_leq = 0
    for run in range(n_runs):
        null_state = KTHDetectionState(
            vocab_size=state.vocab_size,
            key_length=state.key_length,
            seed=stable_hash_int("kth-null", state.seed, run),
            offset=0,
            block_size=state.block_size,
            independent_permutations=state.independent_permutations,
        )
        null_score = kth_alignment_statistic(token_ids, null_state, max_windows, max_offsets)
        if null_score <= observed:
            null_leq += 1
    pvalue = (1 + null_leq) / (n_runs + 1)
    return _normal_p_to_z(pvalue)


# ---------------------------------------------------------------------------
# Shared semantic fallback assets for SIR / X-SIR / k-SemStamp
# ---------------------------------------------------------------------------


class HashingTextEncoder:
    """Deterministic local text/id encoder used when official encoders are absent."""

    def __init__(self, dim: int = 256, seed: int = 17):
        self.dim = dim
        self.seed = seed

    def encode_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        vec = torch.zeros(self.dim, dtype=torch.float32)
        for pos, token_id in enumerate(token_ids):
            idx = stable_hash_int("id", self.seed, int(token_id), pos % 11, modulo=self.dim)
            sign = 1.0 if stable_hash_int("sgn", self.seed, int(token_id), modulo=2) == 0 else -1.0
            vec[idx] += sign
        return F.normalize(vec, dim=0) if float(vec.norm()) > 0 else vec

    def encode_text(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self.dim, dtype=torch.float32)
        terms = re.findall(r"[\w]+", text.lower())
        if not terms:
            terms = [text.lower()]
        for term in terms:
            for n in (2, 3, 4):
                grams = [term[i : i + n] for i in range(max(1, len(term) - n + 1))]
                for gram in grams:
                    idx = stable_hash_int("txt", self.seed, gram, modulo=self.dim)
                    sign = 1.0 if stable_hash_int("txt-sgn", self.seed, gram, modulo=2) == 0 else -1.0
                    vec[idx] += sign
        return F.normalize(vec, dim=0) if float(vec.norm()) > 0 else vec


class LowRankWatermarkProjector:
    """Small semantic-embedding-to-vocabulary watermark model.

    Official SIR uses a trained watermark model. This projector can load saved
    tensors or provide deterministic fallback weights for local smoke tests.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        rank: int = 64,
        seed: int = 15485863,
        state_path: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.rank = rank
        self.seed = seed
        gen = _seeded_generator("sir-projector", seed)
        self.in_proj = torch.randn(embedding_dim, rank, generator=gen) / math.sqrt(max(embedding_dim, 1))
        self.out_proj = torch.randn(rank, vocab_size, generator=gen) / math.sqrt(max(rank, 1))
        if state_path:
            state = torch.load(state_path, map_location="cpu")
            self.in_proj = state["in_proj"].float()
            self.out_proj = state["out_proj"].float()

    def logits(self, embedding: torch.Tensor, device: torch.device) -> torch.Tensor:
        hidden = torch.tanh(embedding.float().cpu() @ self.in_proj)
        logits = hidden @ self.out_proj
        logits = logits - logits.mean()
        logits = logits / (logits.std() + 1e-6)
        return logits.to(device)


class SIRLogitsProcessor(LogitsProcessor):
    """SIR-style semantic watermark logits.

    For official experiments, pass ``projector_state_path`` from a trained SIR
    watermark model and replace ``encoder`` with a BERT/C-BERT adapter.
    """

    def __init__(
        self,
        vocab_size: int,
        delta: float = 2.0,
        hash_key: int = 15485863,
        encoder: Optional[HashingTextEncoder] = None,
        projector_state_path: Optional[str] = None,
        embedding_dim: int = 256,
        rank: int = 64,
        **_: object,
    ):
        self.vocab_size = vocab_size
        self.delta = delta
        self.hash_key = hash_key
        self.encoder = encoder or HashingTextEncoder(dim=embedding_dim, seed=hash_key)
        self.projector = LowRankWatermarkProjector(
            embedding_dim=self.encoder.dim,
            vocab_size=vocab_size,
            rank=rank,
            seed=hash_key,
            state_path=projector_state_path,
        )

    def watermark_logits(self, context_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        embedding = self.encoder.encode_ids(context_ids)
        return self.projector.logits(embedding, device)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for batch_idx in range(input_ids.shape[0]):
            wm_logits = self.watermark_logits(input_ids[batch_idx].tolist(), scores.device)
            scores[batch_idx] += self.delta * wm_logits
        return scores


def detect_sir_tokens(
    token_ids: Sequence[int],
    vocab_size: int,
    delta: float = 2.0,
    hash_key: int = 15485863,
) -> float:
    if len(token_ids) < 2:
        return 0.0
    proc = SIRLogitsProcessor(vocab_size=vocab_size, delta=delta, hash_key=hash_key)
    values = []
    for i in range(1, len(token_ids)):
        logits = proc.watermark_logits(token_ids[:i], torch.device("cpu"))
        values.append(float(logits[int(token_ids[i])].item()))
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or arr.std() == 0:
        return 0.0
    return float(arr.mean() * math.sqrt(arr.size) / (arr.std() + 1e-12))


class XSIRLogitsProcessor(LogitsProcessor):
    """X-SIR-style cross-lingual semantic cluster watermark.

    Official X-SIR uses multilingual semantic clustering assets. This class can
    accept a precomputed ``token_cluster_ids`` tensor. Without it, it falls back
    to deterministic token-id clusters for smoke tests.
    """

    def __init__(
        self,
        vocab_size: int,
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 15485863,
        num_clusters: int = 256,
        token_cluster_ids: Optional[torch.Tensor] = None,
        encoder: Optional[HashingTextEncoder] = None,
        **_: object,
    ):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.num_clusters = num_clusters
        self.encoder = encoder or HashingTextEncoder(dim=256, seed=hash_key)
        if token_cluster_ids is None:
            self.token_cluster_ids = torch.tensor(
                [stable_hash_int("xsir-cluster", hash_key, i, modulo=num_clusters) for i in range(vocab_size)],
                dtype=torch.long,
            )
        else:
            self.token_cluster_ids = token_cluster_ids.long().cpu()

    def _context_seed(self, context_ids: Sequence[int]) -> int:
        emb = self.encoder.encode_ids(context_ids)
        top = torch.topk(emb.abs(), k=min(16, len(emb))).indices.tolist()
        signs = [1 if emb[i] >= 0 else 0 for i in top]
        return stable_hash_int("xsir-context", self.hash_key, tuple(top), tuple(signs))

    def valid_clusters(self, context_ids: Sequence[int]) -> torch.Tensor:
        gen = _seeded_generator("xsir-valid", self._context_seed(context_ids))
        count = max(1, int(self.num_clusters * self.gamma))
        return torch.randperm(self.num_clusters, generator=gen)[:count]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        clusters = self.token_cluster_ids.to(scores.device)
        for batch_idx in range(input_ids.shape[0]):
            valid = self.valid_clusters(input_ids[batch_idx].tolist()).to(scores.device)
            mask = torch.isin(clusters, valid)
            scores[batch_idx, mask] += self.delta
        return scores


def detect_xsir_tokens(
    token_ids: Sequence[int],
    vocab_size: int,
    gamma: float = 0.5,
    hash_key: int = 15485863,
    num_clusters: int = 256,
) -> float:
    if len(token_ids) < 2:
        return 0.0
    proc = XSIRLogitsProcessor(vocab_size, gamma=gamma, hash_key=hash_key, num_clusters=num_clusters)
    green_count = 0
    for i in range(1, len(token_ids)):
        valid = set(proc.valid_clusters(token_ids[:i]).tolist())
        token_cluster = int(proc.token_cluster_ids[int(token_ids[i])].item())
        if token_cluster in valid:
            green_count += 1
    return _z_from_count(green_count, len(token_ids) - 1, gamma)


# ---------------------------------------------------------------------------
# k-SemStamp: sentence-level k-means semantic watermark
# ---------------------------------------------------------------------------


def sentence_tokenize(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class KSemStampWatermark:
    """Sentence-level k-SemStamp generation/detection helper."""

    def __init__(
        self,
        k: int = 64,
        gamma: float = 0.5,
        margin: float = 0.035,
        prime: int = 15485863,
        encoder: Optional[HashingTextEncoder] = None,
        centroids: Optional[np.ndarray] = None,
    ):
        self.k = k
        self.gamma = gamma
        self.margin = margin
        self.prime = prime
        self.encoder = encoder or HashingTextEncoder(dim=256, seed=prime)
        self.centroids = None if centroids is None else self._normalize_np(centroids)

    @staticmethod
    def _normalize_np(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norm

    def fit(self, domain_sentences: Sequence[str], random_state: int = 0) -> "KSemStampWatermark":
        embeddings = torch.stack([self.encoder.encode_text(s) for s in domain_sentences]).numpy()
        if KMeans is None:
            raise RuntimeError("sklearn is required to fit k-SemStamp centroids")
        km = KMeans(n_clusters=self.k, random_state=random_state, n_init="auto")
        km.fit(embeddings)
        self.centroids = self._normalize_np(km.cluster_centers_)
        return self

    def _require_centroids(self) -> np.ndarray:
        if self.centroids is None:
            gen = np.random.default_rng(self.prime)
            centroids = gen.normal(size=(self.k, self.encoder.dim)).astype(np.float32)
            self.centroids = self._normalize_np(centroids)
        return self.centroids

    def assign(self, sentence: str) -> int:
        centroids = self._require_centroids()
        emb = self.encoder.encode_text(sentence).numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        sims = centroids @ emb
        return int(np.argmax(sims))

    def margin_ok(self, sentence: str) -> bool:
        centroids = self._require_centroids()
        emb = self.encoder.encode_text(sentence).numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        distances = 1.0 - (centroids @ emb)
        order = np.argsort(distances)
        if len(order) < 2:
            return True
        return bool(distances[order[0]] < distances[order[1]] - self.margin)

    def valid_clusters(self, previous_cluster: int) -> set:
        gen = _seeded_generator("ksem-valid", int(previous_cluster) * self.prime)
        count = max(1, int(self.k * self.gamma))
        return set(torch.randperm(self.k, generator=gen)[:count].tolist())

    def detect(self, text: str) -> float:
        sentences = sentence_tokenize(text)
        if len(sentences) < 2:
            return 0.0
        clusters = [self.assign(s) for s in sentences]
        valid_count = 0
        for idx in range(1, len(clusters)):
            if clusters[idx] in self.valid_clusters(clusters[idx - 1]):
                valid_count += 1
        return _z_from_count(valid_count, len(clusters) - 1, self.gamma)


class KSemStampLogitsProcessor(LogitsProcessor):
    """Token-level compatibility proxy for k-SemStamp.

    Official k-SemStamp is sentence-level rejection sampling. This processor is
    useful only for smoke tests and should not be reported as the paper-level
    implementation.
    """

    sentence_level_required = True

    def __init__(
        self,
        vocab_size: int,
        k: int = 64,
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 15485863,
        **_: object,
    ):
        self.vocab_size = vocab_size
        self.k = k
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.token_clusters = torch.tensor(
            [stable_hash_int("ksem-token", hash_key, i, modulo=k) for i in range(vocab_size)],
            dtype=torch.long,
        )

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        clusters = self.token_clusters.to(scores.device)
        for batch_idx in range(input_ids.shape[0]):
            prev = int(input_ids[batch_idx, -1].item())
            prev_cluster = stable_hash_int("ksem-prev", self.hash_key, prev, modulo=self.k)
            valid = KSemStampWatermark(k=self.k, gamma=self.gamma, prime=self.hash_key).valid_clusters(prev_cluster)
            valid_tensor = torch.tensor(sorted(valid), dtype=torch.long, device=scores.device)
            scores[batch_idx, torch.isin(clusters, valid_tensor)] += self.delta
        return scores


def detect_k_semstamp_text(
    text: str,
    k: int = 64,
    gamma: float = 0.5,
    prime: int = 15485863,
    centroids: Optional[np.ndarray] = None,
) -> float:
    watermark = KSemStampWatermark(k=k, gamma=gamma, prime=prime, centroids=centroids)
    return watermark.detect(text)


@torch.no_grad()
def generate_k_semstamp_sentences(
    model,
    tokenizer,
    prompt: str,
    watermark: KSemStampWatermark,
    num_sentences: int = 4,
    maxout: int = 20,
    sentence_max_tokens: int = 48,
    device: str = "cuda",
) -> str:
    """Sentence-level k-SemStamp rejection generation.

    This follows the paper structure: generate candidate sentences until their
    cluster is valid for the previous sentence and the margin criterion passes.
    """
    context = prompt.strip()
    previous_sentence = sentence_tokenize(prompt)[-1] if sentence_tokenize(prompt) else prompt

    for _ in range(num_sentences):
        prev_cluster = watermark.assign(previous_sentence)
        valid = watermark.valid_clusters(prev_cluster)
        accepted_sentence: Optional[str] = None
        last_sentence = ""

        for _attempt in range(maxout):
            inputs = tokenizer(context, return_tensors="pt", truncation=True).to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=sentence_max_tokens,
                do_sample=True,
                temperature=0.8,
                num_beams=1,
            )
            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            suffix = decoded[len(context) :].strip() if decoded.startswith(context) else decoded
            candidates = sentence_tokenize(suffix)
            candidate = candidates[0] if candidates else suffix
            last_sentence = candidate
            cluster = watermark.assign(candidate)
            if cluster in valid and watermark.margin_ok(candidate):
                accepted_sentence = candidate
                break

        if accepted_sentence is None:
            accepted_sentence = last_sentence
        if accepted_sentence:
            context = (context + " " + accepted_sentence).strip()
            previous_sentence = accepted_sentence

    return context

