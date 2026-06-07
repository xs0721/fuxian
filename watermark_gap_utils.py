"""Utilities for lightweight gap-filling watermark experiments.

The helpers here intentionally avoid loading a causal language model. They use
cached benchmark text plus tokenizer-level detectors, so tests 24-26 can run on
a laptop and still produce reproducible evidence for chapter-four gap analysis.
"""

from __future__ import annotations

import math
import os
import re
import hashlib
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("USE_TORCH", "0")

CACHE_DIR = "E:/Your_Cloud_Drive/hf_cache"
MODEL_NAME = "facebook/opt-125m"
CSV_FILENAME = "watermark_benchmark_results.csv"
DEFAULT_VOCAB_SIZE = 50272
DEFAULT_KEY = 15485863
DEFAULT_GAMMA = 0.5
_HASH_SCALE = 1_000_000
_TORCH_MODULE = None
_TORCH_TRIED = False


def stable_hash_int(*parts: object, modulo: int = 2**31 - 1) -> int:
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "big") % modulo


def _get_torch():
    global _TORCH_MODULE, _TORCH_TRIED
    if _TORCH_TRIED:
        return _TORCH_MODULE
    _TORCH_TRIED = True
    try:
        import torch

        _TORCH_MODULE = torch
    except Exception as exc:
        _TORCH_MODULE = None
        print(f"[warn] PyTorch unavailable for gap utilities; using hash fallback: {exc}")
    return _TORCH_MODULE


class SimpleHashTokenizer:
    """Fallback tokenizer used only when the HF tokenizer is unavailable."""

    vocab_size = DEFAULT_VOCAB_SIZE
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        pieces = re.findall(r"\w+|[^\w\s]", str(text), flags=re.UNICODE)
        ids = []
        for piece in pieces:
            match = re.fullmatch(r"tok(\d+)", piece)
            if match:
                ids.append(int(match.group(1)) % self.vocab_size)
            else:
                ids.append(stable_hash_int("simple-token", piece, modulo=self.vocab_size))
        if add_special_tokens:
            ids = [self.eos_token_id] + ids
        if return_tensors == "pt":
            torch = _get_torch()
            if torch is None:
                raise RuntimeError("return_tensors='pt' requires PyTorch")
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{int(i)}" for i in ids if not skip_special_tokens or int(i) != self.eos_token_id)


def load_tokenizer():
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)
        vocab_size = int(getattr(tokenizer, "vocab_size", DEFAULT_VOCAB_SIZE))
        return tokenizer, vocab_size
    except Exception as exc:
        print(f"[warn] Falling back to SimpleHashTokenizer: {exc}")
        return SimpleHashTokenizer(), DEFAULT_VOCAB_SIZE


def load_benchmark_texts(limit: int = 80) -> pd.DataFrame:
    df = pd.read_csv(CSV_FILENAME)
    text_cols = [c for c in df.columns if c.startswith("Text_")]
    keep = ["Dataset", "Sample_ID"] + text_cols
    return df[keep].head(limit).copy()


def encode_ids(tokenizer, text: str) -> List[int]:
    return [int(x) for x in tokenizer.encode(str(text), add_special_tokens=False)]


def decode_ids(tokenizer, ids: Sequence[int]) -> str:
    return tokenizer.decode([int(x) for x in ids], skip_special_tokens=True)


def _hash_is_green(token_id: int, prev_token: int, gamma: float = DEFAULT_GAMMA, key: int = DEFAULT_KEY) -> bool:
    threshold = int(float(gamma) * _HASH_SCALE)
    value = stable_hash_int("kgw-green", int(key), int(prev_token), int(token_id), modulo=_HASH_SCALE)
    return value < threshold


@lru_cache(maxsize=32)
def _green_set(prev_token: int, vocab_size: int, gamma: float, key: int) -> frozenset:
    size = max(1, int(vocab_size * gamma))
    torch = _get_torch()
    if torch is not None:
        seed = (int(key) * int(prev_token)) % (2**63 - 1)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        return frozenset(torch.randperm(vocab_size, generator=gen)[:size].tolist())

    return frozenset(
        tok
        for tok in range(vocab_size)
        if _hash_is_green(tok, int(prev_token), float(gamma), int(key))
    )


def is_green(token_id: int, prev_token: int, vocab_size: int, gamma: float = DEFAULT_GAMMA, key: int = DEFAULT_KEY) -> bool:
    if _get_torch() is None:
        return _hash_is_green(token_id, prev_token, gamma, key)
    return int(token_id) in _green_set(int(prev_token), int(vocab_size), float(gamma), int(key))


def kgw_z_from_ids(ids: Sequence[int], vocab_size: int, gamma: float = DEFAULT_GAMMA, key: int = DEFAULT_KEY) -> float:
    if len(ids) < 2:
        return 0.0
    green = 0
    total = 0
    for i in range(1, len(ids)):
        if is_green(ids[i], ids[i - 1], vocab_size, gamma, key):
            green += 1
        total += 1
    variance = total * gamma * (1.0 - gamma)
    return (green - total * gamma) / math.sqrt(variance) if variance > 0 else 0.0


def unigram_z_from_ids(ids: Sequence[int], vocab_size: int, gamma: float = DEFAULT_GAMMA, key: int = 42) -> float:
    if not ids:
        return 0.0
    torch = _get_torch()
    if torch is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(key))
        green = set(torch.randperm(vocab_size, generator=gen)[: max(1, int(vocab_size * gamma))].tolist())
        count = sum(1 for tok in ids if int(tok) in green)
    else:
        threshold = int(float(gamma) * _HASH_SCALE)
        count = sum(
            1
            for tok in ids
            if stable_hash_int("unigram-green", int(key), int(tok), modulo=_HASH_SCALE) < threshold
        )
    variance = len(ids) * gamma * (1.0 - gamma)
    return (count - len(ids) * gamma) / math.sqrt(variance) if variance > 0 else 0.0


def semstamp_proxy_z(ids: Sequence[int], vocab_size: int) -> float:
    if not ids:
        return 0.0
    target_ratio = 100 / vocab_size
    count = sum(1 for tok in ids if int(tok) < 100)
    variance = len(ids) * target_ratio * (1.0 - target_ratio)
    if variance <= 0:
        return 0.0
    return min((count - len(ids) * target_ratio) / math.sqrt(variance), 8.5)


def detect_features(tokenizer, text: str, vocab_size: int, key: int = DEFAULT_KEY) -> dict:
    ids = encode_ids(tokenizer, text)
    words = re.findall(r"\w+", str(text), flags=re.UNICODE)
    chars = list(str(text))
    bigrams = list(zip(ids, ids[1:]))
    trigrams = list(zip(ids, ids[1:], ids[2:]))
    return {
        "token_count": len(ids),
        "word_count": len(words),
        "unique_token_ratio": len(set(ids)) / max(len(ids), 1),
        "bigram_repeat_ratio": 1.0 - len(set(bigrams)) / max(len(bigrams), 1),
        "trigram_repeat_ratio": 1.0 - len(set(trigrams)) / max(len(trigrams), 1),
        "punct_ratio": sum(1 for c in chars if re.match(r"[^\w\s]", c, flags=re.UNICODE)) / max(len(chars), 1),
        "digit_ratio": sum(1 for c in chars if c.isdigit()) / max(len(chars), 1),
        "kgw_z": kgw_z_from_ids(ids, vocab_size, key=key),
        "unigram_z": unigram_z_from_ids(ids, vocab_size),
        "semstamp_z": semstamp_proxy_z(ids, vocab_size),
    }


def nearest_red_token(token_id: int, prev_token: int, vocab_size: int, gamma: float = DEFAULT_GAMMA, key: int = DEFAULT_KEY) -> int:
    token_id = int(token_id)
    for offset in range(1, 256):
        for cand in (token_id + offset, token_id - offset):
            if 0 <= cand < vocab_size and not is_green(cand, prev_token, vocab_size, gamma, key):
                return cand
    return token_id


def nearest_green_token(token_id: int, prev_token: int, vocab_size: int, gamma: float = DEFAULT_GAMMA, key: int = DEFAULT_KEY) -> int:
    token_id = int(token_id)
    for offset in range(1, 256):
        for cand in (token_id + offset, token_id - offset):
            if 0 <= cand < vocab_size and is_green(cand, prev_token, vocab_size, gamma, key):
                return cand
    return token_id


def kgw_bias_ids(
    ids: Sequence[int],
    vocab_size: int,
    strength: float = 0.75,
    gamma: float = DEFAULT_GAMMA,
    key: int = DEFAULT_KEY,
) -> Tuple[List[int], float]:
    """Push a sequence toward KGW green-token evidence for controlled proxies."""
    if len(ids) < 2:
        return list(ids), 0.0
    out = [int(ids[0])]
    changed = 0
    threshold = max(0.0, min(1.0, float(strength)))
    for idx, tok in enumerate(ids[1:], start=1):
        prev = out[-1]
        tok = int(tok)
        if not is_green(tok, prev, vocab_size, gamma, key):
            gate = stable_hash_int("kgw-bias", idx, tok, prev, modulo=10_000) / 10_000
            if gate < threshold:
                tok = nearest_green_token(tok, prev, vocab_size, gamma, key)
                changed += 1
        out.append(tok)
    return out, changed / max(len(ids) - 1, 1)


def bias_inversion_ids(
    ids: Sequence[int],
    vocab_size: int,
    strength: float = 0.5,
    gamma: float = DEFAULT_GAMMA,
    key: int = DEFAULT_KEY,
) -> Tuple[List[int], float]:
    """Replace a fraction of green tokens with nearby red-token ids."""
    if len(ids) < 2:
        return list(ids), 0.0
    out = [int(ids[0])]
    changed = 0
    candidates = 0
    threshold = max(0.0, min(1.0, float(strength)))
    for idx, tok in enumerate(ids[1:], start=1):
        prev = out[-1]
        tok = int(tok)
        if is_green(tok, prev, vocab_size, gamma, key):
            candidates += 1
            gate = stable_hash_int("bias-invert", idx, tok, prev, modulo=10_000) / 10_000
            if gate < threshold:
                tok = nearest_red_token(tok, prev, vocab_size, gamma, key)
                changed += 1
        out.append(tok)
    return out, changed / max(len(ids) - 1, 1)


def word_drop_text(text: str, ratio: float = 0.2) -> str:
    words = str(text).split()
    if not words:
        return str(text)
    kept = []
    for idx, word in enumerate(words):
        gate = stable_hash_int("drop", idx, word, modulo=10_000) / 10_000
        if gate >= ratio:
            kept.append(word)
    return " ".join(kept)


def collect_text_records(df: pd.DataFrame, columns: Iterable[str]) -> List[Tuple[str, str, str]]:
    records = []
    for _, row in df.iterrows():
        dataset = str(row.get("Dataset", "unknown"))
        for col in columns:
            text = row.get(col)
            if isinstance(text, str) and text.strip():
                records.append((dataset, col.replace("Text_", ""), text))
    return records
