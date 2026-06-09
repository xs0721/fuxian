"""test24: Black-box third-party watermark detector proxy.

This is a mechanism reproduction for black-box/third-party detector papers:
the detector only sees text, extracts tokenizer-level watermark evidence, and
reports AUC/FPR/TPR without using model logits or prompts.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from official_watermarks import stable_hash_int
from watermark_gap_utils import (
    DEFAULT_GAMMA,
    DEFAULT_KEY,
    _green_set,
    decode_ids,
    detect_features,
    encode_ids,
    bias_inversion_ids,
    load_benchmark_texts,
    load_tokenizer,
)


OUTPUT = Path("attack_24_blackbox_third_party.png")
FAMILIES = {
    "KGW": "Text_KGW",
    "Unigram": "Text_Unigram",
    "SemStamp": "Text_SemStamp",
}
FAMILY_FEATURE = {
    "KGW": "kgw_z",
    "Unigram": "unigram_z",
    "SemStamp": "semstamp_z",
}


def roc_auc(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def rates_at_threshold(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = scores >= threshold
    pos = max(int(y_true.sum()), 1)
    neg = max(int(len(y_true) - y_true.sum()), 1)
    tpr = float((pred & (y_true == 1)).sum() / pos)
    fpr = float((pred & (y_true == 0)).sum() / neg)
    return tpr, fpr


def unigram_green_set(vocab_size, gamma=DEFAULT_GAMMA, key=42):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(key))
    size = max(1, int(vocab_size * gamma))
    return set(torch.randperm(vocab_size, generator=gen)[:size].tolist())


def nearest_unigram_red(token_id, vocab_size, green):
    token_id = int(token_id)
    for offset in range(1, 256):
        for cand in (token_id + offset, token_id - offset):
            if 0 <= cand < vocab_size and cand not in green:
                return cand
    return token_id


def unigram_inversion_ids(ids, vocab_size, strength=0.85):
    green = unigram_green_set(vocab_size)
    out = []
    changed = 0
    for idx, tok in enumerate(ids):
        tok = int(tok)
        if tok in green:
            gate = stable_hash_int("unigram-invert", idx, tok, modulo=10_000) / 10_000
            if gate < strength:
                tok = nearest_unigram_red(tok, vocab_size, green)
                changed += 1
        out.append(tok)
    return out, changed / max(len(ids), 1)


def semstamp_neutralize_ids(ids, vocab_size, strength=0.85):
    out = []
    changed = 0
    for idx, tok in enumerate(ids):
        tok = int(tok)
        if tok < 100:
            gate = stable_hash_int("semstamp-neutralize", idx, tok, modulo=10_000) / 10_000
            if gate < strength:
                tok = 100 + stable_hash_int("semstamp-replace", idx, tok, modulo=max(vocab_size - 100, 1))
                changed += 1
        out.append(tok)
    return out, changed / max(len(ids), 1)


def neutralize_text(tokenizer, text, family, vocab_size):
    ids = encode_ids(tokenizer, text)
    if family == "KGW":
        new_ids, edit_rate = bias_inversion_ids(ids, vocab_size, strength=0.85, key=DEFAULT_KEY)
    elif family == "Unigram":
        new_ids, edit_rate = unigram_inversion_ids(ids, vocab_size, strength=0.85)
    elif family == "SemStamp":
        new_ids, edit_rate = semstamp_neutralize_ids(ids, vocab_size, strength=0.85)
    else:
        new_ids, edit_rate = list(ids), 0.0
    return decode_ids(tokenizer, new_ids), edit_rate


def third_party_score(features, family):
    detector_evidence = features[FAMILY_FEATURE[family]]
    length_support = min(features["token_count"] / 80.0, 1.0)
    repetition_penalty = 0.25 * features["bigram_repeat_ratio"] + 0.35 * features["trigram_repeat_ratio"]
    return detector_evidence * length_support - repetition_penalty


def build_dataset(tokenizer, vocab_size, limit=32):
    df = load_benchmark_texts(limit=limit)
    rows = []
    for family, col in FAMILIES.items():
        if col not in df.columns:
            continue
        for _, row in df.iterrows():
            text = row.get(col)
            if not isinstance(text, str) or not text.strip():
                continue

            for label, variant, sample_text, edit_rate in [
                (1, "watermarked", text, 0.0),
                (0, "neutralized", *neutralize_text(tokenizer, text, family, vocab_size)),
            ]:
                features = detect_features(tokenizer, sample_text, vocab_size)
                rows.append(
                    {
                        "family": family,
                        "variant": variant,
                        "label": label,
                        "edit_rate": edit_rate,
                        "score": third_party_score(features, family),
                        **features,
                    }
                )
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("Test 24: Black-box third-party detector proxy")
    print("=" * 80)
    print("Input: text only. No logits, no prompt, no model generation.")

    tokenizer, vocab_size = load_tokenizer()
    data = build_dataset(tokenizer, vocab_size)
    if data.empty:
        raise RuntimeError("No benchmark text found. Expected Text_KGW/Text_Unigram/Text_SemStamp columns.")

    overall_auc = roc_auc(data["label"], data["score"])
    tpr_2, fpr_2 = rates_at_threshold(data["label"], data["score"], threshold=2.0)
    tpr_4, fpr_4 = rates_at_threshold(data["label"], data["score"], threshold=4.0)

    family_rows = []
    for family, group in data.groupby("family"):
        family_auc = roc_auc(group["label"], group["score"])
        pos_score = group.loc[group["label"] == 1, "score"].mean()
        neg_score = group.loc[group["label"] == 0, "score"].mean()
        edit_rate = group.loc[group["label"] == 0, "edit_rate"].mean()
        family_rows.append(
            {
                "family": family,
                "auc": family_auc,
                "mean_score_watermarked": pos_score,
                "mean_score_neutralized": neg_score,
                "mean_edit_rate": edit_rate,
            }
        )
    summary = pd.DataFrame(family_rows).sort_values("family")

    print("\nOverall third-party detector:")
    print(f"  AUC: {overall_auc:.3f}")
    print(f"  threshold=2.0  TPR={tpr_2:.3f}  FPR={fpr_2:.3f}")
    print(f"  threshold=4.0  TPR={tpr_4:.3f}  FPR={fpr_4:.3f}")
    print("\nPer-family summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    _green_set.cache_clear()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = ["neutralized", "watermarked"]
    box_data = [data.loc[data["variant"] == label, "score"].to_numpy() for label in labels]
    axes[0].boxplot(box_data, labels=labels, showmeans=True)
    axes[0].axhline(2.0, color="#666666", linestyle="--", linewidth=1, label="threshold 2")
    axes[0].set_title("Third-party text-only score")
    axes[0].set_ylabel("ensemble z-like score")
    axes[0].legend(frameon=False)

    auc_labels = ["Overall"] + summary["family"].tolist()
    auc_values = [overall_auc] + summary["auc"].tolist()
    axes[1].bar(auc_labels, auc_values, color=["#4C78A8", "#72B7B2", "#F58518", "#54A24B"])
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("AUC by detector family")
    axes[1].set_ylabel("AUC")
    for idx, value in enumerate(auc_values):
        if math.isfinite(value):
            axes[1].text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=120)
    print(f"\nSaved plot: {OUTPUT}")
    print("\nNote: proxy/mechanism reproduction, not official unpublished detector code.")


if __name__ == "__main__":
    main()
