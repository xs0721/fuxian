"""test26: Lost in Overlap / logit-watermark collision proxy.

This script studies a central phenomenon behind overlap/collision papers:
different watermark keys can have non-trivial greenlist overlap, which creates
false-attribution risk when a detector tests many candidate providers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from watermark_gap_utils import (
    DEFAULT_GAMMA,
    DEFAULT_KEY,
    _green_set,
    encode_ids,
    kgw_z_from_ids,
    load_benchmark_texts,
    load_tokenizer,
)


OUTPUT = Path("attack_26_logit_collision.png")
KEYS = [15485863, 15485867, 15485869, 15485873, 15485879]


def sample_prev_tokens(tokenizer, texts, max_tokens=36):
    seen = []
    used = set()
    for text in texts:
        ids = encode_ids(tokenizer, text)
        for tok in ids[:-1]:
            tok = int(tok)
            if tok not in used:
                used.add(tok)
                seen.append(tok)
            if len(seen) >= max_tokens:
                return seen
    return seen


def jaccard_for_keys(prev_tokens, vocab_size, key_a, key_b):
    values = []
    for prev in prev_tokens:
        green_a = _green_set(prev, vocab_size, DEFAULT_GAMMA, key_a)
        green_b = _green_set(prev, vocab_size, DEFAULT_GAMMA, key_b)
        inter = len(green_a & green_b)
        union = len(green_a | green_b)
        values.append(inter / max(union, 1))
    return float(np.mean(values)) if values else 0.0


def overlap_matrix(prev_tokens, vocab_size, keys):
    matrix = np.zeros((len(keys), len(keys)), dtype=float)
    for i, key_a in enumerate(keys):
        for j, key_b in enumerate(keys):
            matrix[i, j] = jaccard_for_keys(prev_tokens, vocab_size, key_a, key_b)
    return matrix


def attribution_scores(tokenizer, texts, vocab_size, keys):
    rows = []
    for sample_id, text in enumerate(texts):
        ids = encode_ids(tokenizer, text)
        if len(ids) < 5:
            continue
        for key in keys:
            z = kgw_z_from_ids(ids, vocab_size, key=key)
            rows.append(
                {
                    "sample_id": sample_id,
                    "key": key,
                    "is_true_key": key == DEFAULT_KEY,
                    "z": z,
                }
            )
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("Test 26: Lost in Overlap / logit watermark collision proxy")
    print("=" * 80)

    tokenizer, vocab_size = load_tokenizer()
    df = load_benchmark_texts(limit=48)
    if "Text_KGW" not in df.columns:
        raise RuntimeError("Text_KGW column is required for collision analysis.")
    texts = [str(x) for x in df["Text_KGW"].dropna().tolist() if str(x).strip()][:32]
    if not texts:
        raise RuntimeError("No KGW benchmark text found.")

    prev_tokens = sample_prev_tokens(tokenizer, texts, max_tokens=36)
    if not prev_tokens:
        raise RuntimeError("Unable to sample previous tokens from benchmark text.")

    matrix = overlap_matrix(prev_tokens, vocab_size, KEYS)
    expected = DEFAULT_GAMMA / (2.0 - DEFAULT_GAMMA)
    scores = attribution_scores(tokenizer, texts, vocab_size, KEYS)
    true_scores = scores[scores["is_true_key"]]
    wrong_scores = scores[~scores["is_true_key"]]
    wrong_best = wrong_scores.groupby("sample_id")["z"].max()

    true_detect_rate = float((true_scores["z"] >= 4.0).mean()) if not true_scores.empty else 0.0
    wrong_false_attr_rate = float((wrong_best >= 4.0).mean()) if not wrong_best.empty else 0.0
    margin = true_scores.set_index("sample_id")["z"] - wrong_best

    matrix_df = pd.DataFrame(matrix, index=[str(k) for k in KEYS], columns=[str(k) for k in KEYS])
    print("\nMean greenlist Jaccard overlap across sampled contexts:")
    print(matrix_df.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nExpected independent-key Jaccard at gamma={DEFAULT_GAMMA:.2f}: {expected:.3f}")
    print("\nAttribution risk on KGW benchmark text:")
    print(f"  true-key mean z: {true_scores['z'].mean():.3f}")
    print(f"  wrong-key mean z: {wrong_scores['z'].mean():.3f}")
    print(f"  max wrong-key mean z per sample: {wrong_best.mean():.3f}")
    print(f"  true-key detection rate @ z>=4: {true_detect_rate:.3f}")
    print(f"  wrong-key false attribution rate @ z>=4: {wrong_false_attr_rate:.3f}")
    print(f"  mean attribution margin true_z - best_wrong_z: {margin.mean():.3f}")

    _green_set.cache_clear()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    im = axes[0].imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0].set_xticks(range(len(KEYS)), [str(k) for k in KEYS], rotation=45, ha="right")
    axes[0].set_yticks(range(len(KEYS)), [str(k) for k in KEYS])
    axes[0].set_title("Greenlist Jaccard overlap")
    for i in range(len(KEYS)):
        for j in range(len(KEYS)):
            color = "white" if matrix[i, j] < 0.55 else "black"
            axes[0].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].boxplot(
        [true_scores["z"].to_numpy(), wrong_scores["z"].to_numpy(), wrong_best.to_numpy()],
        labels=["true key", "wrong keys", "best wrong"],
        showmeans=True,
    )
    axes[1].axhline(4.0, color="#666666", linestyle="--", linewidth=1, label="z=4 threshold")
    axes[1].set_title("False-attribution score distribution")
    axes[1].set_ylabel("KGW detector z-score")
    axes[1].legend(frameon=False)
    # X 轴标签旋转
    axes[1].tick_params(axis='x', rotation=45)
    for label in axes[1].get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=120)
    print(f"\nSaved plot: {OUTPUT}")
    print("\nNote: proxy/mechanism reproduction focused on overlap and attribution risk.")


if __name__ == "__main__":
    main()
