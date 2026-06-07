"""test25: Bias Inversion watermark removal proxy.

The experiment estimates green-token bias from KGW-style text and then pushes
candidate green tokens toward nearby red tokens. It is a lightweight mechanism
reproduction, not a claim of using the original paper's private implementation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from watermark_gap_utils import (
    DEFAULT_KEY,
    _green_set,
    bias_inversion_ids,
    decode_ids,
    encode_ids,
    kgw_z_from_ids,
    load_benchmark_texts,
    load_tokenizer,
    word_drop_text,
)


OUTPUT = Path("attack_25_bias_inversion.png")
STRENGTHS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def evaluate_strength(tokenizer, texts, vocab_size, strength):
    rows = []
    for text in texts:
        ids = encode_ids(tokenizer, text)
        if len(ids) < 5:
            continue
        before_z = kgw_z_from_ids(ids, vocab_size, key=DEFAULT_KEY)
        attacked_ids, edit_rate = bias_inversion_ids(ids, vocab_size, strength=strength, key=DEFAULT_KEY)
        after_z = kgw_z_from_ids(attacked_ids, vocab_size, key=DEFAULT_KEY)
        rows.append(
            {
                "strength": strength,
                "before_z": before_z,
                "after_z": after_z,
                "z_drop": before_z - after_z,
                "edit_rate": edit_rate,
                "token_count": len(ids),
            }
        )
    return rows


def evaluate_word_drop_baseline(tokenizer, texts, vocab_size, ratio=0.2):
    rows = []
    for text in texts:
        ids = encode_ids(tokenizer, text)
        if len(ids) < 5:
            continue
        before_z = kgw_z_from_ids(ids, vocab_size, key=DEFAULT_KEY)
        dropped = word_drop_text(text, ratio=ratio)
        dropped_ids = encode_ids(tokenizer, dropped)
        after_z = kgw_z_from_ids(dropped_ids, vocab_size, key=DEFAULT_KEY)
        rows.append(
            {
                "before_z": before_z,
                "after_z": after_z,
                "z_drop": before_z - after_z,
                "edit_rate": 1.0 - len(dropped_ids) / max(len(ids), 1),
            }
        )
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("Test 25: Bias Inversion proxy attack")
    print("=" * 80)

    tokenizer, vocab_size = load_tokenizer()
    df = load_benchmark_texts(limit=64)
    if "Text_KGW" not in df.columns:
        raise RuntimeError("Text_KGW column is required for the Bias Inversion proxy.")
    texts = [str(x) for x in df["Text_KGW"].dropna().tolist() if str(x).strip()]
    texts = texts[:48]
    if not texts:
        raise RuntimeError("No KGW benchmark text found.")

    all_rows = []
    for strength in STRENGTHS:
        all_rows.extend(evaluate_strength(tokenizer, texts, vocab_size, strength))
    result = pd.DataFrame(all_rows)

    summary = (
        result.groupby("strength")
        .agg(
            mean_before_z=("before_z", "mean"),
            mean_after_z=("after_z", "mean"),
            mean_z_drop=("z_drop", "mean"),
            mean_edit_rate=("edit_rate", "mean"),
            detected_rate=("after_z", lambda s: float((s >= 4.0).mean())),
        )
        .reset_index()
    )
    baseline = evaluate_word_drop_baseline(tokenizer, texts, vocab_size, ratio=0.2)

    print("\nBias inversion curve:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nWord-drop baseline (20% deterministic drop):")
    print(
        f"  after_z={baseline['after_z'].mean():.3f}  "
        f"z_drop={baseline['z_drop'].mean():.3f}  "
        f"edit_rate={baseline['edit_rate'].mean():.3f}"
    )

    _green_set.cache_clear()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(summary["strength"], summary["mean_after_z"], marker="o", color="#4C78A8", label="Bias inversion z")
    ax1.axhline(4.0, color="#666666", linestyle="--", linewidth=1, label="z=4 threshold")
    ax1.axhline(baseline["after_z"].mean(), color="#F58518", linestyle=":", linewidth=2, label="word-drop baseline")
    ax1.set_xlabel("inversion strength")
    ax1.set_ylabel("mean KGW z-score after attack")
    ax1.set_ylim(min(summary["mean_after_z"].min(), baseline["after_z"].mean()) - 0.5, summary["mean_before_z"].max() + 0.5)
    ax1.legend(loc="upper right", frameon=False)

    ax2 = ax1.twinx()
    ax2.bar(summary["strength"], summary["mean_edit_rate"], width=0.08, alpha=0.28, color="#54A24B", label="edit rate")
    ax2.set_ylabel("mean token edit rate")
    ax2.set_ylim(0.0, max(summary["mean_edit_rate"].max() * 1.35, 0.1))

    fig.suptitle("Bias Inversion: reversing KGW green-token evidence")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=120)
    print(f"\nSaved plot: {OUTPUT}")
    print("\nNote: proxy/mechanism reproduction, not official unpublished RL or paraphrase code.")


if __name__ == "__main__":
    main()
