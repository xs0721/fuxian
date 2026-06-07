"""Lightweight checks for official-style watermark upgrades.

This script does not import run_experiment.py and does not load OPT. It uses a
tiny fake causal model so STA-1 and KTH custom samplers can be checked quickly.
"""

import types

import torch

import official_watermarks as ow


class TinyCausalModel:
    def __init__(self, vocab_size=32):
        self.vocab_size = vocab_size

    def __call__(self, input_ids, attention_mask=None, past_key_values=None):
        batch = input_ids.shape[0]
        logits = torch.zeros(batch, 1, self.vocab_size, device=input_ids.device)
        for b in range(batch):
            base = int(input_ids[b, -1].item())
            logits[b, 0] = -torch.arange(self.vocab_size, device=input_ids.device).float() / 20.0
            logits[b, 0, (base + 1) % self.vocab_size] += 2.5
            logits[b, 0, (base + 5) % self.vocab_size] += 1.2
        return types.SimpleNamespace(logits=logits, past_key_values=None)


class TinyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(x)) for x in ids)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 32
    model = TinyCausalModel(vocab_size=vocab_size)
    tokenizer = TinyTokenizer()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
    attn = torch.ones_like(prompt)

    sta_ids = ow.generate_sta1(
        model, tokenizer, prompt, attn, max_new_tokens=12, seed=7,
        vocab_size=vocab_size, device=device)
    sta_z = ow.detect_sta1_tokens(sta_ids[0].cpu().tolist(), vocab_size)
    print(f"STA-1 length={sta_ids.shape[1]} z={sta_z:.3f}")

    kth_ids, kth_state = ow.generate_kth_inverse(
        model, tokenizer, prompt, attn, max_new_tokens=12, seed=11,
        vocab_size=vocab_size, key_length=64, block_size=8, device=device)
    kth_z = ow.detect_kth_tokens(
        kth_ids[0].cpu().tolist(), kth_state, n_runs=10,
        max_windows=8, max_offsets=32)
    print(f"KTH length={kth_ids.shape[1]} z={kth_z:.3f}")

    fake_scores = torch.zeros(1, vocab_size, device=device)
    input_ids = torch.tensor([[1, 5, 9, 13]], dtype=torch.long, device=device)

    sir = ow.SIRLogitsProcessor(vocab_size=vocab_size, rank=8)
    sir_scores = sir(input_ids, fake_scores.clone())
    print(f"SIR changed={bool((sir_scores != 0).any().item())}")

    xsir = ow.XSIRLogitsProcessor(vocab_size=vocab_size, num_clusters=8)
    xsir_scores = xsir(input_ids, fake_scores.clone())
    print(f"X-SIR changed={bool((xsir_scores != 0).any().item())}")

    ksem_proc = ow.KSemStampLogitsProcessor(vocab_size=vocab_size, k=8)
    ksem_scores = ksem_proc(input_ids, fake_scores.clone())
    print(f"k-SemStamp proxy changed={bool((ksem_scores != 0).any().item())}")

    ksem = ow.KSemStampWatermark(k=4)
    ksem.fit([
        "Science explains natural systems.",
        "Researchers analyze data carefully.",
        "Stories describe people and places.",
        "A traveler walked through the city.",
        "Models generate text from prompts.",
        "Watermarks help identify generated text.",
        "A chef prepared dinner in the kitchen.",
        "Students read books in the library.",
    ])
    text = "Science explains natural systems. Models generate text from prompts. Watermarks help identify generated text."
    print(f"k-SemStamp detector z={ksem.detect(text):.3f}")


if __name__ == "__main__":
    main()

