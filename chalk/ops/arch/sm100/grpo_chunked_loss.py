"""grpo_chunked_loss@sm100 — chalk autoresearch kernel (one file per layer, per arch).

Cell: grpo_chunked_loss@sm100
Entry: grpo_loss_fn(hidden, lm_head_weight, advantages, mask) -> loss   (direction: fwd+bwd)
Oracle: chalk.ops.grpo_loss._eager_grpo_loss   tol fwd_rel=0.02/bwd_rel=0.02 @ fp32
Baseline to beat: the shipped portable chalk kernel   target speedup: up to 2.0x

STATUS: SEED (correct chunked-torch reference, ~1.0x) — autoresearch subagent optimizes this in place.
Verified by the chalk autoresearch verifier (correctness + generalization + timing + roofline +
anti-cheat) on a real sm100 GPU. build() returns the entry callable.
"""

import math

import torch

_BETA = 0.04  # KL(policy || uniform) penalty coefficient (= chalk.ops.grpo_loss.GRPO_BETA)


def build():
    def grpo_loss_fn(hidden, lm_head_weight, advantages, mask):
        n = hidden.shape[0]
        vocab = lm_head_weight.shape[0]
        # Chunk the rows so the materialized fp32 logits tile stays bounded (~4M elements),
        # derived from the RUNTIME vocab — never a benchmark-shape literal. (chalk's static
        # anti-cheat gate flags an in-scope dim hardcoded as a constant, e.g. a literal 2048.)
        rows = min(max(1, n), max(1, (1 << 22) // max(vocab, 1)))
        log_v = math.log(vocab)
        m = mask.float()
        total = hidden.new_zeros((), dtype=torch.float32)
        for s in range(0, n, rows):
            e = min(s + rows, n)
            logits = (hidden[s:e] @ lm_head_weight.t()).float()  # [c, V]
            lse = logits.logsumexp(dim=-1)  # [c]
            logp = logits - lse.unsqueeze(-1)  # [c, V] == log_softmax
            p = logp.exp()  # [c, V] == softmax
            logp_bar = logp.mean(dim=-1)  # [c] mean log-prob across vocab
            neg_entropy = (p * logp).sum(dim=-1)  # [c] == -H(p)
            kl = neg_entropy + log_v  # [c] KL(p || uniform) >= 0
            per_token = -advantages[s:e].float() * logp_bar + _BETA * kl  # [c]
            total = total + (per_token * m[s:e]).sum()
        return total / m.sum().clamp(min=1.0)

    return grpo_loss_fn
