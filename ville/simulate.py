"""SYNTHETIC replay-trace simulator. NOT paper data.

Generates per-token replay records (mismatch, seeded margin, gap, raw rank) whose benign
mismatch rate falls steeply with the seeded margin, mimicking the *shape* of
arXiv:2610.11843 Fig. 2 (tens of percent at tight ties, <0.1% at wide margins).
It exists to exercise the monitor end-to-end on CPU before a real replay harness
(server vs. trusted verifier on an actual LLM) is plugged in.
"""
from __future__ import annotations

import numpy as np

MARGIN_SCALE = 0.7     # seeded margins ~ Exponential(scale)
TIE_PEAK = 0.25        # benign mismatch probability at margin 0
TIE_WIDTH = 0.02       # decay width of the benign mismatch curve
BENIGN_FLOOR = 2e-5    # rare wide-margin benign mismatches


def benign_mismatch_prob(margin):
    return TIE_PEAK * np.exp(-np.asarray(margin) / TIE_WIDTH) + BENIGN_FLOOR


def _base_response(rng, min_len=100, max_len=400):
    n = int(rng.integers(min_len, max_len + 1))
    margin = rng.exponential(MARGIN_SCALE, n)
    mismatch = rng.random(n) < benign_mismatch_prob(margin)
    rank = rng.geometric(0.6, n) - 1
    return {"mismatch": mismatch, "margin": margin, "gap": np.where(mismatch, margin, 0.0), "rank": rank}


def benign_response(rng):
    return _base_response(rng)


def seed_blind_response(rng, eligible_rate=0.15):
    """Attacker sees clean logits but not the seed: payload choices land at arbitrary seeded margins."""
    r = _base_response(rng)
    n = len(r["margin"])
    eligible = rng.random(n) < eligible_rate        # >=2 clean candidates within 0.5 logits
    flipped = eligible & (rng.random(n) < 0.5)      # payload picks a non-prescribed candidate
    r["mismatch"] = r["mismatch"] | flipped
    r["gap"] = np.where(r["mismatch"], r["margin"], 0.0)
    r["rank"] = np.where(flipped, rng.integers(0, 4, n), r["rank"])
    r["bits"] = int(eligible.sum())
    return r


def seed_aware_response(rng, d_max=0.05):
    """Attacker knows the seed: encodes one bit only where the seeded margin <= d_max.

    Note: the paper's d_max caps a fixed-seed *score* gap; here it caps the seeded margin
    directly, a simplification of the synthetic model.
    """
    r = _base_response(rng)
    n = len(r["margin"])
    eligible = r["margin"] <= d_max
    flipped = eligible & (rng.random(n) < 0.5)      # bit 1 -> emit runner-up
    r["mismatch"] = np.where(eligible, flipped, r["mismatch"])
    r["gap"] = np.where(r["mismatch"], r["margin"], 0.0)
    r["rank"] = np.where(flipped, 1, r["rank"])
    r["bits"] = int(eligible.sum())
    return r


def make_pool(kind, n, rng, **kw):
    gen = {"benign": benign_response, "seed_blind": seed_blind_response, "seed_aware": seed_aware_response}[kind]
    return [gen(rng, **kw) for _ in range(n)]


def bootstrap_streams(pool, n_streams, length, rng):
    for _ in range(n_streams):
        idx = rng.integers(0, len(pool), length)
        yield [pool[i] for i in idx]
