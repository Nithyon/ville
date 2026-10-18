"""Hard per-token alarm baseline (Rinberg et al., arXiv:2511.02620), as instantiated in arXiv:2610.11843 App. C.3.

A response additionally carries:
    gap  : float -- gap on the perturbed logits between the prescribed and the served token (>= 0)
    rank : int   -- zero-indexed raw-logit rank of the served token
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def gls(gap, sigma: float = 1.0) -> np.ndarray:
    """Fixed-seed likelihood score GLS = log Phi(-gap / (sigma * sqrt(2))) (Eq. 11)."""
    return norm.logcdf(-np.asarray(gap, dtype=float) / (sigma * np.sqrt(2.0)))


def has_dangerous_token(response, tau: float = -5.0, R: int = 8, sigma: float = 1.0) -> bool:
    score = gls(response["gap"], sigma)
    rank = np.asarray(response["rank"])
    return bool(np.any((score < tau) & (rank > R)))


def first_alarm(responses, tau: float = -5.0, R: int = 8, sigma: float = 1.0):
    """Alarm at the first response containing a dangerous token; returns 1-based index or None."""
    for j, r in enumerate(responses, start=1):
        if has_dangerous_token(r, tau, R, sigma):
            return j
    return None
