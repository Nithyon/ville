"""Prompt-level e-process monitor (Ortega-Fernandez, Kowalczyk, Warr, arXiv:2610.11843).

Independent, unofficial implementation of Section 2 and Appendix B.

A response is a mapping with two equal-length per-token arrays:
    mismatch : bool   -- served token != token prescribed by the committed seed
    margin   : float  -- seeded margin (gap between the two leading perturbed logits)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta

# Percentile levels used to place the K-1 non-zero margin thresholds (paper Table 4).
# Our reading: the percentiles are taken over the seeded margins of all tokens in the
# edge-fitting benign split; b_0 = 0 is always included, giving K = 7 nested events.
DEFAULT_PERCENTILES = (2, 5, 10, 20, 40, 70)


def response_events(response, thresholds) -> np.ndarray:
    """A_{j,m} = 1{exists t: F_{j,t} = 1 and Delta_{j,t} >= b_m} for every m (Eq. 2)."""
    mismatch = np.asarray(response["mismatch"], dtype=bool)
    margin = np.asarray(response["margin"], dtype=float)
    if not mismatch.any():
        return np.zeros(len(thresholds), dtype=float)
    widest = margin[mismatch].max()
    return (widest >= np.asarray(thresholds)).astype(float)


def clopper_pearson_upper(k: int, n: int, eta: float) -> float:
    """One-sided Clopper-Pearson upper bound q+ solving Eq. 3; q+ = 1 when k == n."""
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - eta, k + 1, n - k))


@dataclass
class EProcessMonitor:
    alpha: float = 0.005
    gamma: float = 0.005
    c: float = 0.5
    percentiles: tuple = DEFAULT_PERCENTILES

    thresholds: np.ndarray = field(default=None, init=False)
    q_plus: np.ndarray = field(default=None, init=False)

    @property
    def K(self) -> int:
        return len(self.percentiles) + 1

    @property
    def alarm_threshold(self) -> float:
        return 1.0 / self.alpha

    def fit(self, benign_edge_responses, benign_cal_responses):
        """Fix margin thresholds on one benign split, calibrate event rates on another."""
        margins = np.concatenate([np.asarray(r["margin"], dtype=float) for r in benign_edge_responses])
        self.thresholds = np.concatenate([[0.0], np.percentile(margins, self.percentiles)])
        events = np.array([response_events(r, self.thresholds) for r in benign_cal_responses])
        n = len(benign_cal_responses)
        eta = self.gamma / self.K
        self.q_plus = np.array([clopper_pearson_upper(int(k), n, eta) for k in events.sum(axis=0)])
        self.reset()
        return self

    def reset(self):
        self.log_M = np.zeros(self.K)  # log of each component M_{J,m}
        self.S = np.zeros(self.K)      # earlier responses that triggered event m
        self.j = 0
        self.alarmed_at = None

    def _lambdas(self) -> np.ndarray:
        """Predictable bet from earlier responses only (Eq. 8, KT estimator)."""
        q = self.q_plus
        r_hat = (self.S + 0.5) / (self.j + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (r_hat - q) / (q * (1.0 - q))
            lam = np.minimum(self.c / q, np.maximum(0.0, lam))
        lam[q >= 1.0] = 0.0
        return lam

    def factors(self, response) -> np.ndarray:
        """Update factors E_{j,m} = 1 + lambda_{j,m} (A_{j,m} - q+_m) (Eq. 5)."""
        A = response_events(response, self.thresholds)
        return 1.0 + self._lambdas() * (A - self.q_plus), A

    def update(self, response):
        """Ingest one complete response. Returns (M_J, alarmed)."""
        E, A = self.factors(response)
        self.log_M += np.log(E)
        self.S += A
        self.j += 1
        M = float(np.mean(np.exp(self.log_M)))  # fixed average over K components (Eq. 6)
        alarmed = M >= self.alarm_threshold
        if alarmed and self.alarmed_at is None:
            self.alarmed_at = self.j
        return M, alarmed

    def run(self, responses):
        """Monitor a stream; returns (evidence path, first alarm index or None)."""
        self.reset()
        path = [self.update(r)[0] for r in responses]
        return np.array(path), self.alarmed_at
