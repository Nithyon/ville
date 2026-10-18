"""Table-1-style evaluation on SYNTHETIC replay traces (not the paper's data).

Mirrors the protocol of arXiv:2610.11843: 125 benign responses fit margin thresholds,
125 calibrate event rates, 250 held-out responses form each test pool, and bootstrap
streams of 250 responses are monitored by the e-process and the hard-alarm baseline.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ville import EProcessMonitor, first_alarm  # noqa: E402
from ville.simulate import bootstrap_streams, make_pool  # noqa: E402


def evaluate(monitor, pool, n_streams, length, rng):
    ep_alarms, bl_alarms, paths = [], [], []
    for stream in bootstrap_streams(pool, n_streams, length, rng):
        path, at = monitor.run(stream)
        ep_alarms.append(at)
        bl_alarms.append(first_alarm(stream))
        if len(paths) < 1:
            paths.append(path)
    return ep_alarms, bl_alarms, paths[0]


def summarize(alarms):
    hit = [a for a in alarms if a is not None]
    pct = 100.0 * len(hit) / len(alarms)
    if not hit:
        return "-", pct
    p10, p50, p90 = np.percentile(hit, [10, 50, 90])
    return f"{p50:.0f} [{p10:.0f}-{p90:.0f}]", pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=200)
    ap.add_argument("--length", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    monitor = EProcessMonitor().fit(make_pool("benign", 125, rng), make_pool("benign", 125, rng))
    print(f"thresholds b_m: {np.round(monitor.thresholds, 4).tolist()}")
    print(f"q+_m:           {np.round(monitor.q_plus, 4).tolist()}")
    print(f"alarm threshold 1/alpha = {monitor.alarm_threshold:.0f}\n")

    benign = make_pool("benign", 250, rng)
    b_ep, b_bl, b_path = evaluate(monitor, benign, args.streams, args.length, rng)
    settings = [("seed-blind", "seed_blind", {})] + [
        (f"seed-aware d_max={d}", "seed_aware", {"d_max": d}) for d in (0.02, 0.05, 0.1)
    ]

    header = f"{'setting':<24}{'bits/tok':>9}  {'e-proc resp. to alarm':<22}{'e-proc % (atk/ben)':<20}{'baseline % (atk/ben)':<20}"
    print(f"SYNTHETIC results, {args.streams} bootstrap streams x {args.length} responses")
    print(header)
    print("-" * len(header))
    paths = {"benign": b_path}
    _, b_ep_pct = summarize(b_ep)
    _, b_bl_pct = summarize(b_bl)
    for label, kind, kw in settings:
        pool = make_pool(kind, 250, rng, **kw)
        rate = sum(r["bits"] for r in pool) / sum(len(r["margin"]) for r in pool)
        ep, bl, path = evaluate(monitor, pool, args.streams, args.length, rng)
        med, ep_pct = summarize(ep)
        _, bl_pct = summarize(bl)
        paths[label] = path
        print(f"{label:<24}{rate:>9.4f}  {med:<22}{ep_pct:5.1f} / {b_ep_pct:<12.1f}{bl_pct:5.1f} / {b_bl_pct:.1f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plot")
        return
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, path in paths.items():
        ax.plot(np.arange(1, len(path) + 1), np.log(np.maximum(path, 1e-12)), label=label)
    ax.axhline(np.log(monitor.alarm_threshold), color="k", ls="--", label="alarm (log 200)")
    ax.set_xlabel("responses")
    ax.set_ylabel("log evidence  log M_J")
    ax.set_title("VILLE e-process (SYNTHETIC traces)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/evidence_paths.png", dpi=150)
    print("\nsaved results/evidence_paths.png")


if __name__ == "__main__":
    main()
