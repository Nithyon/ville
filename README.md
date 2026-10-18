# VILLE — Verified Inference Leak Locator using E-values

> Named after **Ville's inequality**, the result that gives the monitor its anytime false-alarm guarantee.

## Ideation

This is an **ideation-stage hackathon submission** (theme: AI & Cybersecurity, track: Threat Detection / AI Security).
The submission deck describes the plan; this repository holds the working core and will keep growing.
What exists today is a faithful CPU implementation of the detector plus a synthetic test bed.
The real-model replay harness is the next milestone (see Status).

## The problem

A compromised LLM inference server can steal the model's weights — or private prompts — by hiding payload bits
in plausible token choices (e.g. picking the runner-up token in a near-tie to mean "1").
The responses still read normally. The model owner can replay each prompt with the committed sampling seed on a
**trusted verifier** and look for token mismatches, but GPU numerical nondeterminism already causes benign
mismatches in 41–87% of responses. A patient attacker hides inside that noise.

VILLE reproduces the **prompt-level e-process** of Ortega-Fernandez, Kowalczyk & Warr (2026): it turns
whole-response mismatch events at 7 nested seeded-margin thresholds into a betting martingale, accumulates weak
evidence across responses, and alarms when evidence reaches `1/α = 200`. Under the calibration-transfer
assumption, the probability of **ever** raising a false alarm, over an unbounded horizon, is at most `α + γ = 1%`.

## Credit

This is an **independent, unofficial reproduction**. All method credit belongs to the authors:

- I. Ortega-Fernandez, M. Kowalczyk, K. Warr. *Anytime-valid detection of LLM weight exfiltration.*
  NeurIPS 2026 Workshop "E-Values: From Statistics to ML". [arXiv:2610.11843](https://arxiv.org/abs/2610.11843)
- R. Rinberg et al. *Verifying LLM inference to detect model weight exfiltration.* [arXiv:2511.02620](https://arxiv.org/abs/2511.02620) (hard-alarm baseline)
- A. Karvonen et al. *DiFR: Inference verification despite nondeterminism.* [arXiv:2511.20621](https://arxiv.org/abs/2511.20621)

## What's here

| Path | What it does |
|---|---|
| `ville/eprocess.py` | e-process monitor: margin thresholds, nested events (Eq. 2), Clopper–Pearson bounds (Eq. 3), KT-estimator bets (Eq. 8), mixture + alarm (Eq. 5–7) |
| `ville/baseline.py` | Rinberg et al. hard per-token alarm (GLS score, τ = −5, rank R = 8) |
| `ville/simulate.py` | **SYNTHETIC** replay-trace generator: benign, seed-blind and seed-aware attackers |
| `examples/run_synthetic.py` | Table-1-style evaluation with bootstrap streams; saves `results/evidence_paths.png` |
| `tests/` | validity and power checks |

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q tests
python examples/run_synthetic.py --streams 200
```

## Synthetic results (our run, `--streams 200 --seed 0`)

**These come from synthetic traces, not from the paper's models.** They only show the pipeline works end to end.

```
setting                  bits/tok  e-proc resp. to alarm e-proc % (atk/ben)  baseline % (atk/ben)
seed-blind                 0.1503  3 [3-3]               100.0 / 0.0           0.0 / 0.0
seed-aware d_max=0.02      0.0285  184 [97-234]           73.5 / 0.0           0.0 / 0.0
seed-aware d_max=0.05      0.0687  14 [12-19]            100.0 / 0.0           0.0 / 0.0
seed-aware d_max=0.1       0.1312  5 [5-7]               100.0 / 0.0           0.0 / 0.0
```

Caveats: the simulator's seed-aware `d_max` caps the seeded margin directly (the paper caps a fixed-seed score
gap), and it does not model the benign low-rank prescriptions that make the baseline false-alarm in the paper,
so the baseline columns here are not comparable to the paper's.

## Status

- [x] e-process monitor (Section 2 + Appendix B)
- [x] Rinberg et al. hard-alarm baseline
- [x] synthetic simulator + bootstrap evaluation
- [x] tests
- [ ] real replay harness: Hugging Face `transformers` trusted verifier vs. a vLLM / TensorRT-LLM "production" server, seeded Gumbel–Max sampling, on Llama-3.2-3B and NVIDIA Nemotron Nano
- [ ] LMSYS-Chat-1M prompts with the paper's 125 / 125 / 250 splits
- [ ] real seed-blind (arithmetic coding) and seed-aware (top-two binary) codecs
- [ ] continuous-score e-process on the fixed-seed likelihood score (the paper's future work)
- [ ] Nemotron-written SOC alerts when the monitor fires
- [ ] live dashboard

## License

Our code is MIT-licensed (see `LICENSE`). The paper is CC BY 4.0 and belongs to its authors.
