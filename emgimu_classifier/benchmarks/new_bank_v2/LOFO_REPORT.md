# New v2 eight-channel candidate bank: leave-one-family-out

The [protocol](LOFO_PROTOCOL.json) and [runner](lofo_run.py) were frozen in
commit `138f732` before the missing arm was scored. The candidate full bank is
`F0v2+F2a+F3c`. For each removal, the same source-only balanced logistic
pipeline and exactly the same native target trials are used. Full, -F2a and
-F3c probabilities are copied exactly from the independently verified parent
[wearing](WEARING_REPORT.md) and [force](FORCE_REPORT.md) screens. Only -F0v2
(`F2a+F3c`) needs a new source-only fit. The wearing screen previously
selected the full arm on validation users; the force screen selected F0v2
alone, so its full-bank ablation is diagnostic rather than a selected product
model.

Positive values below mean the full bank performs better than the removed
arm; all macro-F1 comparisons use matched trials.

| Dataset | Phase | Full F1 | ΔF1 removing F0v2 | ΔF1 removing F2a | ΔF1 removing F3c |
| --- | --- | ---: | ---: | ---: | ---: |
| Wearing | Validation | 0.7382 | +0.1136 | +0.0540 | +0.0422 |
| Wearing | Final | 0.6560 | +0.0853 | **-0.1178** | +0.0564 |
| Force intensity | Validation | 0.5671 | +0.2526 | +0.0023 | +0.0021 |
| Force intensity | Final | 0.4797 | +0.2047 | +0.0094 | **-0.0560** |

Removing F0v2 damages all four phase–dataset comparisons, including final
log-loss improvements of +0.2901 wearing and +1.1597 force when it is
retained. It is the strongest candidate backbone *for these two native
screens*, not evidence of a universal seven-failure backbone. F2a helps the
wearing validation full bank but harms its descriptive final users: removing
F2a reaches 0.7738 final F1 versus 0.6560 for the full arm, and improves
log loss by 0.1187. This final-only result cannot change the frozen wearing
selection. F3c helps wearing in both phases but harms the force final bank;
removing it raises force final F1 from 0.4797 to 0.5357 and improves log loss
by 0.0565. Thus this three-family concatenation is not a stable shared
full bank across wearing and intensity failures.

The [independent verifier](verify_lofo.py) checks all 5,664 saved arm–trial
probabilities, 1,416 native trial identities, exact parent probability replay
for three arms, source/target trial separation, every matched four-arm set,
all 16 pooled score groups and subject/condition cells, and the full-minus-
removed deltas. The [audit](LOFO_VERIFICATION.json) records paired error
changes. Detailed per-class and per-subject results are retained in
[results](LOFO_RESULTS.json); the canonical full-bank ablation table also
records per-class F1 for every summary cell.
The [idempotent exporter](export_lofo_delivery.py) adds 176 pooled,
subject and condition cells to the required `ablation_full_bank.csv` schema.
Its [delivery audit](LOFO_DELIVERY_AUDIT.json) binds each source table to
the verified result and prediction hashes. Canonical provenance verification
now covers 51,756 records, including all added per-class F1 cells.

LibEMG CIILData wearing transfer is same-user at 200 Hz; LibEMG
ContractionIntensity force transfer is cross-user at 1000 Hz and uses
instructed intensity categories, not measured force. Their class sets,
devices and rates differ. Final users had been inspected in previous project
experiments, so final comparisons are descriptive. This ablation does not
validate the user's 250 Hz live device, later days or a universal feature
bank.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.` using
`python -m benchmarks.new_bank_v2.lofo_run` and
`python -m benchmarks.new_bank_v2.verify_lofo`.
