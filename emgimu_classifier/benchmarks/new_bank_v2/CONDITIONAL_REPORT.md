# New-v2 conditional value of F2a and F3c

This Stage-2 analysis tests whether a candidate still adds information after
the other candidate is already present. It uses only the previously frozen
four-arm predictions and the 64 matched pooled, subject, and controlled
condition cells. The comparisons are `F0v2+F3c → F0v2+F2a+F3c` for F2a
and `F0v2+F2a → F0v2+F2a+F3c` for F3c. The [protocol](CONDITIONAL_PROTOCOL.json),
[128-row table](CONDITIONAL_VALUE.csv), and [audit](CONDITIONAL_AUDIT.json)
form an independent versioned result; `conditional_analysis.py --verify`
reconstructs the table byte for byte. No training, probability calibration,
weight fitting, or final-set model selection occurs here.

Positive Δlog loss means lower held-out log loss when the candidate is
added; positive ΔF1 means better macro-F1. Pooled cells are:

| Axis | Phase | F2a Δlog loss | F2a ΔF1 | F3c Δlog loss | F3c ΔF1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Force | validation | −0.2052 | +0.0023 | −0.3046 | +0.0021 |
| Force | final | −0.0296 | +0.0094 | −0.0565 | −0.0560 |
| Electrode shift | validation | +0.0346 | +0.0540 | +0.2389 | +0.0422 |
| Electrode shift | final | −0.1187 | −0.1178 | +0.2941 | +0.0564 |
| GRABMyo day | validation | −0.0156 | +0.0023 | +0.0007 | −0.0092 |
| GRABMyo day | final | −0.3305 | −0.0197 | +0.0467 | +0.0031 |
| Song same day | validation | +0.0572 | +0.0148 | +0.0168 | +0.0217 |
| Song same day | final | +0.0690 | −0.0002 | +0.0105 | −0.0002 |

F3c retains conditional value for public electrode shift: both validation
and descriptive final log loss and macro-F1 improve when it is added to
F0v2+F2a. F2a does not show a stable conditional benefit there; its final
addition harms both metrics. On force, neither addition improves pooled
log loss in either phase, even where a tiny F1 change is positive. On
GRABMyo day, F2a has negative conditional log-loss value; F3c's log-loss
changes are small and F1 direction differs by phase. Song is one person
on one date and cannot establish population robustness.

These are paired held-out *conditional prediction* comparisons, not a
causal proof of new physiological information. The per-class F1 and every
subject/condition cell remain in the CSV rather than being hidden by the
pooled table. The exporter adds 128 records to the canonical
`conditional_incremental.csv`; [delivery audit](CONDITIONAL_DELIVERY_AUDIT.json)
records input and output hashes. Because final cohorts were inspected in
earlier project studies, their results are descriptive and do not revise
the original validation choices.
