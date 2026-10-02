# Centered F2a/F2c/F3c formula correction

The goal document's F2a definition (lines 1274–1313) first subtracts each
channel's time mean, divides its covariance by `T−1`, applies isotropic
shrinkage, and divides by `trace+epsilon`. F2c (lines 1365–1400) and F3c
(lines 1480–1534) explicitly use **that F2a matrix**. F2b, in contrast,
specifies an uncentered `XXᵀ` matrix. Earlier V2 arms named `Document*`
applied an uncentered moment to F2a/F2c/F3c; their predictions and outcomes
remain valid as alternative features, but the prior “document-exact” label
was incorrect. No saved V2 result is relabelled or overwritten.

The independent [V3 implementation](../../src/emgimu/feature_bank/spec_spatial_v3.py)
provides centered F2a and source-only log-Euclidean F2c and ring F3c based on
the same covariance. Direct matrix, tangent-reference and ring-lag/half-window
oracles check the equations, DC-shift invariance and version separation.
F2c adds a declared minimal positive ridge for singular zero-signal windows;
its reference is fitted only on source windows. This first native comparison
tests F2a and F3c; a [separate frozen F2c follow-up](SPEC_F2C_GRAB_REPORT.md)
now provides the V3 F2c native increment.

The [frozen GRAB protocol](SPEC_SPATIAL_GRAB_PROTOCOL.json) reuses the official
checksum-verified 672-trial, eight-subject, three-day, 2048 Hz ring1 split.
Day1 is the only training source; Day2 and Day3 contain 224 held-out trials
each. The baseline F0 probabilities replay the parent within `1e-8` absolute
error. All 1,344 held-out arm probabilities are in the [trial table](SPEC_SPATIAL_GRAB_PREDICTIONS.csv),
with independent class, trial, probability and metric readback.

| Arm | Day2 macro-F1 | Day2 log loss | Day3 macro-F1 | Day3 log loss |
| --- | ---: | ---: | ---: | ---: |
| F0 | 0.9551 | 0.1507 | 0.9008 | 0.3007 |
| F0 + centered spec F2a | 0.9312 | 0.1871 | 0.8536 | 0.7619 |
| F0 + centered spec F3c | 0.9198 | 0.1710 | 0.8764 | 0.3860 |

On these previously inspected public subjects, correcting the formulas does
not support either addition as a cross-day default. Compared with the earlier
centered candidate, the exact-denominator V3 changes probabilities by at
most 0.000019 for F2a and 0.003826 for F3c; none of the 448 held-out
decisions changes for either arm. These comparisons are retrospective and
Day3 is descriptive. F3c still assumes the hidden-side electrode sequence
continues the official diagram's visible 8–1–2 numbering. No own-device or
live-recognition claim follows.
