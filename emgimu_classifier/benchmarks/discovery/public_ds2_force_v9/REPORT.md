# Public DS2 v9: verified force labels and fixed cross-user screen

The [publisher's v9 release](https://www.kaggle.com/datasets/cinthyazuniga/ds2-emg-signals-three-force-type/versions/9), dated 2026-09-25, adds `LabelForces_All.mat`. It provides 332,108 window-level codes: 0 low, 1 average, 2 high. The v9 MAV and gesture-label MAT files are byte-identical to v8. Because the earlier audit exactly reconstructed every published MAV window from the 2,863 public raw trials, consecutive 116-window groups carry force codes into those raw trials without guessing acquisition order. Every group has a single force code. The [join audit](../DS2_V9_FORCE_LABEL_AUDIT.json) records all hashes and 2,863 trial labels; 2,832 trials also have both a verified subject folder and uniform gesture code. The 30 unknown-subject trials and one mixed-gesture trial are excluded from cross-user scoring.

The [publisher metadata snapshot](../DS2_V9_RELEASE_METADATA.json) preserves the version date and the code-to-force description. The pinned v9 MAT files were downloaded individually using Kaggle's official `kagglehub.dataset_download` API; raw MAT bytes remain outside Git.

The [protocol](PROTOCOL.json) fixes source subject folders 1–12, validation 13–16, and descriptive final 17–20. It excludes code 4 Rest from force physiology, uses four active gesture codes, and averages three nonoverlapping 375-sample raw windows into one prediction per trial. `unseen_high` trains only on source-user low/average trials and tests high force without target or high-force calibration. `product_all` trains on all three source-user force levels. Both use fixed balanced logistic classifiers and source-only feature/scaler fits. F1 is the current reconstructed global-RMS-normalized channel pattern, not the unavailable historical X1-H.

| Mode / arm | Validation all-force macro-F1 | Validation high-force macro-F1 | Final all-force macro-F1 | Final high-force macro-F1 |
|---|---:|---:|---:|---:|
| Unseen high, F0 | 0.2778 | 0.2779 | 0.4230 | 0.4767 |
| Unseen high, F1 | 0.3456 | 0.3663 | 0.3923 | 0.4730 |
| Unseen high, F0+F1 | 0.2871 | 0.2896 | 0.4014 | 0.4845 |
| Product all, F0 | 0.3216 | 0.3212 | 0.3771 | 0.4985 |
| Product all, F1 | 0.3634 | 0.4055 | 0.4551 | 0.5293 |
| Product all, F0+F1 | 0.2993 | 0.2740 | 0.3998 | 0.5298 |

F1 improves validation high-force macro-F1 in both modes. In the unseen-high final users, its high-force score is essentially tied with F0 and the concatenated arm is only slightly higher; hence the screen does not establish a stable historical backbone effect. In product mode, F1 alone improves pooled final macro-F1 and log loss versus F0, but the fixed F0+F1 concatenation does not retain the same gain. Results vary strongly by user and force condition; see [all scores](RESULTS.json). All 5,454 saved trial/arm probabilities were independently [rescored](VERIFICATION.json), and a second complete runner execution reproduced the result JSON and prediction CSV byte-for-byte.

These are instructed subjective force levels, not measured mechanical force. The public source has three EMG channels at 1500 Hz; it is not the user's eight-channel 250 Hz device. The final subjects were inspected by earlier gesture-only experiments, making this a descriptive follow-up rather than a pristine blind test. The v9 force mapping resolves the *public candidate's* missing-label problem, but exact historical B0/X1-H/X2 input identity, code and results remain unavailable. Nothing here changes the live model.

## Fixed 0/1/2/5-shot personal calibration follow-up

The [calibration protocol](CALIBRATION_PROTOCOL.json) freezes nested, per-gesture
trial selection before scoring. In `unseen_high`, all calibration examples come
from low/average force and every evaluated trial has high force. In
`product_all`, five trials per gesture are reserved before any budget, so the
same remaining trials are evaluated at 0, 1, 2 and 5 shots. Each shot means
one native ten-second trial per gesture (four gestures); this is 0/40/80/200
seconds of recorded trial signal, excluding pauses, setup and preparation.
The source models and scalers are unchanged. Source-only variance scales a
personal prototype distribution; its fixed mixing weight is
`shots / (shots + 5)`. No target evaluation label is used to fit it.

| Mode and arm | Validation macro-F1, 0/1/2/5 | Descriptive final macro-F1, 0/1/2/5 |
|---|---|---|
| Unseen high, F0 | .278 / .346 / .339 / .307 | .477 / .521 / .532 / .463 |
| Unseen high, F1 | .366 / .505 / .521 / .574 | .473 / .691 / .703 / .791 |
| Unseen high, F0+F1 | .290 / .323 / .352 / .277 | .485 / .490 / .499 / .549 |
| Product all, F0 | .329 / .357 / .358 / .410 | .393 / .438 / .463 / .472 |
| Product all, F1 | .354 / .407 / .468 / .576 | .461 / .624 / .667 / .738 |
| Product all, F0+F1 | .298 / .325 / .320 / .401 | .417 / .446 / .478 / .548 |

All 48 pooled score cells, per-force and per-subject detail, log loss and Brier
are in [results](CALIBRATION_RESULTS.json) and the compact
[curve](CALIBRATION_CURVE.csv). F1's largest descriptive final improvement is
for unseen high force, but it is uneven across the four final subjects: one
subject remains near .32 macro-F1 at five shots. The concatenated arm does not
inherit F1's gain. Product mode uses high-force target calibration and must not
be presented as high-force zero-shot transfer. These are offline, trial-level,
three-channel public-data results, not a deployment setting for the current
eight-channel device.

The independent [verification](CALIBRATION_VERIFICATION.json) recomputes 336
pooled/force/subject score groups from all 12,576 saved probability rows,
checks the 320 calibration assignments against the frozen hash schedule,
confirms disjoint evaluation and identical trial sets across budgets, and finds
zero probability difference between the 0-shot rows and the parent study.
A complete second run reproduced the assignment CSV, predictions CSV and
results JSON byte-for-byte. The source archive stays outside Git; its SHA-256
is recorded in the results.
