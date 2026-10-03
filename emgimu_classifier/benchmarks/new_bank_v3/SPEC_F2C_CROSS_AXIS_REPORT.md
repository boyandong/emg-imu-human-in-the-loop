# Centered F2c: matched public-axis decision

The goal document defines F2a with **time-mean centered** sample covariance. V3
F2c uses that matrix, a declared positive ridge for singular windows, and a
source-only log-Euclidean reference. Earlier V2 experiments used uncentered
moments; their results remain alternative-feature evidence, not the goal-exact
F2c result. This [frozen protocol](SPEC_F2C_CROSS_AXIS_PROTOCOL.json) replays
V3 F2c on the same held-out native trials as the saved F0v2 probabilities.
Its numerical revision uses a `1e-10` SPD ridge matching the matrix eigensystem
floor, so a zero-variance source window maps to its own tangent origin. The
earlier `1e-12` ridge and results remain recoverable at Git commit `1f1a2bf`;
this revision changes no trials, classifier or decision rule.
Each result JSON now stores the Python, NumPy, SciPy, scikit-learn, joblib and
threadpoolctl versions, BLAS/OpenMP backend identities and thread settings
observed in that run. From `emgimu_classifier`, run
`python -m benchmarks.new_bank_v3.numeric_environment` to reject an unlike
current numerical environment before attempting byte-level replay. This does
not establish the environments of earlier experiments or guarantee bitwise
identity across different hardware.
The [six-cell table](SPEC_F2C_CROSS_AXIS_CELLS.csv), [guard audit](SPEC_F2C_CROSS_AXIS_AUDIT.json), and saved [wearing](SPEC_F2C_WEARING_PREDICTIONS.csv) and [transfer](SPEC_F2C_TRANSFER_PREDICTIONS.csv) probabilities permit independent trial-level readback.

| Axis and phase | Trials | F0 F1 | F0+F2c F1 | F0 loss | F0+F2c loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wearing shift, validation | 120 | 0.5775 | 0.6110 | 1.2916 | 1.0713 |
| Wearing shift, descriptive final | 120 | 0.6170 | 0.5889 | 1.0838 | 1.0421 |
| MANUS session, validation | 108 | 0.3914 | 0.4778 | 2.1203 | 2.4545 |
| MANUS session, descriptive final | 108 | 0.4600 | 0.5284 | 1.7231 | 1.7266 |
| GRAB unseen user, validation | 56 | 0.8054 | 0.7371 | 0.4650 | 0.6960 |
| GRAB unseen user, descriptive final | 56 | 0.9458 | 0.8940 | 0.1994 | 0.3497 |

The existing public default guard requires no validation F1 loss or log-loss
rise on any of these three axes. MANUS validation loss rises; GRAB unseen-user
F1 falls and loss rises. Thus V3 F2c **fails the universal-default guard** and
F0v2 remains the three-axis default. The wearing validation gain is useful
specialist evidence but its final F1 reverses. Final groups are descriptive and
were not used to choose the rule. Separate [GRAB cross-day evidence](SPEC_F2C_GRAB_REPORT.md)
also reverses between Day2 and Day3.

These are previously inspected public cohorts with different native gesture
sets and acquisition rates, not a prospective independent confirmation or a
250 Hz own-device/live-recognition estimate.
