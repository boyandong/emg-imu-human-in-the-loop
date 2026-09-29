# F2 spatial candidates on matched wearing-domain trials

The [F2b protocol](F2B_WEARING_PROTOCOL.json) and [F2a/F2c protocol](F2_AC_WEARING_PROTOCOL.json)
use the same six native eight-channel, 200 Hz subjects: each person's
`training` domain fits all feature and classifier state, with `trial_1`–`trial_4`
held out. [F2b predictions](F2B_WEARING_PREDICTIONS.csv) retain the exact F0v2
parent replay; [F2a/F2c predictions](F2_AC_WEARING_PREDICTIONS.csv) bind to those
same 240 held-out trial identities. There are 960 saved predictions across
the four arms. Each spatial candidate is added to the same 48-dimensional
F0v2 backbone and source-only trial-mean logistic model. F2a and F2c each
add 36 dimensions; document F2b adds 20 dimensions.

| Split | Arm | Pooled macro-F1 | Log loss | Minimum subject F1 | Worst domain F1 |
|---|---|---:|---:|---:|---:|
| Validation users 15–17 | F0v2 | 0.5775 | 1.2916 | 0.4286 | 0.5433 |
| Validation users 15–17 | + F2a trace covariance | 0.6960 | 1.0786 | 0.4551 | 0.6544 |
| Validation users 15–17 | + F2b document CSP | 0.5444 | 1.3655 | 0.3217 | 0.5082 |
| Validation users 15–17 | + F2c SPD tangent | 0.6110 | 1.0713 | 0.3333 | 0.5928 |
| Descriptive final users 18–20 | F0v2 | 0.6170 | 1.0838 | 0.4453 | 0.5505 |
| Descriptive final users 18–20 | + F2a trace covariance | 0.5996 | 1.0901 | 0.4712 | 0.5400 |
| Descriptive final users 18–20 | + F2b document CSP | 0.6510 | 1.0203 | 0.4743 | 0.6078 |
| Descriptive final users 18–20 | + F2c SPD tangent | 0.5889 | 1.0421 | 0.4524 | 0.4375 |

F2a gives the strongest validation F1 and improves every listed validation
measure against F0v2. F2c improves pooled validation F1/loss but lowers
minimum-subject F1. F2b declines on validation. All three shift direction
in at least one important final measure; the favorable F2b final outcome
cannot retroactively select it. The family remains an internally competing
spatial candidate, not a deployed default. This controlled public wearing
task does not by itself prove transfer to unseen users, another day, or the
user's 250 Hz hardware.

The source [results for F2b](F2B_WEARING_RESULTS.json) and
[results for F2a/F2c](F2_AC_WEARING_RESULTS.json) include complete source/
target trial identities, native archive hash and per-subject/domain scores.
Analytical formula checks live in `tests/test_new_bank_v2.py` for centered,
shrunken F2a; `tests/test_document_signal.py` for uncentered F2b; and
`tests/test_spd_anchor.py` for log-Euclidean F2c tangent distances.
