# Reduced new-v2 MANUS spatial and speed-stratified session study

The [protocol](MANUS_SPATIAL_PROTOCOL.json) was committed before model outcomes
were calculated. It uses the verified public sEMG-MANUS archive: eight EMG
channels at 200 Hz, six finger flexion–extension gestures, users 3–8, and
Session 1 as the sole source for feature fitting, standardization and model
training. Sessions 2 and 3 are validation and descriptive final targets.
Each native recording contributes one mean of up to eight disjoint 200 ms
windows. Every candidate uses the same 108 target recordings per session.

MANUS contains no Rest class, so the rest-conditioned F0v2 cannot be fitted.
This *reduced* bank uses original source-fitted F0 local detail with the
independently implemented new-v2 F2a trace covariance and F3c ring-relative
covariance. At 40 samples, F3c omits its optional half-window drift. This
is a separate study, **not a fourth axis of the complete F0v2 envelope**.
Speed strata are observed within changed sessions; speed and session effects
are not isolated experimentally. All three sessions involve the same six
users, so this is not an unseen-user test.

| Arm | Session 2 macro-F1 | Session 2 log loss | Session 3 macro-F1 | Session 3 log loss | Session 3 minimum-user F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F0 | 0.3410 | **1.9544** | 0.4453 | **1.5920** | **0.2381** |
| F0+F2a | **0.4717** | 2.1688 | 0.4869 | 1.7900 | 0.2222 |
| F0+F3c | 0.3653 | 2.4777 | 0.5020 | 1.9257 | 0.2273 |
| F0+F2a+F3c | 0.4346 | 2.4357 | **0.5235** | 1.9796 | 0.2222 |

The frozen selection rule chooses **F0+F2a** using Session 2 pooled
macro-F1. On the already inspected Session 3 recordings it exceeds F0
by 0.0416 macro-F1, but log loss worsens by 0.1980 and the minimum-user
F1 falls by 0.0159. The full bank has the highest descriptive final F1;
that ranking cannot replace the validation selection. There is no stable
multi-metric case for changing the current live model.

The condition-specific table is also mixed. F0+F2a improves slow-speed F1
from 0.2124 to 0.4295 on Session 2 and from 0.3234 to 0.4037 on Session 3;
its medium-speed Session 3 F1 falls from 0.5333 to 0.5203. The fixed
within-session speed cells remain descriptive because speed is not varied
independently of the session transition.

The saved-prediction [paired protocol](MANUS_SPATIAL_PAIRED_PROTOCOL.json)
adds a finite four-arm conditional, error and interaction analysis without
retraining. Positive conditional Δlog loss means the added family lowers
held-out loss when the *other* family is already present:

| Added family | Session 2 Δlog loss | Session 2 Δmacro-F1 | Session 3 Δlog loss | Session 3 Δmacro-F1 |
| --- | ---: | ---: | ---: | ---: |
| F2a given F0+F3c | +0.0421 | +0.0693 | −0.0539 | +0.0216 |
| F3c given F0+F2a | −0.2669 | −0.0371 | −0.1896 | +0.0366 |

For the two single-addition arms, F2a is correct and F3c wrong on 16
Session 2 trials versus four in the reverse direction; Session 3 reverses
to seven versus 13. The two-family negative-log-loss interaction is
+0.2564/+0.1441 on Sessions 2/3, but the joint model still has higher
absolute log loss than F0 in both. An interaction sign is not a license
to promote the full bank.

The [runner](manus_spatial_run.py) stores all [864 whole-trial probability
rows](MANUS_SPATIAL_PREDICTIONS.csv) and the [results](MANUS_SPATIAL_RESULTS.json).
The [independent verifier](verify_manus_spatial.py) checks native user,
session, speed, gesture and trial identity, disjoint source/target trials,
complete 6×3 user–speed cells, probability validity, all 80 pooled/user/speed
score groups and the validation selection. Its [audit](MANUS_SPATIAL_VERIFICATION.json)
passes. The [paired reproducer](manus_spatial_paired.py) exactly replays
those 80 score groups and reconstructs [80 family](MANUS_SPATIAL_SCREEN.csv),
[40 conditional](MANUS_SPATIAL_CONDITIONAL.csv),
[20 error](MANUS_SPATIAL_COMPLEMENTARITY.csv) and
[20 interaction](MANUS_SPATIAL_INTERACTION.csv) rows byte for byte.
The [delivery audit](MANUS_SPATIAL_DELIVERY_AUDIT.json) binds the first
three tables to canonical results; interaction remains standalone. The
canonical source-bound audit now covers 53,548 records.

Session 3 had been inspected in earlier project studies, so its numbers
are descriptive confirmation of this frozen comparison, not a fresh
project-wide holdout. The task has no Rest, open-hand or pinch class, and
its public device cannot establish performance on the user's 250 Hz
hardware or continuous live recognition.
