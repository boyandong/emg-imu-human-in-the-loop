# F4d public MANUS session correction: predictive control

The [frozen protocol](F4D_MANUS_PREDICTIVE_PROTOCOL.json) follows the
[descriptor study](F4D_MANUS_SESSION_REPORT.md) on six users and three native
sessions. Each user's 18 session-1 trials train source-only classifiers.
External ROAM Rest windows provide the same source Rest prior as the existing
MANUS speed screen. Six medium-speed trials in each later session set only
an equal-trial-mass spectral reference; their gesture labels never fit the
classifier. Twelve different slow/fast trials per user/session are held for
evaluation. Session 2 is validation and session 3 is a descriptive final set.

Three arms use identical source trials and held-out trial identities: F0v2,
F0v2 plus a source-long-centered F4d spectrum, and F0v2 plus a
current-session-centered F4d spectrum. The two spectral arms use the **same
source-trained classifier**; only the target subtraction differs. The
[result](F4D_MANUS_PREDICTIVE_RESULTS.json) binds archive, parent and protocol
hashes to all 432 [saved predictions](F4D_MANUS_PREDICTIVE_PREDICTIONS.csv)
(72 native evaluation trials per phase times three arms). The
[readback test](../../tests/test_f4d_manus_predictive_delivery.py) checks
trial separation, probability normalization and recomputed scores.

| Phase | Arm | Pooled macro F1 | Log loss | Minimum user macro F1 |
| --- | --- | ---: | ---: | ---: |
| Validation, session 2 | F0v2 | 0.3898 | 2.2394 | 0.2421 |
| Validation, session 2 | F0v2 + F4d long | 0.3912 | 2.3757 | 0.2407 |
| Validation, session 2 | F0v2 + F4d session | **0.4764** | **1.7234** | **0.3278** |
| Final, session 3 | F0v2 | 0.4471 | 2.7249 | 0.1111 |
| Final, session 3 | F0v2 + F4d long | 0.4131 | 2.9594 | 0.1389 |
| Final, session 3 | F0v2 + F4d session | **0.4611** | **2.1812** | 0.0952 |

The session-centered arm improves the pooled validation macro F1 by 0.0866
against F0v2 and by 0.0852 against long-centering. Its slow and fast
validation macro F1 are 0.4846 and 0.4702, versus 0.3675 and 0.4003 for
F0v2. The final pooled F1 improvement over F0v2 is only 0.0140, and the
minimum-user F1 **decreases** from 0.1111 to 0.0952. Long-centering alone
does not improve final pooled macro F1.

These observations support F4d as a bounded research candidate, not a
default classifier option. Current references are estimated at medium speed
and scored at slow/fast speed, so speed and session effects cannot be
separated. The cohort has only six previously seen users and one native trial
per gesture/speed/session. The data do not certify separate calendar days,
measured fatigue, new-user transfer, or own-device 250 Hz performance.
