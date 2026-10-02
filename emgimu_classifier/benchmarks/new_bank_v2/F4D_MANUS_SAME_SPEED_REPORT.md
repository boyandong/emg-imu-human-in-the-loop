# F4d same-speed public MANUS control

The [versioned protocol](F4D_MANUS_SAME_SPEED_PROTOCOL.json) removes the speed mismatch in the previous [predictive screen](F4D_MANUS_PREDICTIVE_REPORT.md). Each of six users retains the same 18 session-1 source trials and frozen classifiers. For each slow or fast trial in session 2 or 3, the other five gesture trials from **that user, session and speed** set an equal-trial-weighted F4d reference. The excluded native trial alone is evaluated. No target label fits a classifier or reference, and no evaluation windows enter calibration. Session 2 is validation; previously inspected session 3 is descriptive.

The [saved results](F4D_MANUS_SAME_SPEED_RESULTS.json) bind the protocol, parent diagnostic, archive and all 432 [native prediction rows](F4D_MANUS_SAME_SPEED_PREDICTIONS.csv). The [readback test](../../tests/test_f4d_manus_same_speed_delivery.py) verifies five distinct calibration trials, one excluded gesture, matching user/session/speed, exact F0v2 and long-centered prediction parity with the previous screen, and independently recomputed scores.

| Phase | Arm | Pooled macro F1 | Log loss | Minimum user macro F1 |
| --- | --- | ---: | ---: | ---: |
| Validation, session 2 | F0v2 | .3898 | 2.2394 | .2421 |
| Validation, session 2 | F0v2 + F4d long | .3912 | 2.3757 | .2407 |
| Validation, session 2 | F0v2 + F4d same-speed | **.4852** | **1.7187** | **.3000** |
| Final, session 3 | F0v2 | .4471 | 2.7249 | .1111 |
| Final, session 3 | F0v2 + F4d long | .4131 | 2.9594 | .1389 |
| Final, session 3 | F0v2 + F4d same-speed | **.4659** | **2.1554** | .0952 |

Same-speed session subtraction improves validation F1 by .0954 and log loss by .5207 against F0v2, and its final pooled F1 by .0188. The final minimum-user F1 still declines from .1111 to .0952. This control rules out the *calibration/evaluation speed mismatch* as a necessary explanation for the pooled improvement. It does not isolate a pure session effect: each held trial's reference lacks that gesture, whereas the source long reference contains all six, and choosing the excluded trial identity is an offline leave-one-out procedure. This is not a deployable five-gesture calibration protocol, independent validation cohort, measured fatigue effect, or own-device result. F4d remains a bounded candidate rather than the default.
