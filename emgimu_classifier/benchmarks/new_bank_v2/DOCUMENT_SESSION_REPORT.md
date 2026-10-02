# Document-exact normalization in the session pipeline

The [versioned protocol](DOCUMENT_SESSION_PROTOCOL.json) compares the frozen `max(Q95, ε)` denominator with an opt-in `Q95 + ε` session pipeline on the same six public LibEMG Electrode Shift users. Each user's 25 before-wearing native trials fit a separate long-term model for each formula. Within each of four after-wearing domains, five `R_0` trials calibrate a session and five disjoint `R_1` trials are evaluated without passing their labels to prediction. The prior subject-15/trial-1 legacy package replays exactly.

| Phase | User/domain blocks | Paired provider prediction rows | Maximum probability difference | Changed predicted gestures |
| --- | ---: | ---: | ---: | ---: |
| Validation, users 15–17 | 12 | 720 | 0 | 0 |
| Descriptive final, users 18–20 | 12 | 720 | 0 | 0 |

The smallest source active Q95 is 13 and the smallest session Q95 is 4 in the archive's units. Both formulas produce identical saved probabilities after the current float32 feature transforms on this inspected native subset. The [result](DOCUMENT_SESSION_RESULTS.json) retains all 24 source/calibration/evaluation identity blocks, model identities, protocol and [1,440 paired prediction rows](DOCUMENT_SESSION_PREDICTIONS.csv). A readback test recomputes pairwise equality, class/order coverage and frozen-parent parity; two fresh-process runs produce the same result hash.

This closes a software integration gap for the exact calibration formula. It does not show better classification, equivalence on near-zero or faulty channels, a new independent holdout, verified physical re-donning or the user's 250 Hz hardware. The legacy pipeline stays frozen for reproducibility; the exact path is available separately and is not promoted on these equal outcomes.
