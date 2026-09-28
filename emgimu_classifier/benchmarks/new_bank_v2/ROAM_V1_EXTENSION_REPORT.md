# First-stage ROAM screen of four independent feature families

The [protocol](ROAM_V1_EXTENSION_PROTOCOL.json) was committed before the
outcomes. It holds the native ROAM-EMG source/validation/final subjects,
postures, 200 ms windows, bout aggregation, and source-only classifier fitting
fixed to the [posture study](ROAM_POSTURE_REPORT.md). The independent
`new_bank_v1.py` definitions add scale pattern, ring lag, correlation spectrum,
or frequency direction to the exact F0v2 backbone, one at a time. These are
versioned new formulas; their names do not assert equivalence to unavailable
historical X1/RLCS/CES/Frequency code.

| Arm | Validation F1 | Validation loss | Validation worst posture F1 | Final F1 | Final loss | Final worst posture F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0v2 | **0.9630** | **0.1918** | 0.8723 | 0.9162 | 0.2059 | 0.8452 |
| + scale pattern | 0.9509 | 0.1983 | 0.8723 | 0.9452 | 0.1433 | 0.8971 |
| + ring lag | 0.9500 | 0.1920 | **0.8963** | 0.9359 | 0.1738 | **0.9071** |
| + correlation spectrum | 0.9387 | 0.2199 | 0.8495 | 0.8748 | 0.2602 | 0.8308 |
| + frequency direction | 0.9448 | 0.2125 | 0.8495 | **0.9460** | **0.1194** | 0.9028 |

The frozen validation pooled-F1 criterion still selects F0v2. Ring lag
improves the worst validation posture despite lowering pooled F1, so it is a
specialist candidate for a separate, predeclared worst-condition objective.
Scale pattern and frequency direction improve descriptive final scores but
lose on validation; they cannot be promoted from final-only evidence.
Correlation spectrum loses on all three validation criteria here. No family
has demonstrated a general cross-task conditional benefit from this single
posture study.

The [results](ROAM_V1_EXTENSION_RESULTS.json) bind the archive, protocol, and
1,800 [saved bout predictions](ROAM_V1_EXTENSION_PREDICTIONS.csv) by SHA-256.
All 360 F0v2 target probabilities replay the previously frozen posture study
within absolute `1e-10`. The 1,828 Rest calibration windows and 162 source
bouts come only from subjects 1–18 in resting posture. Five validation and
five final subjects remain unseen. Public 200 Hz bout recognition does not
measure online transitions or the user's 250 Hz device.
