# Source-subject OOF probability calibration for independent families

The [protocol](V1_SOURCE_OOF_CAL_PROTOCOL.json) was committed before the
calibration outcome. For each of the five frozen feature arms in ROAM posture,
GRAB unseen-user and GRAB cross-day, each source subject is held out once.
Every fold fits F0v2 Rest thresholds, the scaler and the logistic classifier
using other source subjects only. Candidate-family metadata come from a zero
dummy with the same channel/rate/window configuration, so they contain no
held-subject statistics. A single positive temperature is selected on all
source OOF probabilities from a frozen grid, then applied unchanged to the
previously saved validation and descriptive final probabilities. Target
labels never tune the temperature; target calibration budget is zero.

| Study | Source OOF subjects | F0v2 T | Validation F0v2 Δlog loss | Final F0v2 Δlog loss |
| --- | ---: | ---: | ---: | ---: |
| ROAM posture | 18 | 1.581 | -0.0035 | +0.0080 |
| GRAB unseen user | 4 | 1.843 | -0.0440 | -0.0988 |
| GRAB cross day | 8 | 1.305 | -0.0350 | -0.0169 |

Positive Δlog loss means temperature calibration improved the held-out
target. Source OOF log loss improves for all 15 study/arm fits by the frozen
optimization rule. On validation/final targets it improves log loss in only
10 of 30 arm–phase groups, Brier in 6 and ten-bin ECE in 8. In particular,
F0v2 target log loss worsens in five of six groups. Positive scalar
temperature leaves the class decision and macro-F1 unchanged, as verified.
The target shift is strong enough that this source-only reliability correction
is **not promoted** as a universal default.

After calibration, no added family improves log loss over calibrated F0v2
across all three validation splits. Scale pattern has a small ROAM validation
gain; ring lag a GRAB unseen-user gain; correlation spectrum a GRAB cross-day
gain. Those isolated changes, and source-only zero-shot calibration, do not
establish a calibration-amplifier family or a personal 1/2/5-shot benefit.

The [runner](v1_source_oof_calibration.py) saves 2,490 source OOF plus 4,600
held-out prediction rows in the [prediction table](V1_SOURCE_OOF_CAL_PREDICTIONS.csv).
The [results](V1_SOURCE_OOF_CAL_RESULTS.json) include all 30 source folds,
train/held subject sets, Rest-window counts, selected temperatures and
raw/calibrated scores. The [read-back analysis](v1_source_oof_calibration_analysis.py)
independently recomputes the temperature-grid minimum, frozen target raw
probabilities, calibrated probabilities and 440 pooled, subject and posture
[score cells](V1_SOURCE_OOF_CAL_CELLS.csv). Its
[verification audit](V1_SOURCE_OOF_CAL_VERIFICATION.json) binds those files
by SHA-256. The [idempotent exporter](export_v1_source_oof_calibration.py)
adds both zero-shot methods to the required calibration-curve source table;
the [delivery audit](V1_SOURCE_OOF_CAL_DELIVERY_AUDIT.json) records its hash.
Canonical provenance verification now covers 56,766 records. The
[test](../../tests/test_v1_source_oof_calibration.py) checks subject-fold
exclusion, complete source coverage, grid selection and delivered rows.

The public ROAM/GRAB tasks and rates differ, and the GRAB studies reuse
recordings. This experiment does not estimate personal calibration shots,
own-device setup time or 250 Hz live reliability.
