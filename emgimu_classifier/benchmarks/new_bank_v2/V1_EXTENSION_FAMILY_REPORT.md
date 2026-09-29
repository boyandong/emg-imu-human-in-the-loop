# Three-study independent feature-family screen in canonical delivery

The [read-back](v1_extension_family_screen.py) recomputes F1, accuracy,
log loss, Brier, ten-bin ECE and per-class F1 from the frozen five-arm
held-out probabilities in the [ROAM posture](ROAM_V1_EXTENSION_REPORT.md),
[GRAB unseen-user](GRAB_V1_EXTENSION_REPORT.md), and
[GRAB cross-day](GRAB_DAY_V1_EXTENSION_REPORT.md) experiments. It also
checks that every pooled metric replays the original result. No model is
refitted, and final labels do not alter selection.

| Validation split | F0v2 | + scale pattern | + ring lag | + correlation spectrum | + frequency direction |
| --- | ---: | ---: | ---: | ---: | ---: |
| ROAM posture F1 | **0.9630** | 0.9509 | 0.9500 | 0.9387 | 0.9448 |
| GRAB unseen-user F1 | 0.8054 | 0.7481 | **0.8360** | 0.7784 | 0.8002 |
| GRAB cross-day F1 | 0.9551 | 0.9266 | **0.9686** | 0.9551 | 0.9464 |

Ring lag has an explicit validation benefit on two correlated GRAB splits
but loses pooled ROAM validation F1. Both GRAB final splits reverse its
direction, as shown by the [matched conditional/error analysis](V1_EXTENSION_PAIRED_REPORT.md).
Scale pattern loses validation F1 on all three splits. Correlation spectrum
ties GRAB cross-day validation F1 while improving log loss there, but it
loses validation F1 on the other two splits and has large final-day losses.
Frequency direction raises a ROAM/GRAB worst-condition score in some cells
but not pooled validation F1. No single addition qualifies as a stable
default bank component from these three screens.

The [220-row screen](V1_EXTENSION_FAMILY_SCREEN.csv) contains 100 ROAM,
30 GRAB unseen-user and 90 GRAB cross-day pooled, subject and posture
cells. Its [audit](V1_EXTENSION_FAMILY_AUDIT.json) binds all three saved
prediction/result sources. The [idempotent exporter](export_v1_extension_family.py)
adds all 220 cells to the required feature-family results source table;
the [delivery audit](V1_EXTENSION_FAMILY_DELIVERY_AUDIT.json) records its
hash. Canonical provenance verification now covers 56,326 records. The
[test](../../tests/test_v1_extension_delivery.py) verifies group identities,
class score schema, calibration range and source/export hashes.

ROAM uses nominal 200 Hz Myo data and GRAB uses 2048 Hz forearm recordings;
they have different populations and class problems. The two GRAB studies
reuse recordings. These three rows of evidence are not independent
replications and do not validate the user's 250 Hz online system.
