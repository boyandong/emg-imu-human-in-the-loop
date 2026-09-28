# Qualified new-v2 MANUS speed study with external source Rest

The [frozen protocol](MANUS_REST_TRANSFER_PROTOCOL.json) uses the same
verified [sEMG-MANUS](https://zenodo.org/records/19261324) eight-channel,
nominal 200 Hz recordings as the earlier
[reduced-bank study](MANUS_SPATIAL_REPORT.md): users 3–8, six finger
flexion–extension classes, slow/medium/fast trials in Sessions 1–3.
MANUS has **no labelled Rest class**. To evaluate the exact independent
F0v2/F2a/F3c four-arm feature formulas, the
[reproducer](manus_rest_transfer_run.py) fits F0v2's noise threshold on
1,828 labelled Rest windows from *source subjects 1–18* in the separate
[ROAM-EMG posture protocol](ROAM_POSTURE_REPORT.md). Those public Myo
windows match eight channels, 200 Hz and 40-sample windows. No ROAM
validation/final user and no MANUS Session-2/3 signal fits any feature
state. All classifiers/scalers train only on MANUS Session 1. This is a
**cross-dataset Rest prior**, not a same-dataset Rest calibration.

Each native MANUS recording contributes one aggregate of up to eight
disjoint 200 ms windows. The same 108 Session-2 recordings validate all
arms; 108 Session-3 recordings provide descriptive final results.

| Arm | Session 2 F1 | Session 2 loss | Session 2 worst-speed F1 | Session 3 F1 | Session 3 loss | Session 3 worst-speed F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0v2 | 0.3914 | **2.1203** | 0.2845 | 0.4600 | 1.7231 | 0.4283 |
| F0v2+F2a | **0.5244** | 2.3949 | **0.4677** | **0.5304** | **1.7124** | 0.4491 |
| F0v2+F3c | 0.3892 | 2.8912 | 0.2509 | 0.5251 | 1.9380 | **0.4734** |
| F0v2+F2a+F3c | 0.4742 | 2.6039 | 0.4023 | 0.5077 | 1.8137 | 0.4269 |

The prespecified Session-2 macro-F1 rule selects F0v2+F2a, which also
has the highest validation minimum-speed F1. Its validation log loss is
worse than F0v2 by 0.2746, so the result is metric-dependent. The
already-inspected Session-3 ranking is descriptive. Slow is the weakest
speed stratum for F0v2 and the selected arm on Session 3. Comparing the
old reduced-bank F0 and this external-Rest F0v2 is **not an isolated
effect of Rest transfer** because the baseline feature formulas differ.

The [864 saved arm–trial probability rows](MANUS_REST_TRANSFER_PREDICTIONS.csv)
are bound by the [results](MANUS_REST_TRANSFER_RESULTS.json). The
[matched analysis](manus_rest_transfer_paired.py) checks native user,
session, speed, gesture and trial identity, replays 80 saved pooled/user/
speed score groups, and evaluates 56 pooled, user, speed and
user-by-speed cells with identical native trial IDs across arms. Its
[audit](MANUS_REST_TRANSFER_PAIRED_AUDIT.json) binds 224 family,
112 conditional, 56 error-overlap and 56 interaction rows. The
[delivery audit](MANUS_REST_TRANSFER_DELIVERY_AUDIT.json) binds the
first three tables to canonical results.

All three speeds appear in every session. This tests **speed-stratified
performance across sessions**, not transfer to a previously unseen
speed. Source and target sessions use the same six users; Session 3 has
been inspected in prior project work. MANUS has no neutral, open-hand
or pinch target class and cannot validate the user's 250 Hz device or
live transitions. The external Rest prior may have different electrode
placement and noise statistics, so this is a qualified supplement to
the new-v2 robustness vector, not a replacement for same-dataset Rest.
