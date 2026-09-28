# Single-reference-condition force calibration: cost versus transfer

The [protocol](FORCE_REF_CAL_PROTOCOL.json) and [runner](force_reference_calibration.py)
were frozen in commit `b285c49` before outcomes. It tests whether a short
guided Medium-intensity calibration can replace the prior study's separate
calibration at every target intensity. Source feature state, scaler,
classifier, temperature, fixed 0.5 probability mixture, arms and all native
evaluation trials are identical to the [per-condition study](FORCE_CAL_REPORT.md).
Only the target-prototype source changes: one or two labelled Medium trials
per gesture and user, reused at all ten evaluated intensity conditions.
The 0-shot predictions exactly replay the matched parent within `1e-12`.

For F0v2, the parent validation-selected arm:

| Phase | Method | Calibration trials/user | Recorded signal/user | Macro-F1 | Log loss | Worst-condition F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0-shot | 0 | 0 s | 0.6155 | 1.4218 | 0.1926 |
| Validation | Medium 1-shot | 7 | 21.1 s | 0.5985 | 1.0893 | 0.1448 |
| Validation | Medium 2-shot | 14 | 42.3 s | 0.6778 | 1.0545 | 0.1778 |
| Validation | Per-condition 1-shot | 70 | 211.2 s | 0.7189 | 0.7755 | 0.3458 |
| Validation | Per-condition 2-shot | 140 | 422–423 s | 0.7264 | 0.7355 | 0.3186 |
| Final | 0-shot | 0 | 0 s | 0.5187 | 1.8518 | 0.3320 |
| Final | Medium 1-shot | 7 | 21.0–21.1 s | 0.5781 | 1.2770 | 0.4643 |
| Final | Medium 2-shot | 14 | 42.2–42.4 s | 0.5794 | 1.0993 | 0.4086 |
| Final | Per-condition 1-shot | 70 | 211.5–211.6 s | 0.7254 | 0.7729 | 0.5974 |
| Final | Per-condition 2-shot | 140 | 423.2–423.4 s | 0.7369 | 0.7312 | 0.6718 |

The single-reference protocol cuts labelled trial count and recorded signal
time by about 10×, but loses much of the recognition gain. Its validation
one-shot F1 and worst-condition F1 are below zero-shot despite better log
loss. On final users, one/two-shot calibration corrects 17/18 zero-shot
errors and creates three new errors at each budget. Two shots improve pooled
F1 only 0.0013 beyond one shot, while final minimum-subject F1 falls from
0.4259 to 0.3860. The Medium-condition scores match the per-condition method
because its calibration examples are identical there; transfer gaps occur on
other intensities. For example, final 80P F1 is 0.408 lower than per-condition
two-shot F1. This supports a substantial intensity-specific adaptation cost,
not a universal short-calibration solution.

The fixed diagnostic F0v2+F2a arm reaches final F1 0.5803/0.5865 with
one/two Medium shots, compared with 0.6149/0.6300 under per-condition
calibration. Its final-only zero-shot ranking did not select it for this
experiment. The [independent verifier](verify_force_reference_calibration.py)
checks 3,360 saved prediction rows, four nested user assignments, all trial
identities, ten-condition matched evaluation, source/calibration/evaluation
separation, the 0-shot parent replay, all 12 pooled and subject/condition
score groups, and the raw-file durations in the
[audit](FORCE_REF_CAL_VERIFICATION.json).
The [idempotent exporter](export_force_ref_cal_delivery.py) adds 156 matched
pooled, subject and condition records to the canonical calibration curve;
its [delivery audit](FORCE_REF_CAL_DELIVERY_AUDIT.json) binds source hashes.
Canonical provenance verification now checks all 51,580 source records and
recorded values.

The duration figures count only recorded EMG samples at 1000 Hz. Gesture
instructions, transitions, setup, label confirmation and real hardware
latency are excluded. This public-device, trial-level experiment does not
establish the user's 250 Hz device performance, later-day persistence or
unsegmented live recognition. Final users are descriptive because earlier
project studies had examined them.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.` using
`python -m benchmarks.new_bank_v2.force_reference_calibration` followed by
`python -m benchmarks.new_bank_v2.verify_force_reference_calibration`.
