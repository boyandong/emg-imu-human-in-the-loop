# Fresh EPN612 user confirmation of affine-SPD F7

Protocol and script were pushed as Git commit `c886390` **before** reading
users 22–31. The source Core was fitted on users 1–15, selected on users
16–18, and retrospectively checked on users 19–21. This run is the first
project readout of the separate users 22–31 under a fixed protocol. The
source validation predictions replayed with maximum probability error 0.
No Core parameters, F7 formula, temperature rule or mixture weight were fit
to this cohort. Each target user supplies disjoint 1/2/5 labeled trials per
class for the F7 prototypes; Core remains zero-target-fit. All remaining
complete trials are scored by Core and the fixed `0.5 Core + 0.5 F7` mixture.

| Shots/class | Held-out trials | Core macro-F1 | Mix macro-F1 | Core log loss | Mix log loss | Δ log loss over uniform control |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,440 | 0.4145 | 0.4306 | 1.6505 | 1.4428 | +0.0669 |
| 2 | 1,380 | 0.4128 | 0.4340 | 1.6483 | 1.4258 | +0.0836 |
| **5, primary** | **1,200** | **0.4180** | **0.4420** | **1.6277** | **1.4041** | **+0.0982** |

The five-shot primary conjunction passes all five predeclared criteria:
pooled log loss falls by 0.2236, macro-F1 rises by 0.0240, Brier falls by
0.0109, log loss beats the uniform-softening control by 0.0982, and 7 of
10 subjects improve in log loss. Users 24, 25 and 26 do **not** improve in
log loss; user 26 misses by about 0.00009. All ten user-level macro-F1 scores
rise, though that was not a primary criterion. At five shots, F7 alone is
right where Core is wrong on 307/1,200 trials, while Core alone is right
where F7 is wrong on 208/1,200. This explains a bounded complementarity;
it does not imply F7 always wins.

The [result bundle](F7_AFFINE_FRESH/results.json) contains all 4,020
trial-probability rows across budgets, calibration trial IDs, four-arm
per-user and pooled scores, classwise recall, input and output hashes, and
the predeclared decision. The saved probabilities, trial isolation and
primary metrics are independently read back in
`tests/test_f7_affine_fresh_delivery.py`.

This is stronger public-data evidence for the optional F7 late-fusion arm,
but it does not establish the user's eight-channel electrode montage,
live segmentation, device quality gating or clinical reliability. Core
and F7 are evaluated only on EPN612's six gesture classes. No own-device
default is changed by this public result.
