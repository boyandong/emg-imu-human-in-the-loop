# New-v2 same-day GRABMyo cross-user axis

The [protocol](GRAB_USER_PROTOCOL.json) was committed before these outcomes
were computed. The public GRABMyo eight-forearm-channel subset contains four
gestures including Rest, seven native recordings per class, and 2048 Hz EMG.
Only Day1 is used here: subjects 1–4 train source feature state,
standardization and classifier coefficients; subjects 5–6 select an arm;
subjects 7–8 give descriptive final scores. All three user groups are
disjoint. The 448 selected Day1 WFDB files match the publisher's SHA-256
manifest. F0v2 noise thresholds come from the 560 source-user Rest windows
alone. No target-user signal fits any representation or model state.

| Arm | Validation macro-F1 | Validation log loss | Final macro-F1 | Final log loss | Final minimum-user F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F0v2 | **0.8054** | **0.4650** | 0.9458 | **0.1994** | 0.8877 |
| F0v2+F2a | 0.7485 | 0.6442 | 0.8899 | 0.3918 | 0.7594 |
| F0v2+F3c | 0.7350 | 0.8313 | **0.9462** | 0.2357 | **0.8912** |
| F0v2+F2a+F3c | 0.7256 | 0.8315 | 0.8899 | 0.3298 | 0.7594 |

The prespecified validation rule selects **F0v2**, which wins validation
macro-F1 and log loss. F3c's final macro-F1 edge over F0v2 is about 0.0004
on two previously inspected users; its final log loss is worse and its
validation macro-F1 is 0.0704 lower. This descriptive final edge cannot
change the selection. Compared on the same native validation recordings,
F2a creates six F0v2 errors and corrects three; F3c creates five and
corrects one. On final users F3c creates one and corrects one.

The [read-back verifier](verify_grab_user.py) checks the official manifest
binding, each native Day1 recording identity and gesture, source/target
user disjointness, probability validity, all eight pooled score groups,
the fixed selection and paired correctness counts. The
[verification audit](GRAB_USER_VERIFICATION.json) passes for all
[448 saved arm–record probability rows](GRAB_USER_PREDICTIONS.csv).
The source [result](GRAB_USER_RESULTS.json) retains all source and target
record IDs and exact scores.

The [matched four-arm analysis](grab_user_paired.py) replays 24 pooled/user
score groups and produces [24 family cells](GRAB_USER_SCREEN.csv),
[12 conditional cells](GRAB_USER_CONDITIONAL.csv),
[six error cells](GRAB_USER_COMPLEMENTARITY.csv), and
[six finite interactions](GRAB_USER_INTERACTION.csv). For F2a added to
F0v2+F3c, conditional Δlog loss is −0.0002/−0.0942 on validation/final;
for F3c added to F0v2+F2a it is −0.1872/+0.0619. The final-only latter
benefit does not outweigh its validation loss or establish absolute gain
against F0v2. The interaction in negative log loss is +0.1790/+0.0982,
yet the joint arm's macro-F1 is 0.0798/0.0559 below F0v2. Positive
interaction alone is not evidence to promote the joint bank.

The [four-axis envelope](ENVELOPE_USER_REPORT.md) incorporates this new
user axis without modifying the frozen three-axis result. The
[canonical delivery audit](GRAB_USER_DELIVERY_AUDIT.json) binds 24 family,
12 conditional and six error rows; interaction remains standalone. All
53,590 canonical source records pass provenance verification.

This is a same-day public-device four-class test. It does not cover
posture, speed, measured hardware quality, real-time transitions or the
user's 250 Hz hardware. The final users were previously inspected during
other project studies and are descriptive, not a new project-wide holdout.
