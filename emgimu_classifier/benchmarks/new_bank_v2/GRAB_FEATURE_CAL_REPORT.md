# New-v2 GRABMyo feature-space personal anchors

This independent follow-up tests actual new-v2 **EMG feature-space** class
prototypes on the public GRABMyo Day2/Day3 recordings. It is distinct from
the earlier [probability-space anchor](GRAB_SCORE_CAL_REPORT.md). The
[protocol](GRAB_FEATURE_CAL_PROTOCOL.json) and [runner](grab_feature_calibration.py)
were committed before outcomes were calculated. The source Day1 recordings
alone fit F0v2 rest thresholds, all family metadata, the per-arm
StandardScaler and the population logistic models. The feature matrix is
the mean and standard deviation of each family over twenty disjoint 250 ms
windows per five-second native recording. Resulting dimensions are 96 for
F0v2, 72 for F2a and 48 for F3c; the full concatenation has 216.

For each target user and day, native repetitions 1..N per gesture make a
feature-space mean prototype at N=1, 2 or 5. Source-standardized Euclidean
distances to the four prototypes become a probability with a temperature
derived **only from the calibration prototypes**. The protocol fixes a
50/50 mixture with the frozen source probability. N=0 exactly replays the
source model. All budgets evaluate the identical native repetitions 6 and
7, giving 64 evaluation recordings per day. The 1/2/5-shot burdens are
4/8/20 labelled recordings, about 20/40/100 seconds of recorded signal
per user and session, excluding guidance and rest.

| Arm | Day2 macro-F1 at 0/1/2/5 shots | Day3 macro-F1 at 0/1/2/5 shots |
| --- | --- | --- |
| F0v2 | 0.9687 / 0.9687 / 0.9687 / 0.9687 | 0.9210 / 0.9210 / 0.9363 / 0.9210 |
| F0v2+F2a | 0.9364 / 0.9364 / 0.9209 / 0.9364 | 0.8567 / 0.8882 / 0.8882 / 0.8882 |
| F0v2+F3c | 0.9222 / 0.9222 / 0.9222 / 0.9222 | 0.8892 / 0.9209 / 0.8892 / 0.9047 |
| F0v2+F2a+F3c | 0.9364 / 0.9531 / 0.9531 / 0.9531 | 0.8719 / 0.8894 / 0.8894 / 0.8894 |

The fixed feature-anchor mixture is **not a viable calibrated default**.
Its Day2 log loss worsens for all four arms at every nonzero budget. For
F0v2, Day3 1-shot F1 remains 0.9210 but log loss worsens from 0.2349 to
0.4870; five shots still give 0.9210 F1 and 0.4684 log loss. Some arms
gain a little F1 on Day3, but their probability quality remains weak.
Under the same held-out trials, the earlier *score-space* F0v2 1-shot
method reaches 0.9686 F1 and 0.1556 log loss. This contrast suggests the
raw high-dimensional source-standardized distance is a poor fixed
probability expert here; it does not establish a general superiority of
score-space calibration across devices or tasks. A different anchor metric
or mixture must be selected on development data before any deployment use.

The [curve](GRAB_FEATURE_CAL_CURVE.csv) contains pooled and per-subject F1,
accuracy, log loss, Brier, ECE and per-class F1. The
[2,048 trial probabilities](GRAB_FEATURE_CAL_PREDICTIONS.csv) include 512
exact zero-shot source replays. The [audit](GRAB_FEATURE_CAL_AUDIT.json)
records the official raw-file checksum manifest, protocol/prediction hashes,
feature dimensions and source-model replay diagnostics. Fresh source fits
have identical top-1 labels on all 1,792 frozen target arm-recordings but
are **not byte-identical** in probability: the maximum difference is
2.72×10⁻⁵ for F0v2+F3c. Frozen source probabilities, not the fresh fit,
are used in every calibrated mixture. The original 10⁻¹⁰ replay tolerance
was relaxed after a numerical mismatch appeared, before reporting results;
the complete diagnostic remains visible in the protocol and audit.
`grab_feature_calibration.py --verify` rereads official-checksummed raw
recordings and rebuilds every published output. The idempotent exporter
adds 288 canonical calibration-curve rows, with a separate
[delivery audit](GRAB_FEATURE_CAL_DELIVERY_AUDIT.json).

Day3 was previously inspected in project development and is descriptive.
The eight selected users, four gestures and 64 fixed evaluation trials per
day cannot prove benefit for a new cohort. The experiment does not yet test
personal raw-channel normalization or a session-signature model and says
nothing about the user's 250 Hz hardware or live continuous recognition.
