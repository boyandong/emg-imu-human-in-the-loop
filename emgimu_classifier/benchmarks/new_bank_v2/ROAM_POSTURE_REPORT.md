# Independent new-v2 eight-channel static-posture study

The [frozen protocol](ROAM_POSTURE_PROTOCOL.json) uses the public ROAM-EMG
archive from the [ReactEMG authors](https://github.com/roamlab/reactemg)
(SHA-256 `c9de0e25c187c216e4771d17ccfa19acb0e4028f907ea726b2a87a18d2db5343`).
Its 28 subjects each have native eight-channel, nominal 200 Hz recordings
for resting, hanging, unsupported and reaching arm postures. Native `gt`
codes 0/1/2 mean relax/open/close. All 112 static recordings contain the
expected nine-label sequence. The [reproducer](roam_posture_run.py) retains
each contiguous label bout as an evaluation unit, excludes 200 ms at both
edges, and aggregates disjoint pure-label 200 ms windows. It retains
25,624 windows and 1,008 bouts; the raw archive is external to Git.

Source subjects 1–18 contribute only the resting posture to training. The
F0v2 noise thresholds use only their 1,828 relax windows. Unseen subjects
19–23 validate the four feature arms in all four postures; unseen subjects
24–28 are descriptive final. All three feature families, scaler and
classifier fit source data only. The fixed validation macro-F1 rule selects
**F0v2**. This is a new independent implementation, not a reconstruction
of any unavailable historical B0/X1 code.

| Arm | Validation F1 | Validation loss | Validation worst posture F1 | Final F1 | Final loss | Final worst posture F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0v2 | **0.9630** | **0.1918** | **0.8723** | 0.9162 | 0.2059 | 0.8452 |
| F0v2+F2a | 0.9167 | 0.2622 | 0.8399 | **0.9353** | **0.1838** | **0.9028** |
| F0v2+F3c | 0.9075 | 0.3193 | 0.8561 | 0.8360 | 0.3501 | 0.7433 |
| F0v2+F2a+F3c | 0.9228 | 0.3292 | 0.8638 | 0.8814 | 0.2740 | 0.8533 |

Hanging is the weakest posture for all but the full arm's final split.
For F0v2, relax/open/close recall is 0.988/0.917/0.975 on validation
and 0.963/0.817/1.000 on final. The weakest individual F0v2
subject-by-posture cell is subject 21 hanging at 0.5758 on validation
and subject 27 hanging at 0.5238 on final. The final-only F2a gain cannot
change the preselected arm; the full bank is not a stable default.

The [saved probabilities](ROAM_POSTURE_PREDICTIONS.csv) contain 1,440
arm–bout rows. The [paired protocol](ROAM_POSTURE_PAIRED_PROTOCOL.json)
and [analysis](roam_posture_paired.py) compare identical native bout IDs
within each phase: pooled, four posture, five subject and twenty
subject-by-posture groups per phase. Its
[audit](ROAM_POSTURE_PAIRED_AUDIT.json) checks 60 paired cells, 40 replays
of saved pooled/posture scores, and source hashes. Family, conditional,
error-overlap and interaction tables are versioned independently; the
[export audit](ROAM_POSTURE_DELIVERY_AUDIT.json) binds canonical rows.

This is posture evidence for the exact F0v2/F2a/F3c candidate bank, but
it tests both unseen users and posture shift relative to resting source
users. Bouts from the same recording are correlated; these are not
independent human trials. The Myo's 200 Hz signals and static labels do
not establish 250 Hz own-device performance or online transition quality.
