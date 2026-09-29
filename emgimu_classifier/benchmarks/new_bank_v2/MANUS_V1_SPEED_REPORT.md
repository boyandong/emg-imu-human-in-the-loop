# New-v1 MANUS observed-speed extension

This versioned experiment adds four independently defined new-v1 feature
families to the frozen F0v2 baseline on native MANUS recordings. It uses
Session 1 for model training, Session 2 for validation and arm selection,
and Session 3 only for descriptive final evaluation. Six public users and
all three observed speeds (slow, medium, fast) appear in every session.
There are 108 whole recordings per session, each represented by at most
eight disjoint 200 ms windows from 8 EMG channels sampled at 200 Hz.

MANUS does not label Rest. The F0v2 Rest thresholds therefore come from
exactly 1,828 frozen source-subject ROAM Rest windows, not from target
MANUS recordings. The baseline target probabilities replay the parent
MANUS Rest-transfer experiment exactly (maximum absolute difference 0).
This cross-dataset prior qualifies every result below; it is not evidence
for an unseen speed, the 250 Hz own device, or live operation.

| Arm | Session 2 macro-F1 | Session 2 log loss | Session 3 macro-F1 | Session 3 log loss |
| --- | ---: | ---: | ---: | ---: |
| F0v2 | .3914 | 2.1203 | .4600 | 1.7231 |
| + scale pattern | .3751 | 2.1582 | .5300 | 1.7982 |
| + ring lag | .4264 | 2.7139 | .4136 | 2.0466 |
| + correlation spectrum | .4123 | 2.3134 | .4662 | 1.8414 |
| + frequency direction | .3459 | 2.5400 | .4958 | 2.0851 |

Ring lag is selected by the prespecified Session 2 pooled macro-F1 rule,
but its validation log loss and minimum-speed F1 worsen against F0v2.
On Session 3 it loses .0464 macro-F1 and raises log loss by .3235.
Scale pattern's Session 3 F1 gain is visible only after selection and
comes with worse log loss; it cannot justify a retrospective promotion.
No new-v1 family is promoted as a general speed-robust default.

The saved 1,080 held-target probability rows support 280 family-score,
224 conditional-increment and 560 complete pairwise-error cells across
pooled, user, speed and user-by-speed groups. Source hashes, native trial
identities and 100 saved pooled/user/speed score groups are checked during
analysis. All 1,064 cells are in the canonical feature-bank delivery;
59,506 total canonical records pass the source-provenance verifier.

Reproduce with `manus_v1_speed_run.py`, then `manus_v1_speed_analysis.py`,
then `export_manus_v1_speed.py` using the frozen protocol and public
archives declared there. The final-session data were not used to fit the
source model or choose the arm.
