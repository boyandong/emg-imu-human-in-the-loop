# Independent new-v1 family screen on GRABMyo cross-day transfer

The [protocol](GRAB_DAY_V1_EXTENSION_PROTOCOL.json) was committed before
outcomes. It uses the same 672 official-SHA-256-verified eight-channel
GRABMyo recordings and Day1 source / Day2 validation / Day3 descriptive final
split as the [frozen new-v2 day study](GRABMYO_REPORT.md). All eight subjects
appear on each day, so this is same-person cross-day transfer, not new-user
transfer. Only Day1 Rest windows fit F0v2 thresholds and candidate metadata;
only Day1 recordings fit scaler and classifier. Four independent new-v1
families are added one at a time to the exact F0v2 backbone.

| Arm | Day2 F1 | Day2 loss | Day2 min-user F1 | Day3 F1 | Day3 loss | Day3 min-user F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0v2 | 0.9551 | 0.1507 | 0.8601 | 0.9008 | **0.3007** | **0.6667** |
| + scale pattern | 0.9266 | 0.1515 | 0.7953 | 0.8999 | 0.4437 | 0.6530 |
| + ring lag | **0.9686** | 0.1422 | **0.9271** | 0.8862 | 0.3397 | **0.6667** |
| + correlation spectrum | 0.9551 | **0.1367** | 0.8951 | 0.8363 | 0.4742 | 0.6209 |
| + frequency direction | 0.9464 | 0.1650 | 0.7953 | **0.9089** | 0.4733 | 0.6530 |

The fixed validation pooled-F1 rule selects **F0v2+ring_lag**. Against
F0v2 it corrects six and introduces three Day2 errors, but corrects three
and introduces six Day3 errors. The selected arm falls 0.0146 below the
baseline in Day3 F1, matching the direction of the prior independent-v1
ring screen on these same trials. Correlation spectrum improves validation
log loss but not F1 and sharply worsens both final metrics. Frequency
direction's 0.0081 final F1 edge is final-only and comes with markedly
worse final log loss. No final-only value selects a model.

Together with the [cross-user](GRAB_V1_EXTENSION_REPORT.md) and
[ROAM posture](ROAM_V1_EXTENSION_REPORT.md) screens, ring lag has positive
validation effects in two GRAB splits and a ROAM worst-posture effect, but
negative pooled final GRAB effects. These tests share some GRAB recordings
and are not independent replications. Further source-predeclared validation
is necessary before any promotion; the live bundle remains unchanged.

The [result](GRAB_DAY_V1_EXTENSION_RESULTS.json) binds the official checksum
manifest, protocol and 2,240 [saved target probabilities](GRAB_DAY_V1_EXTENSION_PREDICTIONS.csv).
The 448 F0v2 parent probabilities replay with maximum absolute difference
`1.62e-9`; pooled scores match. The tolerance is `1e-8` for numerical
floating-point reproduction and is checked by the
[read-back test](../../tests/test_v1_extension_delivery.py), which also
recomputes all arm/day metrics from saved probabilities and checks native
trial IDs and source/target separation. The public 2048 Hz isolated trials
do not establish recognition on the user's 250 Hz device.
