# New-v2 GRABMyo same-user score-anchor calibration

This is a bounded personal-calibration follow-up using the already-frozen
new-v2 GRABMyo class-probability predictions. It builds class prototypes in
**four-dimensional probability space**, not in raw EMG or the F7 feature
space. The [protocol](GRAB_SCORE_CAL_PROTOCOL.json) and
[runner](grab_score_calibration.py) were committed before results were
calculated. They fix a softmax temperature of 0.1 and equal 0.5 weights for
the original source probability and the anchor probability; neither
parameter is tuned on Day2 or Day3. No classifier is retrained.

For each subject and each of Day2 validation and Day3 descriptive final,
the first 0, 1, 2 or 5 **native recordings per class** provide the labelled
calibration set. Every budget is evaluated on the same repetitions 6 and 7,
so calibration and evaluation trials never overlap. Each recording is 5 s;
1, 2 and 5 shots/class require 4, 8 and 20 recordings, or about 20, 40 and
100 s of recorded signal per user and session, excluding instructions and
rests. The 0-shot probabilities are an exact replay of the source classifier
on the fixed evaluation subset, not a separate model.

| Arm | Day2 macro-F1 at 0/1/2/5 shots | Day3 macro-F1 at 0/1/2/5 shots |
| --- | --- | --- |
| F0v2 | 0.9687 / 0.9844 / 0.9844 / 0.9687 | 0.9210 / 0.9686 / 0.9527 / 0.9844 |
| F0v2+F2a | 0.9364 / 0.9364 / 0.9364 / 0.9364 | 0.8567 / 0.8719 / 0.9051 / 0.9063 |
| F0v2+F3c | 0.9222 / 0.9375 / 0.9375 / 0.9375 | 0.8892 / 0.9050 / 0.9050 / 0.8892 |
| F0v2+F2a+F3c | 0.9364 / 0.9526 / 0.9688 / 0.9688 | 0.8719 / 0.9047 / 0.9197 / 0.9047 |

The score-space correction can recover some same-user cross-day errors.
For F0v2 on the fixed Day3 subset, 1-shot raises macro-F1 from 0.9210 to
0.9686 and lowers log loss from 0.2349 to 0.1556. Five shots raise F1 to
0.9844 and lower log loss to 0.1307. More labelled recordings do **not**
always improve the other arms: the F3c arm's Day3 F1 returns to its 0-shot
value at 5 shots, and the full arm peaks at 2 shots. This is compatible
with small evaluation sets and a fixed mixture that was not optimized for
each arm; it is not evidence for a universal five-shot prescription.

The [curve](GRAB_SCORE_CAL_CURVE.csv) includes pooled and per-subject F1,
accuracy, log loss, Brier, ten-bin ECE, and per-class F1. The
[2,048 prediction rows](GRAB_SCORE_CAL_PREDICTIONS.csv) and
[assignment audit](GRAB_SCORE_CAL_AUDIT.json) retain exact native trial IDs
for 256 arm–phase–subject–budget assignments, including 512 exact zero-shot
source replays. Running `grab_score_calibration.py --verify` regenerates
both tables and the audit byte for byte. The idempotent exporter adds 288
pooled and subject records to the canonical calibration curve; the
[delivery audit](GRAB_SCORE_CAL_DELIVERY_AUDIT.json) records source hashes.

The eight-user, four-gesture, 64-trial-per-phase evaluation subset is small;
each subject has only eight evaluation trials. Day3 has already been inspected
in other project studies and is descriptive here. The public 2048 Hz,
different-device result neither proves 250 Hz own-device live recognition
nor replaces feature-space personal anchors, per-channel normalization,
or genuine session-signature adaptation.
