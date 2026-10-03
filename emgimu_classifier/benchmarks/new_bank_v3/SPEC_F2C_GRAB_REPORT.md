# Centered V3 F2c native cross-day increment

The [separate frozen protocol](SPEC_F2C_GRAB_PROTOCOL.json) adds the goal-file
F2c tangent feature to the same checksum-verified, eight-subject GRABMyo
three-day split used by the centered V3 F2a/F3c experiment. F2c takes the
centered F2a covariance, adds a declared minimal positive ridge for singular
windows, fits its log-Euclidean reference on all 4,480 **Day1 training** windows,
and never refits on Day2 or Day3. Its per-trial 72 coordinates are appended to
the frozen F0 extraction. All 448 held-out trial IDs match the parent F0
probability table; individual F2c probabilities are [saved](SPEC_F2C_GRAB_PREDICTIONS.csv)
and independently read back for class counts, probability sums, F1, log loss,
recall and paired errors.

Numerical revision (2026-10-03): the SPD ridge is fixed at `1e-10`, matching
the matrix eigensystem floor. The previous `1e-12` ridge made an all-zero
source window map to a nonzero tangent at its own fitted reference. The saved
predictions below were rerun with the corrected ridge and unchanged trials,
classifier and decision rule; the prior bytes remain in Git commit `1f1a2bf`.

| Arm | Day2 macro-F1 | Day2 log loss | Day3 macro-F1 | Day3 log loss |
| --- | ---: | ---: | ---: | ---: |
| F0 | 0.9551 | 0.1507 | 0.9008 | 0.3007 |
| F0 + centered spec F2c | 0.9687 | 0.0990 | 0.8814 | 0.3288 |

On Day2, F2c corrects six F0 mistakes and creates three; on Day3 it corrects
seven and creates eleven. Thus its validation improvement reverses on the
descriptive final day. The class and subject details remain in the saved
[result](SPEC_F2C_GRAB_RESULTS.json). These public subjects and days have been
inspected in earlier studies; the result is a formula-correct native increment,
not an independent prospective confirmation or evidence for promoting F2c as
a universal cross-day default. It does not validate the user's 250 Hz device.
