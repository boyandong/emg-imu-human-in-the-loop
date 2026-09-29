# Feature-space personal anchors for independent new-v1 families

The [frozen protocol](V1_FEATURE_ANCHOR_PROTOCOL.json) was committed before
outcomes. It extends the exact same five source-trained F0v2/new-v1 arms on
official GRABMyo eight-channel 2048 Hz recordings. Day1 alone fits the F0v2
Rest thresholds, each arm's scaler and its classifier. Day2 is validation;
Day3 is a descriptive final phase because it had been inspected previously.
All eight subjects appear on all three days. Source-model predictions replay
the saved parent probabilities **exactly** for every arm and target trial.

For each target subject and day, native repetitions 1..N of every class
form 0/1/2/5-shot calibration sets. Repetitions 6/7 always evaluate, so
budgets share the same 64-trial held-out subset on each day and calibration
never overlaps evaluation. Day1-fitted scalers transform all trial features.
Allowed target examples form one mean standardized feature prototype per
class. The fixed squared-distance softmax temperature is the median positive
inter-prototype distance; a 0.5 mixture with the frozen classifier probability
is used for every arm and nonzero budget. This rule copies the earlier
new-v2 feature-anchor control without tuning to these new-v1 outcomes.
One, two and five shots/class require 20, 40 and 100 seconds of signal,
respectively, excluding setup and rest.

| Arm | Day2 F1 at 0/1/2/5 shots | Day3 F1 at 0/1/2/5 shots |
| --- | --- | --- |
| F0v2 | .9687/.9687/.9687/.9687 | .9210/.9210/.9363/.9210 |
| + scale pattern | .9045/.9375/.9686/.9218 | .9056/.9056/.9222/.9056 |
| + ring lag | .9844/.9686/.9527/.9686 | .8867/.9197/.8867/.9040 |
| + correlation spectrum | .9530/.9844/.9692/.9692 | .8088/.8260/.8254/.8414 |
| + frequency direction | .9228/.9228/.9228/.9228 | .9353/.9353/.9353/.9353 |

The [complete curve](V1_FEATURE_ANCHOR_CURVE.csv) also reports log loss,
Brier, ECE, class F1 and individual subjects. The fixed feature-anchor rule
worsens **pooled log loss at
every nonzero budget for every arm on both days**. For F0v2, Day2 loss
rises from .1464 at zero shots to .4496 at one; Day3 rises from .2349 to
.4870. Some F1 changes are positive, but this severe probability degradation
rejects the tested rule as a default personal-calibration method. It does
not imply that all possible feature-space anchors or personalized models fail.

The [runner](v1_feature_anchor.py) saves the Day1-standardized target
feature matrices, the [2,560 predictions](V1_FEATURE_ANCHOR_PREDICTIONS.csv)
and [320 native assignments](V1_FEATURE_ANCHOR_RESULTS.json). The separate
[read-back verifier](V1_FEATURE_ANCHOR_VERIFICATION.json) recomputes all
anchor probabilities from saved matrices, checks all 360 pooled/per-subject
score cells and 640 exact zero-shot parent replays. All 360 cells are in the
required calibration delivery; canonical provenance covers 57,606 rows.

This study is a fixed public-data feature-space calibration control, not a
claim of 250 Hz own-device recognition, live setup time, or cross-person
generalization. It tests a raw-feature Personal Anchor candidate without
per-channel target normalization or more complex session adaptation.
