# Seven-axis new-v1 default-bank decision

The [frozen rule](V1_CROSS_AXIS_DECISION_PROTOCOL.json) compares four
independently defined additions with F0v2 on seven available public-data
axes. It uses each experiment's **validation** split only to select a
deployment default. The comparison is within each task: a candidate must
avoid pooled macro-F1 loss and log-loss increase on every axis, with a
strict improvement somewhere. Synthetic quality uses its prespecified
six-coordinate fault-family mean and mean named-fault loss; clean and
minimum named-fault F1 must also not fall. Final splits are shown only
after this choice is made. This is a conservative engineering gate,
not a claim that the public datasets are independent replications.

Validation macro-F1 changes relative to each matched F0v2 baseline:

| Axis | Scale pattern | Ring lag | Correlation spectrum | Frequency direction |
| --- | ---: | ---: | ---: | ---: |
| Posture | -.012 | -.013 | -.024 | -.018 |
| Unseen user | -.057 | +.031 | -.027 | -.005 |
| Cross day | -.028 | +.013 | .000 | -.009 |
| Force intensity | -.009 | -.017 | -.022 | -.009 |
| Wearing shift | -.016 | +.006 | +.035 | -.081 |
| Observed speed | -.016 | +.035 | +.021 | -.045 |
| Synthetic quality | -.002 | -.073 | -.036 | -.032 |

No addition satisfies the seven-axis gate. The saved
[decision audit](V1_CROSS_AXIS_DECISION_AUDIT.json) records 14, 8, 9 and
14 validation guard violations for scale pattern, ring lag, correlation
spectrum and frequency direction respectively, counting F1, log loss
and quality-specific checks. The deployment default **within this finite
new-v1 candidate set remains F0v2**. This is consistent with the earlier
[ring-lag × frequency-direction interaction](RING_FREQ_INTERACTION_REPORT.md)
and [leave-one-family-out study](RING_FREQ_LOFO_REPORT.md): their
combined candidate failed validation and its added families had
negative validation contributions. Conditional specialists can still be
investigated with separate task-specific validation.

The 70-row [matched table](V1_CROSS_AXIS_DECISION_CELLS.csv) contains
validation and descriptive-final F1, loss, baseline changes and the
quality guards for five arms on seven axes. It is regenerated only after
checking hashes of all seven saved result files and their prediction
files. Several final-only gains reverse the validation direction; for
example, ring lag improves unseen-user validation F1 by .031 but lowers
final F1 by .056, while correlation spectrum improves wearing validation
F1 by .035 but lowers final F1 by .073. Those final rows did not enter
selection.

The selected public-data default is a **zero-target-shot F0v2** model.
The existing [source-subject OOF temperature test](V1_SOURCE_OOF_CAL_REPORT.md)
does not support a universal reliability transform: target F0v2 log loss
worsens in five of six ROAM/GRAB groups. The fixed
[score-space personal correction](V1_PERSONAL_SCORE_CAL_REPORT.md) yields
some 1/2/5-shot validation gains but no added-family advantage that holds
through the descriptive final splits. The fixed
[feature-space anchor](V1_FEATURE_ANCHOR_REPORT.md) worsens pooled log
loss for every arm at every nonzero budget on both tested days. These
three calibration controls remain separately available, but none is
silently attached to the selected default. This decision does not rule
out future, independently validated personal calibration methods.

Posture and quality share ROAM recordings; unseen-user and cross-day
reuse GRAB recordings. MANUS speed uses a cross-dataset Rest prior,
quality faults are synthetic, and the tasks have different classes,
rates and populations. Thus raw scores are never averaged across axes.
This decision does not establish performance on the user's 250 Hz
device, multi-person own-device generalization, or physical fault
robustness. Reproduce with `v1_cross_axis_decision.py` from the frozen
public experiment artifacts.
