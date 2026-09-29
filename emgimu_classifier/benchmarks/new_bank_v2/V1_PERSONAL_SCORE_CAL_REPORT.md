# Personal score-anchor calibration for independent new-v1 families

The [protocol](V1_PERSONAL_SCORE_CAL_PROTOCOL.json) and
[runner](v1_personal_score_calibration.py) were committed before outcomes.
It reuses the five frozen F0v2/new-v1 source models and their native trial
probabilities, without refitting a classifier. The user split has Day1 source
users 1–4, validation users 5–6 and descriptive final users 7–8. The
cross-day split has Day1 source, Day2 validation and descriptive Day3 final
for eight repeated users. Both use official GRABMyo eight-channel 2048 Hz
recordings. Each class supplies target-subject native repetitions 1..N for
0/1/2/5 shots; repetitions 6/7 alone are evaluated. Therefore every budget
has the same held-out evaluation trials and no calibration/evaluation overlap.

This is a fixed **four-dimensional score-space** correction: per-class means
of labelled calibration probability vectors become prototypes; squared
distance with fixed temperature 0.1 supplies anchor probabilities, mixed
equally with the frozen source probability. The rule and all budgets come
from an existing public GRAB control; no validation/final result selects its
parameters. Zero shots replay the source classifier exactly. One, two and
five shots per class require 20, 40 and 100 seconds of recorded signal,
respectively, excluding rest and setup.

| Study | Arm | Validation F1 at 0/1/2/5 shots | Descriptive final F1 at 0/1/2/5 |
| --- | --- | --- | --- |
| Unseen user | F0v2 | .7571/.8128/.8128/.8128 | .8667/.8667/.8667/.8667 |
| Unseen user | + ring lag | .8185/.8185/.8185/.8185 | .8667/.8667/.8667/.8667 |
| Cross day | F0v2 | .9687/.9844/.9844/.9687 | .9210/.9686/.9527/.9844 |
| Cross day | + ring lag | .9844/1.0000/.9686/.9844 | .8867/.8863/.8863/.9373 |

These are representative arms; the [curve](V1_PERSONAL_SCORE_CAL_CURVE.csv)
contains every arm, budget, phase, pooled and per-subject cell with F1,
accuracy, log loss, Brier, ECE and per-class F1. Calibration raises the
unseen-user scale-pattern validation F1 from .6290 to .8128 at one shot,
but it merely reaches the calibrated F0v2 F1 and has worse log loss. Ring
lag remains a narrow validation specialist: it has higher F1 than F0v2 at
zero shot and reaches perfect F1 on the 64-trial Day2 cross-day evaluation
subset at one shot, but its descriptive Day3 F1 remains below F0v2 at all
budgets. No added family has a reliable all-split personal-calibration
advantage. The validation subset for unseen users has only 16 trials; per-
subject cells have eight, so these changes are sensitive to a few errors.

The [3,200 saved predictions](V1_PERSONAL_SCORE_CAL_PREDICTIONS.csv) and
[assignment audit](V1_PERSONAL_SCORE_CAL_AUDIT.json) retain all 400 native
subject/arm/phase/budget calibration and evaluation sets. Independent
[read-back verification](V1_PERSONAL_SCORE_CAL_VERIFICATION.json) checks all
3,200 adapted probabilities, 480 score cells and 800 exact zero-shot parent
replays. The [delivery audit](V1_PERSONAL_SCORE_CAL_DELIVERY_AUDIT.json)
binds the 480 canonical calibration-curve records by hash; the canonical
provenance check covers 57,246 records.

The study tests whether independently implemented feature **model scores**
become useful after one fixed personal correction. It does not establish
feature-space Personal Anchor F7, per-channel normalization, target model
refitting, a product setup time, or own-device 250 Hz performance. Reused
public cohorts and previously inspected final phases are descriptive.
