# New v2 eight-channel force personal calibration

The [protocol](FORCE_CAL_PROTOCOL.json) and [runner](force_calibration.py)
were frozen in commit `db546de` before outcomes. This follow-up fixes the
parent validation-selected F0v2 as its primary arm; F0v2+F2a is a diagnostic
comparison, not a final-selected replacement. Six source users fit features,
StandardScaler and a balanced logistic classifier on Ramp trials. Users 7–8
are validation and 9–10 are descriptive final users. All 10 included target
intensity conditions have four native repetitions per gesture and user.
Repetitions 1–2 are available for same-condition personal prototypes;
repetitions 3–4 are the identical evaluation trials for 0, 1 and 2 shots per
class. MVC has only two repetitions and is excluded rather than inventing a
two-shot held-out comparison. No target trial refits the source classifier or
selects a fusion weight. The nonzero-shot prediction always mixes source
logistic and source-scaled target-prototype probabilities at fixed 0.5/0.5.

| Phase | Arm | Shots/class | Macro-F1 | Log loss | Brier | Worst-condition F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validation | F0v2 | 0 | 0.6155 | 1.4218 | 0.5432 | 0.1926 |
| Validation | F0v2 | 1 | 0.7189 | 0.7755 | 0.3573 | **0.3458** |
| Validation | F0v2 | 2 | **0.7264** | **0.7355** | **0.3477** | 0.3186 |
| Validation | F0v2+F2a | 0 | 0.5583 | 2.0156 | 0.6410 | 0.1206 |
| Validation | F0v2+F2a | 1 | **0.6088** | 0.8396 | 0.4097 | 0.1803 |
| Validation | F0v2+F2a | 2 | 0.5862 | **0.8103** | **0.4020** | 0.1803 |
| Final | F0v2 | 0 | 0.5187 | 1.8518 | 0.6677 | 0.3320 |
| Final | F0v2 | 1 | 0.7254 | 0.7729 | 0.3886 | 0.5974 |
| Final | F0v2 | 2 | **0.7369** | **0.7312** | **0.3688** | **0.6718** |
| Final | F0v2+F2a | 0 | 0.5680 | 1.7037 | 0.6314 | 0.3714 |
| Final | F0v2+F2a | 1 | 0.6149 | 0.8507 | 0.4275 | 0.4352 |
| Final | F0v2+F2a | 2 | **0.6300** | **0.8233** | **0.4155** | 0.4352 |

On the matched final F0v2 trials, one shot corrects 54 source-only errors and
creates three; two shots correct 55 and create none. The gain is large but not
uniform: final minimum-subject F1 is 0.3028/0.4836/0.4851 for 0/1/2 shots,
well below pooled F1. Validation worst-condition F1 falls from 0.3458 at one
shot to 0.3186 at two, so more calibration is not monotonic on every axis.
The F2a arm starts above F0v2 on the matched final zero-shot subset but ends
below F0v2 after calibration; this does not change the prespecified primary
arm or authorize retrospective selection from final users.

The [independent verifier](verify_force_calibration.py) checks all 3,360
saved probability rows, 40 user-condition assignments, every source,
calibration and evaluation trial ID, nested 0/1/2-shot allocation, exact
source-only replay of the frozen parent probabilities within `1e-12`, and
all 12 pooled arm–budget score groups plus subject/condition cells. The
[verification audit](FORCE_CAL_VERIFICATION.json) contains paired error counts.
The protocol's expected row-count assertion was corrected from 1,680 to
3,360 after the first execution; no split, model, fusion or endpoint changed.

This is public-device, native-trial recognition under instructed intensity
conditions, with calibration gestures supplied by label. It does not measure
physical force, new-day retention, automatic gesture onset, calibration setup
time or the user's 250 Hz eight-channel live device. Final users had been
examined in earlier project studies and these scores are descriptive.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.` using
`python -m benchmarks.new_bank_v2.force_calibration` followed by
`python -m benchmarks.new_bank_v2.verify_force_calibration`.
