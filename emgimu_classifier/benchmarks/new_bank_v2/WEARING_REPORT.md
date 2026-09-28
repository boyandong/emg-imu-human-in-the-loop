# New eight-channel v2 electrode-shift experiment

The [protocol](WEARING_PROTOCOL.json) and [runner](wearing_run.py) were frozen
in commit `db62fc8` before outcomes were inspected. This is a new, independently
implemented F0v2/F2a/F3c comparison on LibEMG CIILData's native 8-channel,
200 Hz wearing domains. Each subject's unshifted `training` trials fit feature
state, standardization and a balanced logistic classifier; `trial_1`–`trial_4`
are held out. Subjects 15–17 select an arm, and subjects 18–20 evaluate that
fixed choice. No target-domain samples fit any state or threshold.

| Arm | Validation macro-F1 | Validation log loss | Final macro-F1 | Final log loss | Final worst-domain F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous F0, exact replay | 0.4533 | 1.5535 | 0.4912 | 1.2047 | 0.3938 |
| New F0v2 | 0.5775 | 1.2916 | 0.6170 | 1.0838 | 0.5505 |
| F0v2 + F2a | 0.6960 | 1.0786 | 0.5996 | 1.0901 | 0.5400 |
| F0v2 + F3c | 0.6842 | 0.8743 | **0.7737** | **0.6773** | **0.7523** |
| F0v2 + F2a + F3c | **0.7382** | **0.8397** | 0.6560 | 0.7960 | 0.6027 |

The frozen validation rule selects **F0v2+F2a+F3c**, which raises final
macro-F1 from 0.4912 to 0.6560 against the prior F0 baseline and raises
worst-domain F1 from 0.3938 to 0.6027. On the 120 final native trials it
corrects 29 F0 errors and creates 11 new errors. F0v2 alone also improves over
F0. The F0v2+F3c arm wins the final set but was *not* selected on validation;
promoting it on that basis would use final outcomes for selection. F2a's weaker
final showing also cautions against declaring every spatial block useful.

The [saved predictions](WEARING_TRIAL_PREDICTIONS.csv) contain 1,200 arm–trial
probability rows. The [independent verifier](verify_wearing.py) checks native
trial identity and disjoint source/target trials, recomputes all ten pooled
score groups and subject/domain cells, redoes validation selection, and matches
the earlier F0 probabilities within `7.1e-14`. A second run produced byte-identical
results and predictions. Full precision, archive/protocol hashes and trial IDs
are in [results](WEARING_RESULTS.json) and [verification](WEARING_VERIFICATION.json).

This is positive same-user, public-device, trial-level electrode-shift evidence.
It does not validate the user's 250 Hz hardware, physical continuous recognition,
or cross-user transfer. The final subjects were used in earlier project studies,
so this is an independently split *new v2 comparison*, not a wholly untouched
project-wide confirmation set. The live model is unchanged.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.` using
`python -m benchmarks.new_bank_v2.wearing_run` and
`python -m benchmarks.new_bank_v2.verify_wearing`.
