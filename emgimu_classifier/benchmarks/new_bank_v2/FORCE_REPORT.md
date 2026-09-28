# New eight-channel v2 cross-user intensity experiment

The [protocol](FORCE_PROTOCOL.json) and [runner](force_run.py) were committed
as `5fd4653` before outcomes. The public LibEMG ContractionIntensity dataset
records eight channels at 1000 Hz and seven classes, including No Motion
([official class table](https://libemg.github.io/libemg/documentation/data/data.html)).
New F0v2 thresholds fit only the 192 source-user No Motion windows. Subjects
1–6 and Ramp trials fit feature state, trial-level standardization and one
balanced logistic classifier per arm. Users 7–8 validate, users 9–10 provide
the fixed final comparison across eleven intensity conditions. No target
subject or condition fits any state.

| Arm | Validation macro-F1 | Validation log loss | Final macro-F1 | Final log loss | Final worst-condition F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous F0, numerical replay | 0.4835 | 1.7543 | **0.5118** | 2.1005 | 0.4102 |
| New F0v2 | **0.6295** | **1.3440** | 0.4939 | 1.9177 | 0.3579 |
| F0v2 + F2a | 0.5651 | 1.8652 | **0.5357** | **1.8469** | **0.4206** |
| F0v2 + F3c | 0.5649 | 1.9645 | 0.4703 | 1.8739 | 0.3545 |
| F0v2 + F2a + F3c | 0.5671 | 2.1697 | 0.4797 | 1.9035 | 0.4142 |

The prespecified validation rule selects **F0v2 alone**. It corrects 99 F0
errors and creates 14 on validation, but on final users it corrects 28 and
creates 42. Final macro-F1 drops from 0.5118 to 0.4939 and worst-condition
F1 drops from 0.4102 to 0.3579; final log loss improves while Brier worsens
from 0.6666 to 0.6893. F0v2+F2a is the final-only F1 winner and may warrant
a future new validation cohort, but cannot replace the validation-selected
arm using this final result. F3c has no stable cross-user force benefit here.

The [independent verifier](verify_force.py) checks all 5,880 saved arm–trial
probability rows, source/target subject and trial separation, all ten pooled
score groups and subject/condition cells, and the frozen selection rule. The
previous F0 probabilities match within `4.4e-7` (the exact macro-F1 and
log-loss/Brier within `1e-6`), so this is a numerical-tolerance replay rather
than byte-identical reproduction of the older run. A second execution of this
*new* runner produced byte-identical [results](FORCE_RESULTS.json) and
[predictions](FORCE_TRIAL_PREDICTIONS.csv).

This negative transfer is material: the same new spatial features have
positive same-user electrode-shift evidence but do not generalize as a
default cross-user force bank. Intensity conditions are instructed categories,
not measured force. The public device, 1000 Hz rate and trial-level averaging
do not establish the user's 250 Hz live behavior. Final users had been examined
in earlier project studies, so this is descriptive confirmation rather than
a wholly untouched project-wide test.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.` using
`python -m benchmarks.new_bank_v2.force_run` followed by
`python -m benchmarks.new_bank_v2.verify_force`.
