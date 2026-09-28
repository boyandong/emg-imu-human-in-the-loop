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

## Paired feature-family analysis

The [paired analysis](force_paired_analysis.py) uses only the frozen 5,880
arm–trial probability rows. Each comparison matches the same native trials:
588 validation and 588 final. The [conditional table](FORCE_CONDITIONAL.csv),
[complementarity table](FORCE_COMPLEMENTARITY.csv), and
[interaction table](FORCE_INTERACTION.csv) include pooled, subject, and
intensity-condition cells. A positive log-loss delta means improvement over
F0v2; interaction uses `P = -log_loss` and
`P(both) - P(F2a) - P(F3c) + P(F0v2)`.

| Phase | F2a F1 delta | F3c F1 delta | Both F1 delta | F2a/F3c error correlation | Log-loss interaction | F1 interaction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | -0.0644 | -0.0646 | -0.0624 | 0.7505 | +0.3159 | +0.0667 |
| Final | +0.0418 | -0.0236 | -0.0142 | 0.6939 | -0.1003 | -0.0324 |

The validation interaction is positive only relative to adding the two
families separately: the joint arm is still worse than F0v2 by 0.0624 F1 and
0.8257 log loss. On final users, F2a alone helps F1 but the combination loses
0.0142 F1 relative to F0v2. F2a and F3c prediction disagreements rise from
18.88% to 29.93%, while their error correlation remains high. Thus their
errors are only partly complementary and the combination has no stable
cross-user benefit. These final-user diagnostics are descriptive; they do not
change the validation-selected F0v2 arm. The [audit](FORCE_PAIRED_AUDIT.json)
records the source prediction SHA-256 and row counts. Run
`python -m benchmarks.new_bank_v2.force_paired_analysis --verify` to check
that every published table is byte-identical to a fresh derivation.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.` using
`python -m benchmarks.new_bank_v2.force_run` followed by
`python -m benchmarks.new_bank_v2.verify_force` and
`python -m benchmarks.new_bank_v2.force_paired_analysis --verify`.
