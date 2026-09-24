# Public DS2 v8: subject-held-out gesture-code increments

This is a new, bounded experiment on the [publisher-linked public DS2 candidate](https://www.mdpi.com/2306-5729/10/12/194). It does not reproduce the unavailable historical B0/X1-H/X2 force runs. The exact raw-to-MAV window join establishes the published gesture-code order, and the exact TDMS waveform join assigns subject folders to 2,833 of 2,863 raw trials. We exclude the 30 unmatched trials and the one mixed-code trial, leaving 2,832 uniformly labelled trials. No low/medium/high force label is inferred.

Subjects 01–12 train, 13–16 validate and 17–20 form the final held-out partition. Each eligible trial contributes three non-overlapping 375-sample action windows at raw offsets 3,000, 6,000 and 9,000. Feature-family references, F0 thresholds, scaling and logistic models fit only on training subjects. Each validation/final trial gets one vote from the mean of its three window probabilities. Every arm uses the same fixed balanced logistic classifier (`C=1`) and the same trials; no final-subject score selects or refits a model. This protocol estimates cross-subject **gesture-code** transfer of the current reference families, not force robustness or own-device 8-channel recognition.

| Arm | Validation macro-F1 / LogLoss (587 trials) | Final macro-F1 / LogLoss (558 trials) | Final ΔF1 / ΔLogLoss vs F0 |
|---|---:|---:|---:|
| F0 | 0.4049 / 1.4528 | 0.4413 / 1.7615 | — |
| F0 + F1 scale pattern | 0.4211 / 1.4241 | 0.4475 / 1.7167 | +0.0062 / +0.0448 |
| F0 + F2a trace covariance | 0.3894 / 1.4908 | 0.4534 / 1.7910 | +0.0121 / −0.0294 |
| F0 + F2c SPD tangent | 0.3842 / 1.4483 | 0.4834 / 1.8095 | +0.0421 / −0.0479 |
| F0 + F4 spectral | 0.4336 / 1.3390 | 0.5183 / 1.8344 | +0.0770 / −0.0729 |

Positive ΔLogLoss means an improvement. F1 scale pattern is the only addition that improves final LogLoss, Brier and macro-F1 together, but its F1 gain is small. SPD and spectral features improve final macro-F1 while worsening LogLoss. F4's 55 corrected versus 23 new wrong final trials explain the accuracy gain, but subject 19's LogLoss rises from 2.8645 to 3.3491; this is not an unconditional probability-quality improvement. The four final subject F4 macro-F1 values rise relative to F0, yet validation subject 14 falls. These are exploratory descriptive results from four final subjects, not proof of a universal specialist or deployable model.

[RESULTS.json](RESULTS.json) contains all scores, per-subject results and source hashes. [TRIAL_PREDICTIONS.csv](TRIAL_PREDICTIONS.csv), [CONDITIONAL_INCREMENTAL.csv](CONDITIONAL_INCREMENTAL.csv) and [ERROR_COMPLEMENTARITY.csv](ERROR_COMPLEMENTARITY.csv) preserve held-out probabilities and paired comparisons. [VERIFICATION.json](VERIFICATION.json) records a read-back check of all 5,725 prediction rows, subject/gesture joins and recomputed scores. The executable producer and verifier are in `../scripts/`.

The split uses verified subject folders only; 30 MAT trials have no exact TDMS counterpart, and one otherwise matched trial has mixed window labels. The public release lacks per-trial force labels and cannot be identified as the original historical experiment input. Three channels and 1500 Hz also differ from the user's Song 8-channel device. Those limits prevent this study from satisfying the historical DS2 force requirement or the real-device multiuser/multiday requirement.
