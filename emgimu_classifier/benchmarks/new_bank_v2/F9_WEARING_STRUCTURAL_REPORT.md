# Exploratory F9 structural specificity under electrode re-wearing

The [frozen rule](F9_WEARING_STRUCTURAL_PROTOCOL.json) was fitted separately
on each subject's training-domain, eight-channel 200 Hz windows in the public
LibEMG Electrode Shift dataset. It flags a native trial only when a channel
is zero or flat for at least half of one window. ADC clipping is unavailable
because the archive does not attest a rail; mains and pre-highpass states are
also unavailable. The unchanged [F0v2 predictions](WEARING_TRIAL_PREDICTIONS.csv)
are matched by native trial ID. This study did not retrain a classifier or
select a threshold using after-wearing outcomes.

| Phase | After-wearing domains | Trials retained | Correct rejected | Errors rejected | Existing F0v2 errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation, subjects 15–17 | 4 | 120/120 | 0 | 0 | 49 |
| Descriptive final, subjects 18–20 | 4 | 120/120 | 0 | 0 | 45 |

Every individual after-wearing domain retains all 30 trials per phase.
Holding one channel constant in a copied target signal triggers the rule
for all 240 trials. Thus ordinary electrode re-wearing did not produce a
structural fault flag in this selected public set, while the synthetic
flatline remained observable. This does not establish sensitivity to
measured physical faults, improvements in gesture classification, or safe
live behavior. The same users and target domains had already been inspected
in Feature Bank studies, so this is exploratory specificity evidence.
The structural rule is **not deployed**.

The [result](F9_WEARING_STRUCTURAL_RESULTS.json) binds the archive, protocol,
per-subject fitted state hashes and [trial rows](F9_WEARING_STRUCTURAL_TRIALS.csv).
An independent test reads those rows back against frozen F0v2 probabilities.
