# One-shot calibration of the new eight-channel wearing bank

The [protocol](WEARING_CAL_PROTOCOL.json) and [runner](wearing_calibration.py)
were frozen in commit `bfa19ae` before outcomes. This follow-up uses the
previously validation-selected F0v2+F2a+F3c arm and a prespecified F0v2-only
control. For each of six LibEMG CIILData subjects and four shifted wearing
domains, each class has two complete native trials. One trial per class fits
same-domain prototypes; the other is scored. The two reciprocal assignments
let every trial enter evaluation once per method, never in the calibration
set of that evaluation. Source features, scaler, classifier and prototype
temperature fit only unshifted source trials. Calibration uses five target
trials per domain, one per gesture; no classifier or fusion weight is refit.

| Phase | Feature arm | Shots/class | Macro-F1 | Log loss | Brier | Worst-domain F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validation | F0v2 | 0 | 0.5775 | 1.2916 | 0.6159 | 0.5433 |
| Validation | F0v2 | 1 | **0.9752** | 0.3219 | 0.1770 | **0.9664** |
| Validation | F0v2+F2a+F3c | 0 | 0.7382 | 0.8397 | 0.3941 | 0.6870 |
| Validation | F0v2+F2a+F3c | 1 | 0.9420 | **0.2643** | **0.1408** | 0.8993 |
| Final | F0v2 | 0 | 0.6170 | 1.0838 | 0.5484 | 0.5505 |
| Final | F0v2 | 1 | **0.9338** | 0.4040 | 0.2201 | **0.8629** |
| Final | F0v2+F2a+F3c | 0 | 0.6560 | 0.7960 | 0.4436 | 0.6027 |
| Final | F0v2+F2a+F3c | 1 | 0.9163 | **0.3835** | **0.1920** | 0.8615 |

On matched final trials, the fixed one-shot F0v2 branch corrects 39 source-only
errors and creates two new errors; the fixed full-feature branch corrects 34
and creates four. Thus target-domain calibration is strongly useful on this
specific two-repetition benchmark. F0v2 has higher calibrated macro-F1,
while the full arm has lower calibrated log loss and Brier. The protocol did
not authorize a new arm-selection step based on these outcomes, so neither
ranking changes the previously selected zero-shot arm by itself.

The [independent verifier](verify_wearing_calibration.py) checks all 48
calibration/evaluation assignments, native trial identity, disjoint source,
calibration and evaluation IDs, all 960 saved probability rows, and all eight
pooled score groups plus subject/domain cells. Zero-shot probabilities match
the frozen parent experiment exactly; rerunning the study yields byte-identical
[results](WEARING_CAL_RESULTS.json) and
[predictions](WEARING_CAL_PREDICTIONS.csv). A `float32` probability-sum warning
in the first execution was corrected by `float64` prototype softmax and row
normalization; the frozen allocation, features, temperature and 0.5 mixture
rule did not change.

This measures an *assisted, same-domain, one-shot-per-gesture* user flow on a
public 200 Hz device, not the current 250 Hz hardware. The second same-domain
trial is a narrow test of reapplication adaptation; it cannot establish
later-day persistence, online onset detection, or physical setup/calibration
time. Two shots per class would leave no same-domain evaluation trial, so
2- and 5-shot cells are unsupported here rather than extrapolated.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.` using
`python -m benchmarks.new_bank_v2.wearing_calibration` and
`python -m benchmarks.new_bank_v2.verify_wearing_calibration`.
