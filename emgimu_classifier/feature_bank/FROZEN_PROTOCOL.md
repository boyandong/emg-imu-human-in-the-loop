# Feature Bank frozen protocol

## Data units and identifiers

Every source adapter emits immutable trial records containing dataset, subject, session or domain,
condition, original gesture, canonical gesture when approved, trial ID, sample rate, channel layout
and source-file checksum. The split table contains trial IDs before any window extraction.

## Development and final evaluation

Each dataset defines training, validation and final test groups from real factors. Model and family
screening uses training and validation only. A final test partition is opened once after the family
shortlist, calibration hyperparameters and fusion rules are frozen. Dataset-native tasks use simple
LDA, shrinkage LDA, logistic regression or RBF-SVM; a deep network cannot hide representation
effects.

## Calibration

Personal calibration holds out target users and evaluates 0, 1, 2 and 5 labelled trials per class
when available. Session calibration builds a long-term profile from source sessions, takes the same
budgets from a target session, and tests only its remaining trials. A fixed seed chooses calibration
trials and the selected IDs are saved. Unsupported budgets are reported explicitly.

## Feature state

Every run serializes family ID and dimension, fitting rows, channel count, sample rate, thresholds,
normalization, CSP filters, covariance reference, spectral bands, temporal template trial IDs,
prototype trial IDs, probability calibration and fusion parameters. Transforming validation or test
data may not mutate any state.

## Metrics and analysis

Reports include accuracy, macro-F1, per-class F1, log-loss, Brier, ECE, subject/domain cells,
calibration curves, nuisance and gesture distances where defined, conditional OOF deltas, error
complementarity and the available robustness vector. The primary system comparison uses the worst
available robustness cell as well as the mean. Missing factors remain `N/A`.

## Reproducibility

The default random seed is 20260915. Run IDs are unique and outputs are never overwritten. Raw data
is read-only under `D:\emg-imu-benchmarks\data\raw`; processed data and run artifacts use separate
directories. RTX 4060 is selected automatically for later PyTorch experiments, while the required
classical Feature Bank experiments execute on CPU.
