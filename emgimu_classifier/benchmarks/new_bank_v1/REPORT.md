# New eight-channel EMG feature bank v1

This is a new implementation, not a reconstruction of unavailable historical code. The four opt-in families live in [`new_bank_v1.py`](../../src/emgimu/feature_bank/new_bank_v1.py) and are available through `new_bank_v1_registry()`. Existing default or deployed model definitions were not altered. The [protocol](PROTOCOL.json) was committed as `606050d` before outcome inspection.

To use one family in another experiment, construct `FeatureBatch(source_windows, 250.0)`, call `FrequencyDirectionV1().fit(source_batch)`, then call the fitted object's `transform(query_batch)`. `feature_names` gives the ordered output schema. The same contract applies to the other three families; the source fit must be persisted alongside a downstream classifier.

The input contract is finite `[window, sample, 8]` EMG with an explicit sampling rate. All families fit only source metadata and reject different channel count, rate or window length at transform time. The module uses no historical/reference feature transform. Scale pattern is eight per-channel RMS values divided by their global RMS (no amplitude coordinate). Ring lag gives the mean and standard deviation of 25 ms moving-RMS envelope correlations for circular lags 1–4 (eight values). Correlation spectrum gives eight descending nonnegative envelope-correlation eigenvalues normalized to sum one. Frequency direction gives four Hann-FFT bandwise channel-power vectors, each normalized to unit L2 norm (32 values); band limits are clipped below Nyquist, including at 250 Hz. Zero signals stay finite. The property tests verify global-gain invariance, circular rotation, arbitrary channel permutation, band-specific responses and rate/channel mismatch rejection.

Both native public datasets use eight channels and nonoverlapping 200 ms windows, aggregated to one mean feature vector per native trial. The classifier is a source-fitted balanced logistic model with source-fitted standardization, fixed C=1. Force trains on six subjects' Ramp recordings and tests on separate validation subjects 7–8 and final subjects 9–10 over eleven held-out intensity conditions. Wearing trains a personal model on each subject's unshifted trials and tests four shifted domains, with validation subjects 15–17 and final subjects 18–20. Force archive SHA-256 is `237212d87f50c9b621e572bc780d99fac3a9af60b99ec8b455fb3d3dbb2d15e7`; wearing archive SHA-256 is `4cfa9a4861193f230179fa87d53eda7503e66b5cfedeae81c81e476f53f6b2b6`.

| Force arm | Validation macro-F1 | Final macro-F1 | Final log loss | Final Brier | Final worst-condition macro-F1 |
|---|---:|---:|---:|---:|---:|
| F0 | 0.4835 | **0.5118** | **2.1005** | **0.6666** | **0.4102** |
| F0 + scale pattern | 0.5247 | 0.4942 | 2.1811 | 0.7016 | **0.4102** |
| F0 + frequency direction | 0.5476 | 0.4841 | 2.3953 | 0.7485 | 0.3784 |
| F0 + both | **0.5504** | 0.4837 | 2.4231 | 0.7512 | 0.4023 |

| Wearing arm | Validation macro-F1 | Final macro-F1 | Final log loss | Final Brier | Final worst-domain macro-F1 |
|---|---:|---:|---:|---:|---:|
| F0 | 0.4533 | 0.4912 | 1.2047 | 0.6216 | **0.3938** |
| F0 + ring lag | 0.5326 | 0.4747 | 1.1409 | 0.6210 | 0.3705 |
| F0 + correlation spectrum | 0.5398 | **0.5092** | 1.1938 | 0.6271 | 0.3760 |
| F0 + both | **0.5599** | 0.4625 | **1.1157** | **0.6079** | 0.3562 |

The validation-best arm is “F0 + both” for each task. Both regress below F0 macro-F1 on their independent final subjects. The final-only correlation-spectrum row is promising descriptively, but choosing it because it wins final F1 would use the final set for model selection. We therefore leave v1 as an available research module and **do not enable it in the live recognizer**. The result is a useful negative selection outcome: a clean new implementation does not by itself fix cross-force or cross-wearing transfer. Further selection needs a new frozen validation cohort, especially from the user's own 250 Hz device. Neither public dataset proves live hand-open accuracy.

Unrounded pooled, subject and condition cells plus source/target trial IDs are in [`RESULTS.json`](RESULTS.json). [`TRIAL_PREDICTIONS.csv`](TRIAL_PREDICTIONS.csv) contains 5,664 arm–trial probabilities. Every probability vector sums to one, all native source/target trial sets are disjoint, and macro-F1/log loss were recomputed exactly from the CSV. Reproduce with `PYTHONPATH=emgimu_classifier/src;emgimu_classifier` and `python -m benchmarks.new_bank_v1.run` from the repository root.

## Own-device 250 Hz exploratory check

The [separately frozen Song protocol](SONG_PROTOCOL.json) applies the same new module to the user's four available 8-channel HDF5 sessions, using the existing causal filter and formal-trial labels. S01/S02 train, S03 validates, and S04 is an exploratory final session (it has already been inspected in earlier project studies). All four sessions are from one participant on one day. Models fit source windows and average predicted probabilities across each held-out trial. This is **not** continuous event recognition. The 1,704 arm–trial probabilities are in [`SONG_TRIAL_PREDICTIONS.csv`](SONG_TRIAL_PREDICTIONS.csv), with unrounded scores and HDF5 hashes in [`SONG_RESULTS.json`](SONG_RESULTS.json).

| Song arm | S03 macro-F1 | S04 macro-F1 | S04 log loss | S04 hand-open recall | S04 neutral recall |
|---|---:|---:|---:|---:|---:|
| F0 | 0.9359 | 0.9068 | 0.4272 | 0.9722 | 0.9444 |
| F0 + scale pattern | 0.9217 | 0.8763 | 0.4197 | 0.9722 | 0.9444 |
| F0 + ring lag | 0.9283 | 0.9081 | 0.3861 | 0.9722 | 0.9444 |
| F0 + correlation spectrum | 0.9359 | **0.9137** | 0.4074 | **1.0000** | 0.9444 |
| F0 + frequency direction | **0.9499** | 0.8862 | 0.4104 | **1.0000** | 0.9444 |
| F0 + all four | 0.9071 | 0.8854 | **0.3707** | 0.9722 | 0.9444 |

S03 selects F0 + frequency direction by the frozen macro-F1 rule, yet it regresses below F0 on S04. The S04-only correlation-spectrum gain is too small and too selection-dependent for deployment. The high trial-level hand-open recall is compatible with the user's poor real-time experience: complete cued trials and averaged probabilities do not measure action onset, transient rest decisions or latency. A new live model should therefore be selected using a newly recorded, time-stamped continuous session rather than this already-used S04 set.

## Independent electrode re-placement dataset

A separately frozen [file-level check](ZENODO_REPLACEMENT_REPORT.md) used the official eight-channel, 1000 Hz Zenodo archive. Nine subjects had valid P1 galleries; P2 and P3 each contributed 79 matched complete movement recordings. The P2-selected F0 + ring lag + correlation spectrum arm reached 0.5972 mean-subject top-1 on P2 and 0.6188 on P3, versus F0's 0.5170 and 0.6003. This small P3 advantage is a cross-position **whole-recording retrieval** result, not trial-level or online recognition. It does not change the decision to keep the live recognizer unchanged.

## Independent public cross-day check

The [frozen GRABMyo screen](GRABMYO_REPORT.md) adds a trial-level three-day check of the new ring families with the exact earlier F0 baseline replayed. Day2 selected F0 + ring lag + correlation spectrum at macro-F1 0.9687 versus F0 0.9551, but its Day3 macro-F1 fell to 0.8552 versus F0 0.9008. This validation-to-final reversal strengthens the decision not to enable the new bank in the live model without new own-device continuous and cross-day validation.
