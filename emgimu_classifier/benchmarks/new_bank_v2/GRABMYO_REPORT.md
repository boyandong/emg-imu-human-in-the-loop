# Independent new-bank F0/F2a/F3c cross-day screen

The [new implementation](../../src/emgimu/feature_bank/new_bank_v2.py), [frozen protocol](GRABMYO_PROTOCOL.json), and [runner](grabmyo_run.py) were committed as `9e51cd4` before the outcomes were computed. The new F0 thresholds come only from Day1 Rest windows; F2a uses centered covariance, fixed 0.05 shrinkage and trace normalization; F3c summarizes circular lags of that covariance, including half-window drift for these 512-sample windows. The four unchanged v1 families remain opt-in and are not silently mixed into these arms. This is forward development under new versioned names, not recovery of historical X1-H/RLCS/CES.

The [official GRABMyo subset](../grabmyo_crossday/REPORT.md) has eight selected users, four gestures, seven recordings per class on each of three days, and eight forearm channels at 2048 Hz. Every one of the 1,344 selected WFDB files is checked against the publisher's SHA256SUMS. All twenty disjoint 250 ms windows per recording are aggregated by feature mean and standard deviation; Day1 alone fits feature state, standardization and logistic coefficients. Day2 selects the arm by macro-F1 then log loss; Day3 is untouched for selection. The new F0 baseline reproduces the previous F0 probabilities to within `1e-10` for all 448 held-out recordings.

| Arm | Day2 macro-F1 | Day2 log loss | Day3 macro-F1 | Day3 log loss | Day3 Brier | Day3 minimum-subject F1 |
|---|---:|---:|---:|---:|---:|---:|
| F0 | **0.9551** | **0.1507** | **0.9008** | **0.3007** | **0.1552** | 0.6667 |
| F0 + F2a | 0.9312 | 0.1871 | 0.8536 | 0.7619 | 0.2034 | 0.6583 |
| F0 + F3c | 0.9198 | 0.1708 | 0.8764 | 0.3848 | 0.1876 | **0.7300** |
| F0 + F2a + F3c | 0.9220 | 0.1865 | 0.8567 | 0.7153 | 0.1980 | 0.6583 |

Day2 selects **F0**. On Day3, adding F2a corrects two F0 mistakes but creates 12 new errors; F3c corrects five and creates ten; both together correct five and create 14. The [read-back verifier](GRABMYO_VERIFICATION.json) recomputes every reported metric group and these paired counts from [1,792 saved probability rows](GRABMYO_TRIAL_PREDICTIONS.csv). F3c has a higher Day3 minimum-subject F1 than F0, but this is a final-only descriptive observation; its Day2 minimum-subject F1 is lower than F0, and its pooled Day3 F1 and log loss regress. It is therefore retained as a research candidate, not selected for the live model.

This study demonstrates formula implementation and a negative cross-day test on different hardware. It does not validate the user's 250 Hz device, continuous action onsets, force, real electrode re-donning, IMU calibration or the remaining F0–F9 families. No current recognizer or historical baseline is overwritten.

To reproduce from `emgimu_classifier` with `PYTHONPATH=src;.`, run `python -m benchmarks.new_bank_v2.grabmyo_run` and `python -m benchmarks.new_bank_v2.verify_grabmyo`; raw files stay outside Git at `D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1`.
