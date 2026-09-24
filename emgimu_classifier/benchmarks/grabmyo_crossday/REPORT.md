# GRABMyo selected eight-channel cross-day confirmation

This is an independent **public-dataset** screen of the existing Feature Bank families. It is not a test of the project's 250 Hz eight-channel device, a UniBo model, or a reproduction of historical DS2 force results. The [frozen protocol](PROTOCOL.json) was committed as `11c2f8b` before acquisition and outcome inspection. No feature, split, class or hyperparameter was changed after results were seen.

## Source and integrity

The source is [PhysioNet GRABMyo v1.1.0](https://physionet.org/content/grabmyo/1.1.0/) (Jiang, Pradhan and He, DOI [10.13026/89dm-f662](https://doi.org/10.13026/89dm-f662)). Its current landing page specifies CC BY 4.0; its older `readme.txt` still says ODC-BY-1.0, so the repository metadata follows the current landing page and records this discrepancy. The publisher's `GestureList.JPG`, whose SHA-256 matched the publisher manifest, confirms codes 4 (thumb-index opposition), 15 (hand open), and 16 (hand close); code 17 is rest. The public collection has 43 participants and three days. This screen deliberately uses only participants 1–8, all three days, four classes, and all seven trials per cell.

The 672 selected WFDB records comprise 1,344 files and 442,479,520 bytes, kept outside Git at `D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1`. Every selected file matched the publisher's `SHA256SUMS.txt`; the manifest itself has SHA-256 `757ea64fa5134b7b3b84d9f79e30cfa1ac4d2c65a40ce9b427db11dd8258974e`. The full 9.4 GB collection was **not** downloaded. Only channels F1–F8, the first forearm ring, are read; 2048 Hz is preserved. Each five-second trial becomes twenty disjoint 250 ms windows. Feature means and standard deviations make one vector per trial. Training-day rest windows alone fit the F0 rest-noise reference. Scaler and logistic classifier fit day 1 only; day 2 is validation and day 3 is final. All three days have 224 distinct trials. No trial or window crosses a split.

## Frozen outcome

| Arm | Day 2 macro-F1 | Day 2 LL | Day 2 Brier | Day 3 macro-F1 | Day 3 LL | Day 3 Brier | Day 3 worst-subject macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 | 0.9551 | 0.1507 | 0.0795 | **0.9008** | **0.3007** | **0.1552** | **0.6667** |
| F0 + F2a | 0.9267 | 0.1875 | 0.1050 | 0.8536 | 0.7640 | 0.2042 | 0.6583 |
| F0 + F4 | 0.9413 | **0.1444** | **0.0757** | 0.9002 | 0.4767 | 0.1567 | 0.6530 |

Numbers are rounded display values from [RESULTS.json](RESULTS.json); all 1,344 day-2/day-3 arm–trial predictions and probabilities are in [TRIAL_PREDICTIONS.csv](TRIAL_PREDICTIONS.csv). F0 is best on day 3 for all three primary metrics. F2a loses 0.0472 macro-F1 and adds 0.4633 LL relative to F0. F4 has a small day-2 calibration gain but loses it on day 3; its day-3 macro-F1 difference is -0.0006 and LL difference +0.1759. Rest recall is 1.0 for every arm on the final day, while F0 hand-open recall is 0.875 and thumb-index opposition recall 0.786. The lowest individual day-3 F0 macro-F1 is 0.667, so aggregate accuracy hides user variation.

The outcome argues against adding either family by default for this specific cross-day four-class protocol. It does **not** disprove those families in other settings. The eight subjects are a fixed first-ID subset, not a representative sample of all 43, and trials are labelled five-second recordings rather than continuous online detections. Electrode design, sample rate and acquisition electronics differ from the user's device. This study cannot resolve own-device real-time false rest, onset, force, posture, IMU fusion, re-donning isolated from day, or historical X1-H/RLCS/CES equivalence.

## Reproduce

From the repository root with `PYTHONPATH=emgimu_classifier/src;emgimu_classifier` on Windows, run `python -m benchmarks.grabmyo_crossday.run acquire` and then `python -m benchmarks.grabmyo_crossday.run evaluate`. `acquire` verifies existing files against official digests before reuse; `evaluate` verifies all selected files again. The selected raw data stay outside Git. The parser/split checks are in `emgimu_classifier/tests/test_grabmyo_crossday.py`.
