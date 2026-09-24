# Independent new-bank ring features on GRABMyo across days

The [protocol](GRABMYO_PROTOCOL.json) and [runner](grabmyo_run.py) were committed as `52b0dd0` before this outcome was computed. This is a public eight-channel, three-day screen of the new v1 ring-lag and correlation-spectrum families. It uses the same fixed eight subjects, F1–F8 channels, four gestures, 672 five-second recordings, 2048 Hz rate and Day1/Day2/Day3 split as the [earlier GRABMyo study](../grabmyo_crossday/REPORT.md). Every one of the 1,344 selected source files was checked against the publisher's `SHA256SUMS.txt`; the local manifest SHA-256 and frozen protocol hash are in [results](GRABMYO_RESULTS.json). The earlier F0 probabilities were reproduced to within `1e-10` for all 448 held-out recordings, so the added-family comparisons share the baseline rather than silently changing it.

All 20 disjoint 250 ms windows from one recording are summarized by the mean and standard deviation of each feature; no window enters a different day split. Day1 rest alone sets F0's noise reference. New-family metadata, standardization and logistic coefficients fit Day1 only. Day2 chooses the arm by macro-F1 with log-loss tie-break; Day3 is final. Four models differ only in their feature blocks.

| Arm | Day2 macro-F1 | Day2 log loss | Day2 Brier | Day3 macro-F1 | Day3 log loss | Day3 Brier | Day3 minimum subject F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 | 0.9551 | 0.1507 | 0.0795 | **0.9008** | **0.3007** | **0.1552** | **0.6667** |
| F0 + ring lag | 0.9686 | 0.1422 | 0.0733 | 0.8862 | 0.3397 | 0.1774 | 0.6667 |
| F0 + correlation spectrum | 0.9551 | 0.1367 | 0.0679 | 0.8363 | 0.4742 | 0.2405 | 0.6209 |
| F0 + both | **0.9687** | **0.1363** | **0.0666** | 0.8552 | 0.3877 | 0.1968 | 0.6209 |

Day2 selects **F0 + both**. Its Day3 macro-F1 is 0.0456 below F0, its log loss increases by 0.0870, and its Brier score increases by 0.0416. On Day3 it fixes three recordings that F0 missed, but loses 13 that F0 classified correctly. Thumb-index opposition recall drops from 0.7857 to 0.7321, open-hand recall from 0.8750 to 0.8036, and close-hand recall from 0.9464 to 0.8929; rest recall remains 1.0. The [read-back audit](GRABMYO_VERIFICATION.json) recomputes all eight split/arm metric groups and paired counts from all [1,792 prediction rows](GRABMYO_TRIAL_PREDICTIONS.csv).

The positive Day2 result does not transfer to the final day. The new ring families remain research options and are **not promoted to the live recognizer**. The evidence is limited to 8 of the public dataset's 43 participants, four gesture classes in isolated recordings, a 2048 Hz forearm ring from different hardware, and no continuous onset/latency or own-device test. The file-level Zenodo re-placement result and same-day Song result answer different questions and do not override this cross-day regression.

To reproduce from `emgimu_classifier` with `PYTHONPATH=src;.`, run `python -m benchmarks.new_bank_v1.grabmyo_run` using the already verified subset at `D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1`, then `python -m benchmarks.new_bank_v1.verify_grabmyo`. Raw files remain outside Git.
