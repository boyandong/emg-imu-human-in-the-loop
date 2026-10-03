# Document-exact D/E reliability: frozen GRAB replay

This opt-in experiment reuses two frozen Day1-trained provider probability streams and source-fitted features on [GRABMyo v1.1.0](https://physionet.org/content/grabmyo/1.1.0/) (DOI: 10.13026/89dm-f662, CC BY 4.0). `F0v2` and `F0v2+scale_pattern` are highly related providers, so this is a formula/protocol check rather than proof of useful independent fusion. The feature array in this delivery is a public-data derivative, not a new recording.

The checksum-bound [protocol](DOCUMENT_RELIABILITY_GRAB_PROTOCOL.json) selects `n0=2` and temperature `0.5` by Day2 leave-one-subject-out source CV across eight users and 1/2/5 trials per class. The population prior is then recomputed from all 64 Day2 repetition-6/7 trials. Each Day3 user contributes separate calibration repetitions 1..N and evaluation repetitions 6/7; no evaluation trial enters that user's calibration. The Day1 classifier and feature scaler remain frozen. Day3 had been inspected in earlier project work, so these are **descriptive** results, not a prospective holdout.

| Day3 shots/class | Arm | Macro F1 | Log loss | Brier |
|---:|---|---:|---:|---:|
| 1 | Day2 population prior | 0.9222 | 0.2557 | 0.1262 |
| 1 | Document-exact personal D/E | 0.9222 | 0.2582 | 0.1275 |
| 2 | Day2 population prior | 0.9222 | 0.2557 | 0.1262 |
| 2 | Document-exact personal D/E | 0.9055 | 0.2577 | 0.1280 |
| 5 | Day2 population prior | 0.9222 | 0.2557 | 0.1262 |
| 5 | Document-exact personal D/E | 0.9222 | 0.2563 | 0.1264 |

Each row evaluates the same 64 Day3 trials (eight users × four gestures × two repetitions). The legacy formula with the **same** selected source policy agrees to displayed precision because these calibration classes have nonzero between-class spread; an analytical zero-spread oracle separately distinguishes the equations. Here personal reliability does not improve Day3 log loss or Brier at any budget and reduces macro F1 at two shots. It is therefore not promoted as a default. This says nothing about live 250 Hz eight-channel recognition or a calibrated body frame.

The exact 576 probability rows, 24 calibration cells, selected source grid, weights, hashes and boundary are in [predictions](DOCUMENT_RELIABILITY_GRAB_PREDICTIONS.csv) and [results](DOCUMENT_RELIABILITY_GRAB_RESULTS.json). Reproduce from the bundled derived arrays and frozen parent predictions with `PYTHONPATH=src;. python benchmarks/new_bank_v3/document_reliability_grab_run.py`; raw GRAB archives are unnecessary for this readback.
