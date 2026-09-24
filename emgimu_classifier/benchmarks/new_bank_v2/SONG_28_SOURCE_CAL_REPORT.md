# Source-only probability calibration for Song's 28-state factorized model

The [protocol](SONG_28_SOURCE_CAL_PROTOCOL.json) and [runner](song_28_source_calibration.py) were committed as `a92cfde` before this result was computed. This fills a **narrow probability-calibration gap** in the earlier [factorized hand × arm comparison](SONG_ARM_CAL_REPORT.md): that study multiplied two uncalibrated classifier probabilities. It does not change the live recognizer or calibrate from target-session labels.

S01 and S02 each serve once as the training session and once as the held-out source session. Every feature-family state, scaler and balanced logistic classifier is refit within its fold. The 285 held-out native-trial probability rows select hand and arm scalar temperatures independently by pooled source OOF trial LogLoss from the fixed candidates 0.5, 0.75, 1, 1.25, 1.5, 2 and 3. The selected values are **0.5 for hand** and **0.75 for arm**. Full source models fit S01+S02 only; S03 and S04 enter solely for descriptive evaluation. Temperatures act on trial-averaged branch probabilities before the normalized 28-state outer product.

| Native trial set | Joint model | LogLoss ↓ | Brier ↓ | Macro-F1 | Accuracy |
|---|---|---:|---:|---:|---:|
| S03, 140 trials | Original source factorization | 1.0553 | 0.0177 | 0.7159 | 0.7429 |
| S03, 140 trials | Source OOF temperatures | **0.8858** | **0.0149** | 0.7159 | 0.7429 |
| S04, 144 trials | Original source factorization | 1.1887 | 0.0198 | 0.5789 | 0.6458 |
| S04, 144 trials | Source OOF temperatures | **1.0157** | **0.0175** | 0.5789 | 0.6458 |

The probability scores improve on both sessions, but **no joint hard decision changes**: separate positive scalar temperatures preserve each branch's argmax, and the argmax of their outer product follows the same pair. This is evidence for better probability quality under the available same-person/day cue-labelled conditions, not higher recognition accuracy or lower continuous false-alarm rate. The earlier S04 result had already been inspected, S01–S03 failed formal collection readiness, and there are no new-day or new-user recordings.

The [read-back verifier](verify_song_28_source_calibration.py) recomputes the temperature selection from all [285 source OOF rows](SONG_28_SOURCE_CAL_OOF.csv), all four metric groups from [568 saved held-out prediction rows](SONG_28_SOURCE_CAL_TRIAL_PREDICTIONS.csv), and the calibrated outer product from the uncalibrated joint marginals. The uncalibrated rows match the frozen factorized baseline exactly; the maximum calibrated replay difference is `4.44e-16`. Full-precision [results](SONG_28_SOURCE_CAL_RESULTS.json) and [verification](SONG_28_SOURCE_CAL_VERIFICATION.json) retain hashes and scores.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.` using `D:/miniconda/python.exe -m benchmarks.new_bank_v2.song_28_source_calibration` followed by `D:/miniconda/python.exe -m benchmarks.new_bank_v2.verify_song_28_source_calibration`. Raw Song HDF5 recordings remain outside Git.
