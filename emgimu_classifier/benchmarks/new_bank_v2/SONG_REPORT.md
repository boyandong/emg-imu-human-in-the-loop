# New-bank v2 on the user's 250 Hz eight-channel Song recordings

The [protocol](SONG_PROTOCOL.json) and [runner](song_run.py) were frozen in commit `3746257` before outcomes were computed. This is a follow-up to the [public cross-day v2 screen](GRABMYO_REPORT.md), which selected F0 alone and found no pooled gain from F2a/F3c. This Song screen tests whether the same independently coded formulas are useful at the user's actual 250 Hz rate; it does not revise that public negative result.

The four hash-checked HDF5 v3 sessions belong to one person on 2026-09-18. Continuous EMG is filtered causally with fixed 40 Hz high-pass and 50/100 Hz notches. S01+S02 alone fit the source-Rest F0v2 noise thresholds, all feature state, standardization and balanced logistic coefficients. S03 scores and selects; S04 is reported descriptively. Each formal native trial contributes one averaged probability vector from its available stable 200 ms windows. These are cue-labelled offline stable intervals, not live continuous-action detection.

| Arm | S03 macro-F1 | S03 log loss | S04 macro-F1 | S04 log loss | S04 open recall | S04 pinch recall |
|---|---:|---:|---:|---:|---:|---:|
| Previous F0, exact replay | 0.9359 | 0.3109 | 0.9068 | 0.4272 | 0.9722 | 0.7222 |
| New F0v2 | 0.9277 | 0.3183 | 0.8919 | 0.4207 | 1.0000 | 0.6944 |
| F0v2 + F2a | 0.9496 | 0.2061 | 0.9368 | 0.2479 | 1.0000 | 0.8056 |
| F0v2 + F3c | 0.9566 | 0.2465 | 0.9367 | 0.3064 | 1.0000 | 0.8333 |
| F0v2 + F2a + F3c | **0.9714** | **0.1893** | 0.9366 | **0.2374** | 1.0000 | 0.8056 |

S03 selects **F0v2+F2a+F3c** by the frozen macro-F1 rule. On S04 that arm scores 0.9375 accuracy and 0.1174 Brier over 144 trials. Against F0v2 alone, it corrects six previously wrong trials and creates none; on S03 it corrects seven and creates one. Its S04 open-hand recall is 36/36, and pinch recall is 29/36. The selected arm's S04 macro-F1 exceeds previous F0 by 0.0298, but this is not an independent confirmatory gain: S04 has already been inspected in earlier studies, all sessions share one date and wearer, and the scoring uses stable intervals rather than online action boundaries.

The [read-back verifier](verify_song.py) checked all 1,420 probability rows, ten arm/split metric groups, native-trial coverage and S03 selection. The previous F0 probabilities exactly match all 284 held-out trial rows from the earlier [v1 Song run](../new_bank_v1/SONG_TRIAL_PREDICTIONS.csv), with maximum absolute error zero. The [results](SONG_RESULTS.json), [probabilities](SONG_TRIAL_PREDICTIONS.csv) and [verification](SONG_VERIFICATION.json) retain full precision and source hashes.

The same F2a/F3c families lowered pooled macro-F1 and worsened log loss on GRABMyo Day3. The divergent Song result is therefore evidence of dataset/device dependence, not a validated universal improvement. A fresh day/wearer with unseen electrode placement and physically recorded event onsets is needed before replacing the live F0+SPD bundle. No current recognizer has been changed.

To reproduce from `emgimu_classifier` with `PYTHONPATH=src;.`, run `python -m benchmarks.new_bank_v2.song_run` and `python -m benchmarks.new_bank_v2.verify_song`. The raw recordings remain outside Git at `E:/qxy/emg_meta/emg_meta/data/Song`.
