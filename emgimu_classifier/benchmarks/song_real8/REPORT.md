# Song real 8-channel exploratory study

The user supplied one participant's four HDF5 v3 sessions, recorded on 2026-09-18 with 8-channel EMG at nominal 250 Hz and 6-axis IMU at nominal 112 Hz. Raw files remain at `E:/qxy/emg_meta/emg_meta/data/Song/` and are not redistributed. The script verifies every HDF5 file against its collection-readiness SHA-256 before use. Full machine-readable counts and confusion matrices are in [RESULTS.json](RESULTS.json).

The collection-assigned split was held fixed: S01/S02 train, S03 validation, S04 final test. We used only valid, completed formal trials and up to three non-overlapping 200 ms windows wholly inside each recorded stable interval. Model fitting and feature standardization used S01/S02 only. We selected the four-hand-state feature configuration by S03 trial-level macro-F1, then evaluated S04 once. This selects F0 local EMG detail alone over F0+scale pattern and F0+IMU. A separately specified 28-state endpoint used F0+IMU without test-set model selection. Prediction probabilities were averaged within each trial before scoring; a trial counts once.

| Endpoint | Validation S03 | Final test S04 |
| --- | ---: | ---: |
| Four hand states, accuracy / macro-F1 | 96.4% / 96.5% (140 trials) | 90.3% / 90.0% (144 trials) |
| 28 hand×arm states, accuracy / macro-F1 | 70.0% / 61.2% (140 trials) | 64.6% / 53.1% (144 trials) |

On S04, four-state recall was 33/36 fist, 27/36 index pinch, 34/36 neutral, and 36/36 open hand. This suggests the earlier “almost always rest, rarely open” behavior is not inherent to every classifier on these recordings; it does not prove the existing live UniBo pipeline has been repaired. The offline study operates on cued stable segments and uses zero-phase filtering; it does not reproduce live continuous decoding, transition periods, online calibration, or end-to-end latency.

S01–S03 each failed collection readiness. The study deliberately uses their individually valid trials for exploratory analysis under the user's instruction; it does not satisfy the formal data-collection acceptance gate. S04 passed readiness. S01/S02/S03/S04 yielded 143/142/140/144 usable formal trials respectively. All data come from one participant on one date. Cue labels have `raw_cues_only` semantics, rather than verified physiological movement onsets. There is no cross-participant or cross-day generalization estimate, clinical claim, or deployment approval.

Reproduce locally from the classifier directory with `PYTHONPATH=src python benchmarks/song_real8_study.py --source E:/qxy/emg_meta/emg_meta/data/Song --output benchmarks/song_real8/RESULTS.json`. On PowerShell, set `$env:PYTHONPATH='src'` before calling Python. The script prints per-session and feature-family progress. Unit checks: `PYTHONPATH=src;. python -m unittest tests.test_song_real8_study -v` (PowerShell syntax for the environment variable).
