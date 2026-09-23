# Song real 8-channel exploratory study

The user supplied one participant's four HDF5 v3 sessions, recorded on 2026-09-18 with 8-channel EMG at nominal 250 Hz and 6-axis IMU at nominal 112 Hz. Raw files remain at `E:/qxy/emg_meta/emg_meta/data/Song/` and are not redistributed. The script verifies every HDF5 file against its collection-readiness SHA-256 before use. Full machine-readable counts and confusion matrices are in [RESULTS.json](RESULTS.json).

The collection-assigned split was held fixed: S01/S02 train, S03 validation, S04 final test. We used only valid, completed formal trials and up to three non-overlapping 200 ms windows wholly inside each recorded stable interval. Model fitting and feature standardization used S01/S02 only. We selected the four-hand-state feature configuration by S03 trial-level macro-F1, then evaluated S04 once. This selects F0 local EMG detail alone over F0+scale pattern and F0+IMU. A separately specified 28-state endpoint used F0+IMU without test-set model selection. Prediction probabilities were averaged within each trial before scoring; a trial counts once.

| Endpoint | Validation S03 | Final test S04 |
| --- | ---: | ---: |
| Four hand states, accuracy / macro-F1 | 96.4% / 96.5% (140 trials) | 90.3% / 90.0% (144 trials) |
| 28 hand×arm states, accuracy / macro-F1 | 70.0% / 61.2% (140 trials) | 64.6% / 53.1% (144 trials) |

On S04, four-state recall was 33/36 fist, 27/36 index pinch, 34/36 neutral, and 36/36 open hand. This suggests the earlier “almost always rest, rarely open” behavior is not inherent to every classifier on these recordings; it does not prove the existing live UniBo pipeline has been repaired. The offline study operates on cued stable segments and uses zero-phase filtering; it does not reproduce live continuous decoding, transition periods, online calibration, or end-to-end latency.

The actual pre-formal calibration blocks also support a paired 0/1/2-shot
session-calibration test ([CALIBRATION_RESULTS.json](CALIBRATION_RESULTS.json)).
Each shot uses one neutral, pinch, fist and open block; each contributes three
200 ms windows. The source model uses S01/S02 only. S03 chooses the strength
of neutral feature recentering and personal prototype mixing separately for
each budget, before testing S04. Both budgets selected zero recentering and a
0.5 personal-prototype mixing weight. All 144 S04 formal trials remain test
trials because calibration comes from distinct earlier protocol blocks.

| S04 four-state endpoint | 0 shot | 1 shot | 2 shots |
| --- | ---: | ---: | ---: |
| Trial accuracy | 90.3% | 93.8% | 93.8% |
| Trial macro-F1 | 90.0% | 93.6% | 93.6% |
| Changed decisions vs zero shot | — | 5 corrected, 0 new errors | 5 corrected, 0 new errors |

The paired macro-F1 gain is 3.56 percentage points. A descriptive 4,000-draw
class-stratified trial bootstrap gives a 95% interval of 0.72–7.00 points;
the exact paired accuracy discordance test gives p=0.0625. The gain is
promising but uncertain in this single-session test. A second shot provided
no additional benefit. The selected cue durations sum to 14/28 seconds, while
the actual S04 spans from the first to last selected block were 26.0/63.99
seconds because other protocol blocks occurred between them. Neither measure
includes device preparation or operator time, so these are not product
onboarding times.

We repeated the full source fit, S03 selection and S04 evaluation with causal
continuous 40 Hz high-pass and 50/100 Hz notch filters. A prefix-invariance
test confirms future samples cannot alter earlier filtered outputs. This is
still **offline scoring of cued stable intervals**, not live continuous
recognition. The causal [classification results](CAUSAL_RESULTS.json) and
[calibration results](CAUSAL_CALIBRATION_RESULTS.json) are:

| Causal S04 endpoint | 0 shot | 1 shot | 2 shots |
| --- | ---: | ---: | ---: |
| Four-state trial accuracy | 91.0% | 91.7% | 91.0% |
| Four-state trial macro-F1 | 90.7% | 91.3% | 90.6% |
| Corrected / new errors vs zero | — | 2 / 1 | 2 / 2 |

Causal 28-state trial accuracy/macro-F1 are 68.1%/54.9% with the fixed
F0+IMU design. The one-shot causal macro-F1 difference versus zero is only
+0.57 points, with a descriptive paired 95% bootstrap interval of -1.70 to
+3.28 points. Two-shot is slightly worse. Thus the larger zero-phase
calibration improvement is not robust to this necessary real-time-compatible
signal-processing change. The difference itself is one-session evidence and
does not isolate a physiological cause.

S01–S03 each failed collection readiness. The study deliberately uses their individually valid trials for exploratory analysis under the user's instruction; it does not satisfy the formal data-collection acceptance gate. S04 passed readiness. S01/S02/S03/S04 yielded 143/142/140/144 usable formal trials respectively. All data come from one participant on one date. Cue labels have `raw_cues_only` semantics, rather than verified physiological movement onsets. There is no cross-participant or cross-day generalization estimate, clinical claim, or deployment approval.

Reproduce locally from the classifier directory with `PYTHONPATH=src python benchmarks/song_real8_study.py --source E:/qxy/emg_meta/emg_meta/data/Song --output benchmarks/song_real8/RESULTS.json`. On PowerShell, set `$env:PYTHONPATH='src'` before calling Python. The script prints per-session and feature-family progress. Unit checks: `PYTHONPATH=src;. python -m unittest tests.test_song_real8_study -v` (PowerShell syntax for the environment variable).

For calibration reproduction, use `PYTHONPATH=src;. python benchmarks/song_calibration_study.py --source E:/qxy/emg_meta/emg_meta/data/Song --output benchmarks/song_real8/CALIBRATION_RESULTS.json` on PowerShell. The scripts verify the original HDF5 hashes and reject incomplete calibration blocks or overlap with formal data.
Pass `--filter-mode causal` to either script and use the matching `CAUSAL_*.json` output filename to reproduce the causal comparison.

The S01/S02 causal F0 model can now be exported as a small hash-checked local
JSON bundle for the collection application's realtime page. The bundle is
generated by `benchmarks/export_song_live_model.py` and remains outside Git
under the application's ignored `models/song_real8_f0/` directory. The
[runtime replay audit](LIVE_EXPORT_REPLAY_AUDIT.json) rebuilt the source
classifier and checked all 416 S03 windows (maximum probability discrepancy
`1.41e-7`), then fed the first 100 seconds of S04 in 37-sample chunks; all
999 emitted 200 ms windows matched the offline causal filter exactly. A Qt
offscreen test loaded a schema-conforming synthetic bundle and displayed a probability
frame. This verifies the preprocessing/model transfer, not the USB device,
continuous event scoring or latency.
In the local CPU replay, 37-sample chunks took 0.58 ms at the 95th percentile
(0.97 ms maximum), compared with 148 ms of nominal sample time per chunk.
These are processing times only; acquisition, UI queueing and output data age
remain unmeasured.
