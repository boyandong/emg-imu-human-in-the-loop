# Continuous replay of the signed device-axis arm candidate

The [fixed protocol](SONG_ARM_SIGNED_CONTINUOUS_PROTOCOL.json) was committed as `68469c8` before scoring. It compares the source-only signed-axis arm candidate with the already frozen [28-state continuous baseline](SONG_JOINT28_CONTINUOUS_REPORT.md) on the complete S03/S04 recordings. Both use the same causal EMG filter, native indexed IMU watermark, 50-sample EMG and 22-sample IMU window, global 25-sample hop, 0.15 event threshold, three-frame persistence and cue-interval scoring. The tracked baseline hand model is unchanged. The candidate's seven-way arm classifier is fitted only on S01/S02, with the single feature and hyperparameter choice frozen by the [trial study](SONG_ARM_SIGNED_REPORT.md).

Before stream scoring, the refitted candidate reproduces all saved trial arm probabilities within `3.33e-16`; on formal stream windows, its hand marginal matches the tracked baseline within `6.66e-16`. Every emitted frame is paired by session and EMG end index with the frozen baseline and has identical cue-interval assignment. Both runs emit 25,360 frames with zero indexed-IMU drops.

| Endpoint | S03 baseline | S03 signed | S04 baseline | S04 signed |
|---|---:|---:|---:|---:|
| Stable-window 28-state macro-F1 | 0.5426 | **0.6064** | 0.4901 | **0.5629** |
| Mean-joint-probability formal-trial macro-F1 | 0.6566 | **0.7581** | 0.6127 | **0.7399** |
| Stable-window arm accuracy | 0.6682 | **0.7073** | 0.6334 | **0.6883** |
| Stable-window arm `left` recall | 0.3158 | **0.5000** | 0.2105 | **0.4737** |
| Stable-window arm `backward` recall | 0.4342 | **0.5395** | 0.2740 | **0.5616** |
| Explicit rest active-hand displays | 0/144 | 0/144 | 0/144 | 0/144 |
| Non-null state transitions over whole recording | 883 | 873 | 862 | 864 |

The positive trial and stable-window results agree in direction, though their values differ from the previous trial-relative study because the continuous stream uses a fixed global frame grid. S03 has 139 scored formal trials out of 141; the two excluded trials cannot contain a full global-grid window. S04 has 144/144. Only the two explicit stable rest blocks per session have verified rest labels. Most of each full recording lacks independently verified continuous action labels, so the state transitions outside scored intervals cannot be called false positives. Zero displayed active-hand states in these 288 rest frames does not prove low false-alarm rate during an ordinary session.

The [model coefficients](SONG_ARM_SIGNED_ARM_MODEL.json), [full frame table](SONG_ARM_SIGNED_CONTINUOUS_FRAMES.csv), [results](SONG_ARM_SIGNED_CONTINUOUS_RESULTS.json), and [read-back verifier](verify_song_arm_signed_continuous.py) are saved. The verifier checks artifact hashes, all 25,360 native frame identities and cue assignments, unchanged hand marginals, 28-class probabilities, state transitions, explicit rest counts, trial aggregation and all reported scores. Its [verification](SONG_ARM_SIGNED_CONTINUOUS_VERIFICATION.json) passed. Focused scientific tests pass (8 tests), and existing collection/runtime tests pass (5 tests with a fresh temporary directory).

This remains a one-person, one-day software replay. S01–S03 did not pass formal collection readiness; S04 was previously inspected. Device axes are not a calibrated body frame. Replay CPU time is not hardware latency or end-to-end UI latency. No live model has been replaced. A fresh physical-session evaluation with independent action annotations, measured synchronization and latency is still needed before deciding whether this arm candidate should become the default recognizer.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.`:

```powershell
D:/miniconda/python.exe -m benchmarks.new_bank_v2.song_arm_signed_continuous
D:/miniconda/python.exe -m benchmarks.new_bank_v2.verify_song_arm_signed_continuous
```
