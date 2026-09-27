# Song 28-state continuous replay after live-page integration

The [protocol](SONG_JOINT28_CONTINUOUS_PROTOCOL.json) and initial runner were
committed as `7ec14a6` before this full-stream result. The exported S01+S02
source-only model, 0.15 event threshold and three-frame persistence rule are
unchanged. The only post-run runner correction permits valid trials that have
no full window on the continuous global 25-sample grid; they are named below
instead of being silently counted as scored. The read-back additions report
class recall and the location of state changes without tuning the model.

Both complete S03 and S04 raw recordings were replayed with continuous causal
filter state and their recorded 112 Hz IMU-to-EMG indices. A frame is released
after an IMU watermark passes its EMG end index. This tests the software path
on real recorded signals, including prompt and transition periods; it is not
a real-time execution or device-clock latency measurement.

| Endpoint | S03 | S04 |
|---|---:|---:|
| Emitted frames / dropped by indexed IMU guard | 13,444 / 0 | 11,916 / 0 |
| Labelled formal stable windows | 895 | 911 |
| Stable-window joint accuracy / macro-F1 | 0.6078 / 0.5426 | 0.5928 / 0.4901 |
| Scored native formal trials | 139 of 141 | 144 of 144 |
| Trial mean-joint-probability accuracy / macro-F1 | 0.7122 / 0.6566 | 0.7083 / 0.6127 |
| Stable-window hand / arm accuracy | 0.9039 / 0.6682 | 0.8913 / 0.6334 |
| Explicit rest-block active-hand argmax frames | 2 of 144 | 0 of 144 |
| Explicit rest-block active-hand **display** frames | 0 of 144 | 0 of 144 |
| Non-null state transitions across entire recording | 883 | 862 |

S03 trial 79 (`up_fist`, 20 stable samples) cannot contain a 50-sample
window; trial 121 (`up_fist`, 59 stable samples) has none on the fixed global
25-sample-hop grid. Both are excluded from the 139-trial result with their
identities recorded. The earlier trial study used trial-relative window
starts and different probability aggregation, so its 140-trial S03 and
144-trial S04 numbers are context, not an exact same-window baseline.

Within labelled stable windows, `open_hand` recall is 0.9457 on S03 and
0.9383 on S04; `fist` recall is 0.9326 and 0.9204. The arm branch is much
weaker for `left` (0.3158, 0.2105) and `backward` (0.4342, 0.2740). These
same-person/day cue results suggest the joint-label bottleneck is arm
direction, not a complete inability to recognize the hand-open signal in the
recordings. They do not explain the user's prior UniBo live experience or
predict accuracy after re-donning the device.

Of the 883/862 non-null transitions, 777/756 occur outside the explicitly
scored formal-stable and rest-stable intervals, 103/101 in formal-stable
intervals, and 3/5 in rest-stable intervals. Most raw recording time has no
verified continuous action label. These transitions therefore **cannot be
called false positives**. The two explicit rest blocks per session support
only the narrow zero active-hand display observation above. No physiological
onset/offset or operating-time accuracy is inferred from prompt timestamps.

The [frame table](SONG_JOINT28_CONTINUOUS_FRAMES.csv) stores all 25,360
emitted frame indices, cue-interval assignment, peak and displayed label,
event flag, and full probabilities only for scored formal windows. The
[read-back verifier](verify_song_joint28_continuous.py) checks source/model/
protocol hashes, the complete global stream grid, cue assignment against
native HDF5 intervals, every saved online state transition, all reported
metrics and trial-average predictions. It produced the saved
[verification](SONG_JOINT28_CONTINUOUS_VERIFICATION.json) with no discrepancy.
Raw HDF5 data remain outside Git.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.` and the recorded
scientific Python runtime:

```powershell
D:/miniconda/python.exe -m benchmarks.new_bank_v2.song_joint28_continuous_replay
D:/miniconda/python.exe -m benchmarks.new_bank_v2.verify_song_joint28_continuous
```

The model remains an opt-in experimental UI choice. The next acceptance
evidence must come from a physical connected recording with independent
action annotations, device synchronization and end-to-end latency measurement,
then new-day and new-person sessions. Existing one-person/day data cannot
establish those properties.
