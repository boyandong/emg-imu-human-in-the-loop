# Song 28-state source model: window bundle and replay

The [exporter](export_song_joint28_window.py) refits the frozen source-only
S01+S02 hand and arm branches from the four hashed Song HDF5 sessions. It saves
the small, inspectable JSON [model and manifest](../../../collection/emg_meta/emg_meta/model_assets/song_joint28_window/)
for the independent [collection runtime](../../../collection/emg_meta/emg_meta/emgforce/inference/song_joint28_local.py).
Neither the model nor this replay uses S03/S04 windows or labels for fitting.

The window runtime accepts **already causally filtered** 50×8 EMG windows and matched
22×6 native IMU windows, at 250 and 112 Hz respectively. A separate causal
stream wrapper now waits for an IMU index beyond each EMG window end before
selecting the final 22 prior IMU samples. Its chunk-invariance test compares
every emitted synthetic frame to independent whole-stream filtering. The
collection page exposes it as a **manually selected experimental option** and
the offscreen test feeds both sensors through the Qt worker. The controller
passes a global EMG boundary with each IMU packet batch, using the same
batch-level alignment convention as the HDF5 recorder. It outputs four hand,
seven arm and 28 joint window probabilities, with the same frozen class order as
the native-trial experiment. Source-OOF temperatures selected for trial-averaged
probabilities are intentionally absent from window inference: their quality has
not been evaluated on continuously emitted window probabilities.

The exporter recomputes the reference feature matrices and classifiers, then
compares the independent runtime on **every S03 and S04 held-out stable window**.
Its [read-back audit](../../../collection/emg_meta/emg_meta/model_assets/song_joint28_window/song_joint28_replay_audit.json)
records 416+431 windows and 140+144 trial aggregates: maximum feature error 0,
maximum branch probability error 2.76e-7, maximum frozen trial probability error
1.18e-7. Model, runtime and source hashes are saved there. The tests exercise
the tracked artifact's class mapping, normalization, malformed inputs and hash
tamper rejection.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.` and the recorded
Python 3.13.9 scientific runtime:

```powershell
D:/miniconda/python.exe -m benchmarks.new_bank_v2.export_song_joint28_window
```

The online label holds only after three consecutive 28-state frames exceed
the current UI threshold (initially 0.15). That threshold and rule are an
untuned display control, not a validated event detector. This is an exact
offline model-export check plus a software-path smoke test, **not** evidence of
continuous recognition quality, end-to-end latency, physical-device
synchronization or cross-day/person performance. New physical-device
recordings are needed before making those claims.
