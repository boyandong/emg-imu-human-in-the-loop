# UniBo complete-bout G5 versus DTW replay assets

The validation directory preserves the frozen Days 1–5 personal-source models,
Day-5 probability calibration, and Day-6 predictions. The final directory
preserves the Days 7–8 predictions and native bout identities. Raw EMG is not
included; download and preprocess the official UniBo dataset with this project
to obtain `unibo_benchmark_20260915/trials`.

From `emgimu_classifier`, with `PYTHONPATH=src;.`, independently replay the
final result using the preserved states and native trials:

```powershell
python -m emgimu.feature_bank.unibo_sequence_temporal `
  D:/emg-imu-benchmarks/data/processed/unibo_benchmark_20260915 `
  feature_bank/replay_assets/unibo_sequence_temporal_final_20260928 `
  --final-parent feature_bank/replay_assets/unibo_sequence_temporal_validation_20260916 `
  --replay
```

The command loads native final recordings, checks trial hashes and source
artifact hashes, and compares every saved G5/DTW probability without fitting.
Both methods use complete bouts with oracle boundaries. Earlier project
experiments examined Days 7–8, so the additional final comparison is
descriptive confirmation rather than a newly untouched holdout.
