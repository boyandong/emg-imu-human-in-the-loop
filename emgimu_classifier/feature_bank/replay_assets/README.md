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

## Reconstructed RLCS by Personal Anchor

The three `feature_bank_epn_rlcs_anchor_*_20260928` directories preserve the
source-user model, source OOF predictions, exact target calibration/evaluation
trial IDs, four-arm scores, paired errors, and target prediction arrays. The
source model uses public EPN612 users 1–15 only; the validation and final
directories use users 16–18 and 19–21 respectively. The protocol was committed
before scoring. The method is newly reconstructed and is not historical RLCS.

With the public EPN612 archive and the existing frozen F0 source run present,
replay one phase from `emgimu_classifier`:

```powershell
$env:PYTHONPATH = 'src;.'
python benchmarks/reconstructed_rlcs_anchor_interaction.py verify `
  D:/emg-imu-benchmarks/data/raw/epn612/EMG-EPN612-Dataset.zip `
  feature_bank/replay_assets/feature_bank_epn_rlcs_anchor_source_20260928 `
  D:/emg-imu-benchmarks/data/processed/feature_bank_epn_probability_final_20260915 `
  feature_bank/replay_assets/feature_bank_epn_rlcs_anchor_final_20260928 final
```

Use the corresponding `validation` directories and phase to replay users
16–18. Both phases recompute every target probability from native trials and
saved population states; personal anchors use only the saved calibration
trial IDs. The EPN final users had already been examined by earlier studies.
