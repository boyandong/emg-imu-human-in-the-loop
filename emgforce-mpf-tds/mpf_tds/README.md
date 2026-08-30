# EMGForce MPF+TDS

Independent single-participant implementation reconstructed from the MPF+TDS architecture described in Kaifosh and Reardon (2025). The default preset accepts 200 Hz, 8-channel input; a legacy 2 kHz MPF preset remains available for paper/Meta-compatible data. Meta did not publish the single-participant TDS source or checkpoints.

The model predicts four discrete events: index press/release and middle-finger press/release.

## User-paced continuous collection

Use `ContinuousGestureRecorder` from `mpf_tds.collector`. It continuously
appends `[samples, 8]` EMG blocks to HDF5, detects activity with adaptive
high/low thresholds, and assigns onset timestamps in the fixed order
`index_press`, `index_release`, `middle_press`, `middle_release`. There is no
action deadline and no completion key. See the repository-level README for the
integration example.

Train with:

```bash
python -m mpf_tds.train --data-root /home/qxy/qxy/emg_data/emgforce_dataset --sample-rate-hz 200 --epochs 300 --batch-size 4
```

The optimizer follows the paper: Adam, five-epoch linear warm-up to `1e-3`, one-time decay to `5e-4` after epoch 25, and 300 epochs. Checkpoints are selected by the lowest validation mean per-class FNR at the paper's fixed `0.35` threshold.
