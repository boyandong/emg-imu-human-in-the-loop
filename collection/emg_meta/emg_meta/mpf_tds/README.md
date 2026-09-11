# EMGForce MPF+TDS

Independent single-participant implementation reconstructed from the MPF+TDS architecture described in Kaifosh and Reardon (2025). MPF is numerically checked against Meta's published `MultivariatePowerFrequencyFeatures`; Meta did not publish the single-participant TDS source or checkpoints.

Train with:

```bash
python -m mpf_tds.train --data-root /home/qxy/qxy/emg_data/emgforce_dataset --epochs 300 --batch-size 4
```

The optimizer follows the paper: Adam, five-epoch linear warm-up to `1e-3`, one-time decay to `5e-4` after epoch 25, and 300 epochs. Checkpoints are selected by the lowest validation mean per-class FNR at the paper's fixed `0.35` threshold.
