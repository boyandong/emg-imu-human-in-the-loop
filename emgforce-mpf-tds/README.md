# EMGForce MPF+TDS

Independent single-participant implementation reconstructed from the MPF+TDS architecture described in Kaifosh and Reardon (2025). The legacy 2 kHz MPF preset is numerically checked against Meta's published `MultivariatePowerFrequencyFeatures`; the default preset supports 200 Hz, 8-channel input with six bands below its 100 Hz Nyquist frequency. Meta did not publish the single-participant TDS source or checkpoints.

The model predicts four discrete events: index press/release and middle-finger press/release.

## User-paced continuous collection

`ContinuousGestureRecorder` stores every incoming EMG sample immediately and
marks activity onsets without requiring a space-bar confirmation or a fixed
action deadline.  Labels advance in the fixed order `index_press`,
`index_release`, `middle_press`, `middle_release` only after the signal has
returned to rest.

```python
from mpf_tds.collector import ContinuousGestureRecorder

recorder = ContinuousGestureRecorder("session.hdf5")
recorder.start()

# Call this for every block received from the acquisition device.  `samples`
# has shape [samples, 8].  Supplying device timestamps is strongly preferred.
events = recorder.append(samples, timestamps)
for event in events:
    print(event.name, event.onset_time, "next:", recorder.expected_label)

# Optional recovery from a movement artifact; this does not delete raw EMG.
recorder.undo_last_event()
recorder.stop()
```

The first three seconds after `start()` are resting-baseline calibration.  Input
defaults to 200 Hz and must already use the same `emg_8ch_200hz_v1`
preprocessing expected by training.  The
result keeps continuous samples in `/data`, compatible `(name, time)` onset
labels in `/prompts`, and detailed detector events in `/collection_events`.
New recordings declare `prompt_time_reference=emg_onset`, so training does not
apply the legacy 100 ms cue-response shift.  Older cue-timed datasets retain the
shift automatically.

Train with:

```bash
python -m mpf_tds.train --data-root /home/qxy/qxy/emg_data/emgforce_dataset --sample-rate-hz 200 --epochs 300 --batch-size 4
```

The optimizer follows the paper: Adam, five-epoch linear warm-up to `1e-3`, one-time decay to `5e-4` after epoch 25, and 300 epochs. Checkpoints are selected by the lowest validation mean per-class FNR at the paper's fixed `0.35` threshold.
