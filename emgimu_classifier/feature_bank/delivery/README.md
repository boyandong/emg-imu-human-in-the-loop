The five CSVs provide the required delivery schemas. Original heterogeneous result
tables remain unchanged in `../results`. Each delivery row retains run ID, source
artifact, one-based source record index and a hash of the full original record.
`metadata_notes_json` identifies alias mappings, manifest-derived metadata and missing
evidence. `N/A` is explicit missingness, not zero performance.

With the archived raw datasets and processed source runs available, regenerate
the prediction-disagreement recovery from the classifier directory:

```powershell
python benchmarks/recover_prediction_disagreement.py feature_bank/results D:/emg-imu-benchmarks/data/processed feature_bank/results/prediction_disagreement_recovery.json --raw-root D:/emg-imu-benchmarks/data/raw
```

Then regenerate and verify the canonical tables:

```powershell
python benchmarks/canonical_delivery.py feature_bank/results feature_bank/delivery
python benchmarks/canonical_delivery.py feature_bank/results feature_bank/delivery --verify
```

`SCHEMA_AUDIT.json` lists all missing required fields by artifact and run.
`PROVENANCE_AUDIT.json` checks every source record and unchanged recorded value.
The old source field `disagreement` records correctness disagreement, so it is
never aliased to prediction disagreement. The recovery artifact binds all 402
legacy rates to source rows and input hashes: 102 from archived predictions,
300 from frozen-state replay on the original target trial splits. No model was
fitted during recovery.
The current evidence status is partial: schema presence and record provenance do not
prove that every requested experiment is complete. Unsupported budgets retain N/A
metrics. Reused training states from another phase never supply its target domain.

The versioned Song source run under `../source_runs/feature_bank_song_real8_spd_delivery_20260924`
adds four zero-shot family rows, two F0→F0+SPD increments, two paired error rows
and ten calibration-curve rows. Six curve rows separately describe the
source-F0 plus personal-SPD-anchor method at 0/1/2 shots; 5-shot is unsupported.
Its exporter reads the saved Song SPD
study and trial-reliability audit; it does not retrain a model or copy raw EMG.
It is one participant on one date with S03 validation and previously inspected
S04 final data. Rebuild that run with `benchmarks/export_song_real8_delivery.py`
and then consolidate using the original processed root plus the workspace's
`work/benchmark_runs` as `--local-root`. The consolidation script also reads
the repository's versioned Song source run. Existing source records are unchanged.
