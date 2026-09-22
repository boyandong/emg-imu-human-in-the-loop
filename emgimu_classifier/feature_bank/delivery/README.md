The five CSVs provide the required delivery schemas. Original heterogeneous result
tables remain unchanged in `../results`. Each delivery row retains run ID, source
artifact, one-based source record index and a hash of the full original record.
`metadata_notes_json` identifies alias mappings, manifest-derived metadata and missing
evidence. `N/A` is explicit missingness, not zero performance.

Regenerate and verify from the classifier directory:

```powershell
python benchmarks/canonical_delivery.py feature_bank/results feature_bank/delivery
python benchmarks/canonical_delivery.py feature_bank/results feature_bank/delivery --verify
```

`SCHEMA_AUDIT.json` lists all missing required fields by artifact and run.
`PROVENANCE_AUDIT.json` checks every source record and unchanged recorded value.
For the two archived held-out prediction matrices, regenerate recovered
prediction-disagreement rates before rebuilding the canonical tables:

```powershell
python benchmarks/recover_prediction_disagreement.py feature_bank/results D:/emg-imu-benchmarks/data/processed feature_bank/results/prediction_disagreement_recovery.json
```

The old source field `disagreement` records correctness disagreement, so it is
never aliased to prediction disagreement. The recovery artifact binds 102
rates to source rows and archived prediction hashes. The remaining 300 such
rows keep `N/A` until actual prediction arrays are replayed.
The current evidence status is partial: schema presence and record provenance do not
prove that every requested experiment is complete. Unsupported budgets retain N/A
metrics. Reused training states from another phase never supply its target domain.
