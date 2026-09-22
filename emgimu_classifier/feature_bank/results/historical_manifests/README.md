# Superseded experiment manifests

These 18 JSON files were preserved byte for byte from local commit `ffb3999` after the active-results consolidator stopped listing their runs. `ARCHIVE_AUDIT.json` records each original Git path, source revision, size and SHA-256.

They are historical protocol evidence, outside the active `results/manifests` namespace. The current result tables, their integrity checks and the reproducibility metadata inventory do not include them. A past manifest can explain a historical run but does not validate its score or replace the missing original DS2 experiments.

Regenerate and verify this archive from the repository root with:

```powershell
python emgimu_classifier/benchmarks/archive_historical_manifests.py . emgimu_classifier/feature_bank/results/manifests emgimu_classifier/feature_bank/results/historical_manifests
```
