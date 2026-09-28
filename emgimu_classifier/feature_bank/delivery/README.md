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

The reconstructed RLCS/PersonalAnchor runs append 96 four-arm scores, 72
conditional increments, 24 paired-error rows and 24 interaction rows. The
source-user classifier and temperature were fixed before validation and reused
for descriptive final users. The label `RLCS_reconstructed_v1` distinguishes
this new method from the unavailable historical RLCS implementation. Frozen
states, prediction arrays, exact trial splits and replay audits are retained
under `../replay_assets/feature_bank_epn_rlcs_anchor_*_20260928`.

The public DS2 v9 personal-calibration study adds 240 pooled and per-subject
rows to `calibration_curve.csv`; its matched frozen-prediction analysis adds
200 conditional-increment rows and 400 error-complementarity rows. Its exporter
`../../benchmarks/export_public_ds2_calibration_delivery.py` reads only frozen,
independently verified scores, checks the source hashes and can be rerun without
duplicating rows. Force-ZeroShot calibration uses low/average force only;
ProductMode reserves all five calibration trials per gesture before any budget.
These three-channel public scores are descriptive and do not replace own-device
validation or the unavailable historical X1-H experiment.
The existing 402 legacy disagreement recoveries retain their original row
indices and hashes; the new DS2 rows record disagreement directly.

The new-v2 public eight-channel cross-user intensity analysis adds 84
`conditional_incremental.csv` and 28 `error_complementarity.csv` rows from
matched frozen native-trial predictions. The exporter is
`../../benchmarks/new_bank_v2/export_force_delivery.py`; its SHA-bound audit
is `../../benchmarks/new_bank_v2/FORCE_DELIVERY_AUDIT.json`. Paired
interactions are retained in the standalone force study because the five
canonical schemas have no interaction field. At that export, 51,268 canonical
records passed provenance verification; the scientific evidence status remains
partial.

The separate new-v2 eight-channel force calibration study appends 156
matched 0/1/2-shot pooled, subject and condition rows to
`calibration_curve.csv`. Its exporter is
`../../benchmarks/new_bank_v2/export_force_cal_delivery.py`, and its source
hash audit is `../../benchmarks/new_bank_v2/FORCE_CAL_DELIVERY_AUDIT.json`.
MVC is excluded because two repetitions per gesture cannot support the
fixed two-shot calibration and held-out evaluation split. The current
canonical provenance check covered 51,424 rows at that export.

The single-Medium-reference calibration comparison adds a further 156
matched 0/1/2-shot rows to `calibration_curve.csv`. It reuses one set of
7/14 labelled trials per user across all ten evaluation intensities, with
source results from `../../benchmarks/new_bank_v2/FORCE_REF_CAL_REPORT.md`.
The exporter and SHA audit are
`../../benchmarks/new_bank_v2/export_force_ref_cal_delivery.py` and
`../../benchmarks/new_bank_v2/FORCE_REF_CAL_DELIVERY_AUDIT.json`.
The canonical provenance check covered 51,580 rows at that export.

The new-v2 eight-channel candidate full-bank leave-one-family-out study adds
176 pooled, subject and condition records to `ablation_full_bank.csv`, with
per-class F1 and full-versus-removed F1 deltas. Its independently verified
source study and exporter are
`../../benchmarks/new_bank_v2/LOFO_REPORT.md` and
`../../benchmarks/new_bank_v2/export_lofo_delivery.py`; the SHA audit is
`../../benchmarks/new_bank_v2/LOFO_DELIVERY_AUDIT.json`. The current
canonical provenance check covered 51,756 rows at that export.

The new-v2 Stage-1 screening export adds 256 pooled, subject and condition
rows to `feature_family_results.csv`, including per-class F1 and ten-bin ECE
recomputed from frozen trial probabilities. The corresponding seven-axis
vector explicitly records four unavailable axes as N/A and keeps the
one-person Song results separate. Source files are
`../../benchmarks/new_bank_v2/FAMILY_SCREEN_REPORT.md` and
`../../benchmarks/new_bank_v2/export_family_screen_delivery.py`; the hash
record is `../../benchmarks/new_bank_v2/FAMILY_SCREEN_DELIVERY_AUDIT.json`.
The current canonical provenance check covers 52,012 rows.

The new-v2 matched-error analysis adds 384 exact native-trial arm pairs to
`error_complementarity.csv`. The 64 F2a×F3c interaction cells and the
interpretation boundary are in
`../../benchmarks/new_bank_v2/PAIR_REPORT.md`; independent verification and
source hashes are in `../../benchmarks/new_bank_v2/PAIR_AUDIT.json` and
`../../benchmarks/new_bank_v2/PAIR_DELIVERY_AUDIT.json`. The current
canonical provenance check covers 52,396 rows.

The new-v2 Stage-2 conditional comparison adds 128 source-only records to
`conditional_incremental.csv`. It compares each of F2a and F3c against a
core already containing the other, on the same matched native trials.
The full source result, sign convention, and evidence limits are in
`../../benchmarks/new_bank_v2/CONDITIONAL_REPORT.md`; the deterministic
reproducer and hash records are
`../../benchmarks/new_bank_v2/conditional_analysis.py`,
`../../benchmarks/new_bank_v2/CONDITIONAL_AUDIT.json`, and
`../../benchmarks/new_bank_v2/CONDITIONAL_DELIVERY_AUDIT.json`.
The current canonical provenance check covers 52,524 rows.

The new-v2 GRABMyo same-user score-anchor follow-up adds 288 pooled and
subject calibration-curve cells at 0/1/2/5 shots per class. The full
native-trial split, probability-space method, and limitations are in
`../../benchmarks/new_bank_v2/GRAB_SCORE_CAL_REPORT.md`; the source and
delivery hash records are `../../benchmarks/new_bank_v2/GRAB_SCORE_CAL_AUDIT.json`
and `../../benchmarks/new_bank_v2/GRAB_SCORE_CAL_DELIVERY_AUDIT.json`.
The current canonical provenance check covers 52,812 rows.

The distinct new-v2 GRABMyo **feature-space** anchor follow-up adds 288
more pooled and subject calibration-curve cells. Its fixed 50/50 mixture
mostly worsens held-out log loss, despite occasional F1 gains; see
`../../benchmarks/new_bank_v2/GRAB_FEATURE_CAL_REPORT.md`. Raw-checksum,
trial-split and source-model replay evidence are in
`../../benchmarks/new_bank_v2/GRAB_FEATURE_CAL_AUDIT.json`; output hashes
are in `../../benchmarks/new_bank_v2/GRAB_FEATURE_CAL_DELIVERY_AUDIT.json`.
The current canonical provenance check covers 53,100 rows.

The independent new-v1 matched-pair analysis adds 176 family-screen,
88 conditional, and 44 error-complementarity rows from the original frozen
LibEMG force/wearing trial probabilities. Its 44 scale×frequency and
ring×correlation-spectrum interaction cells stay in the separately
versioned source table. See `../../benchmarks/new_bank_v1/PAIRED_REPORT.md`
and `../../benchmarks/new_bank_v1/PAIRED_DELIVERY_AUDIT.json` for the
native-trial pairing, hashes and interpretation. The current canonical
provenance check covers 53,408 rows.
