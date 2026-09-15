# Public sEMG benchmark inventory

This inventory separates benchmarks already used by this repository from newly selected
failure tests. Raw data lives outside Git at `D:\emg-imu-benchmarks\data\raw`; processed
data and manifests use sibling directories. Raw archives and extracted data are immutable.

| Dataset | Status | Existing adapter | Existing results | Expected local path | Download status |
|---|---|---|---|---|---|
| Historical DS2 force | historical | no | referenced by the supplied research brief, absent from this repository and Git history | `data/raw/ds2_force` | source identified; Kaggle access pending |
| LibEMG Contraction Intensity | new | native CSV adapter | Feature Bank, calibration, ablation | `data/raw/libemg_force` | verified complete |
| LibEMG Electrode Shift | new | native ZIP adapter | held-out before/after | `data/raw/libemg_electrode_shift` | verified complete |
| UniBo-INAIL | historical | yes | prior 33-run ablation; Feature Bank chronological | `data/raw/unibo_inail` | verified complete |
| EMG-EPN612 | new | labelled JSON adapter | cross-user calibration and late fusion | `data/raw/epn612` | verified complete |
| sEMG-MANUS | new | native ZIP adapter | session/speed and calibration | `data/raw/semg_manus` | verified complete |
| EMG-FMG load and limb position | new | EMG-only ZIP adapter | load/position held-out study | `data/raw/emg_fmg` | verified complete |
| GREAT | new | no | no | `data/raw/great` | secondary, deferred until Tier 1 validation |
| NinaPro DB6 | new | no | no | `data/raw/ninapro_db6` | secondary; NinaPro account and terms required |
| GRABMyo | new | metadata entry only | no | `data/raw/grabmyo` | secondary; 9.4 GB uncompressed |
| NinaPro DB5 | new | generic 16-to-two-8-channel policy | no | `data/raw/ninapro_db5` | secondary; NinaPro account and terms required |
| FORS-EMG | new | no | no | `data/raw/fors_emg` | supplementary; original release is Kaggle |
| Three-position electrode replacement | new | no | no | `data/raw/electrode_replacement` | secondary; public Zenodo archive |
| Hyser | new | no | no | `data/raw/hyser` | metadata only; full collection is about 143 GB |

The repository search found no files, commits, configurations or reports identifying the
historical DS2 source. The supplied brief matches the 2025 DS2 publication exactly: 20
subjects, 3 channels at 1500 Hz, five gestures, and subjective low/medium/high effort.
It is therefore recorded as the likely source, but old-result reproduction remains blocked
until its archive identity is matched to the historical experiment artifacts.
