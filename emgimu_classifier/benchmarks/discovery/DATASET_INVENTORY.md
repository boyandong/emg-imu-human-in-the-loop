# Public sEMG benchmark inventory

This inventory separates benchmarks already used by this repository from newly selected
failure tests. Raw data lives outside Git at `D:\emg-imu-benchmarks\data\raw`; processed
data and manifests use sibling directories. Raw archives and extracted data are immutable.

| Dataset | Status | Existing adapter | Existing results | Expected local path | Download status |
|---|---|---|---|---|---|
| Historical DS2 force | historical candidate | native adapter pending | referenced by the supplied research brief, absent from this repository and Git history | `work/datasets/historical_ds2_candidate/ds2_kaggle_v8.zip` | publisher-linked Kaggle v8 archive downloaded and CRC verified; identity with old experiment input remains unproven |
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
On 2026-09-23 the publisher-linked Kaggle public API resolved version 8 and
listed 102 files totaling 1,312,583,609 uncompressed bytes. A resumable download
obtained the complete 1,123,505,003-byte archive; all five MAT and 97 TDMS files
pass ZIP CRC. `DS2_ACCESS_AUDIT.json` and `DS2_ARCHIVE_AUDIT.json` preserve the
source, hash, file inventory and CC-BY-4.0 publication versus CC0 Kaggle license
conflict. This version is not yet proven to have produced the historical
B0/X1-H/X2 results.
The extracted public v8 `Data_all_Raw.mat` contains 2,863 finite raw trials of
3 channels × 15,000 samples. A deterministic six-trial sample had no exact
zeros and nonzero RMS on all channels. The separate gesture-window label MAT
has 332,108 labels in five integer classes. [DS2_NATIVE_MAT_AUDIT.json](DS2_NATIVE_MAT_AUDIT.json)
records the source hashes and sample values. It does not establish how raw
trials map to subjects, gestures or force levels; those labels cannot be
joined to the window-label file by row count alone.

A subsequent [complete MAT-window join audit](DS2_MAT_TRIAL_WINDOW_JOIN_AUDIT.json)
recomputed all 996,324 MAV channel values from the 2,863 raw trials using the
published 375-sample window and 75-sample hop after the 3,000-sample rest.
Every value matches the corresponding block of 116 published MAV rows exactly.
The accompanying [trial index](DS2_MAT_TRIAL_WINDOW_JOIN.csv) finds 2,862
uniform 116-window gesture-code blocks; zero-based trial 209 contains 115
windows of code 2 and one of code 3, so it remains ambiguous. This verifies
raw-to-MAV order and supports numerical gesture codes for the uniform blocks;
it does not identify subjects, force levels, TDMS segments, gesture names or
the historical B0/X1-H/X2 input.

The separate [TDMS first-segment inventory](DS2_TDMS_FIRST_METADATA_AUDIT.json)
reads the file-level `name` property from all 97 TDMS members using the
[NI TDMS segment definition](https://www.ni.com/en/support/documentation/supplemental/07/tdms-file-format-internal-structure.html).
Their folder names cover subjects `01`–`20`; filename movement indices 1–4
appear for all 20, while index 5 appears for 18. Subjects `01` and `02` lack
an Mv5 filename, and subject `04` has one combined `Mv4_Mv5` member.
File-level names contain gesture clues such as `Puño`, `Tacita`, `Meñique`,
`Extension` and `Reposo`, but include inconsistent spellings and generic
`Prueba1`–`Prueba4` names. These first-segment properties neither provide
per-trial force labels nor map the 2,863 aggregate MAT trials to subjects.
No labelled DS2 force experiment has been reconstructed from this inventory.

An independent [full TDMS metadata pass](DS2_TDMS_GROUP_AUDIT.json) with
[npTDMS](https://nptdms.readthedocs.io/en/stable/reading.html) traversed all
segments, finding 3,210 named groups with three sensor channels each. Every
group has zero group-level properties; the channel properties describe sensors,
units, scaling and timing rather than gesture/force labels. The waveform
increment is about 0.667 ms, consistent with 1500 Hz. Group lengths range
from 100 to 980,100 samples; only 1,284 have exactly 15,000 samples, while
the aggregate MAT has 2,863 arrays of that length. A positional join of these
different populations would invent subject, gesture and force labels.

A later [exact waveform join](DS2_TDMS_RAW_EXACT_JOIN_AUDIT.json) now maps
2,833 of 2,863 MAT trials to a unique TDMS group and subject folder by matching
all 45,000 values across three channels. The remaining zero-based MAT trials
389–418 have no exact match at group starts or at any possible nonzero offset
in the released TDMS groups. The [join index](DS2_TDMS_RAW_EXACT_JOIN.csv)
records only verified matches; the 30 unmatched trials retain unknown subject
identity. This materially improves public-v8 provenance but supplies neither
per-trial force labels nor proof that v8 was the historical experiment input.
The [subject-held-out gesture-code study](public_ds2_subject_gesture/REPORT.md)
uses the 2,832 uniquely matched trials with uniform labels. Its five fixed
F0/reference-family arms supply new cross-user gesture evidence, but no force
condition analysis or reproduction of the old DS2 baselines.

The [force-annotation source audit](DS2_FORCE_ANNOTATION_SOURCE_AUDIT.json)
checked the publisher-linked Kaggle v8 description, all five MAT variable
inventories and the already parsed TDMS metadata. The release declares three
subjective effort conditions and ten repetitions per gesture per condition,
but provides no verified trial-to-force key. Its description also conflicts
with itself about 120 versus roughly 150 samples per subject and calls 375
samples at 1500 Hz a 375 ms window (the duration is 250 ms). These do not
establish an acquisition order. Gesture and subject joins therefore cannot be
repurposed as low/medium/high labels; force-stratified DS2 experiments remain
unavailable until an authoritative annotation or ordering record is obtained.

The six selected native benchmark archives were also reread byte for byte.
All 15,935,850,342 bytes matched the recorded SHA-256 digests (and recorded
MD5 digests where available); the [fresh digest audit](CORE_ARCHIVE_DIGEST_AUDIT.json)
records each check and file modification time. This verifies archive integrity,
not the original research phase order or any historical DS2 result.
