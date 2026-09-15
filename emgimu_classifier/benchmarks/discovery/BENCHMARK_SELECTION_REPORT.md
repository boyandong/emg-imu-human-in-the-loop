# Benchmark selection report

## Decision

The suite is a failure test bench rather than a dataset leaderboard. Tier 1 covers subjective
and target contraction intensity, electrode shift, day and posture, cross-user calibration,
execution speed, and three multi-factor designs. Large or redundant datasets remain secondary.

## Primary selections

**Force.** Historical DS2 is mandatory for comparability once its exact archive is confirmed.
LibEMG Contraction Intensity is the primary new test because its ramp training and 20 percent
through MVC test levels manipulate intensity directly. EMG-FMG is retained as a distinct
load-by-position interaction: grasped mass is not voluntary contraction level. Hyser offers
measured finger forces but is deferred to subsets because the five collections total roughly
143 GB.

**Wearing.** LibEMG Electrode Shift is primary because it records 21 people before and after
the intended confound with an eight-channel Myo. The three-position Zenodo dataset provides an
independent, controlled replacement check. FORS-EMG varies orientation and coarse placement
region, so it cannot replace either benchmark.

**Users and calibration.** EMG-EPN612 is primary: 612 users, a Myo ring, six relevant classes
and many repetitions support held-out-user and 0/1/2/5-shot protocols. It does not provide a
strong multi-session re-donning protocol.

**Day and posture.** UniBo-INAIL is primary because its 7 subjects by 8 days by 4 postures form
a clean factorial design and the repository already has a verified adapter. Its four named
channels and oracle posture labels limit deployment claims. NinaPro DB6, GRABMyo and GREAT are
independent secondary confirmations with higher download or access cost.

**Speed.** sEMG-MANUS is primary because filenames explicitly encode slow, medium and fast,
with repeated sessions and eight-channel Myo plus IMU. Three incomplete users and three known
trial-count anomalies are preserved in its quality manifest.

## Calibration support and limits

EMG-EPN612 best supports first-use personal calibration. UniBo, sEMG-MANUS, NinaPro DB6 and
GRABMyo support session calibration because the same user returns. Electrode-shift data supports
a targeted wearing calibration. Force-ZeroShot calibration must use source-force trials only;
Force-ProductMode is reported separately when onboarding includes multiple intensities.

No current public benchmark reproduces all properties of the own eight-channel device. The
suite therefore reports a robustness vector and its worst available cell. It never averages
away missing factors or reports synthetic quality corruption as a real sensor failure.

## Download policy

Official GitHub, Zenodo, PhysioNet, Dryad and NinaPro sources are used. Kaggle is accepted only
for Historical DS2 and FORS-EMG because it is the original publication location. Every archive
is downloaded resumably, hashed after completion, and extracted into an immutable raw tree.
Processing, label mapping, resampling and windowing are written to `data/processed`.

## Acquisition evidence, 2026-09-15

Six Tier-1 archives are complete (15,935,850,342 bytes; approximately 15.94 decimal GB combined). Exact sizes and
SHA-256 hashes are in DATASET_MANIFEST.json; official MD5 checks match for the Zenodo releases.
Large EMG-FMG raw CSV expansion would require approximately 52.35 GB, so the adapter streams
the archive. MANUS also streams its archive; UniBo and LibEMG force have immutable extracted
trees. Sanity reports are under `D:/emg-imu-benchmarks/data/manifests/sanity`.
Historical DS2 remains blocked by an unavailable original release (Kaggle page/API 404) and
missing historical experiment artifacts. Public force data is supplementary evidence and
is not silently substituted as historical DS2. Secondary downloads remain deferred rather
than represented as complete. A 14-byte failed UniBo master-branch download is ignored;
the verified main-branch archive is the only source used.

## Reproducible sanity sample audit

`SANITY_AUDIT.json` now records six datasets, each with three subjects randomly selected
without replacement using seed 20260915 and two prespecified conditions. All 36 sampled
recordings are readable, have the expected native EMG channel count and zero measured
NaN/Inf fraction. Each has a raw/envelope/periodogram SVG with its actual subject and
condition caption. The audit includes source-report and plot SHA-256 hashes; plots stay
outside Git beside the raw-data manifests. Historical unreferenced plots remain on disk;
the audit identifies exactly which 36 plots belong to the current sample.

The highest adjacent-equal-sample fraction is 0.2614 in sampled EPN data; Myo quantization
can cause repeated values, so this is not automatically a dead channel. The largest
near-observed-extrema fraction is 0.01159 in the wearing data; it does not prove hardware
clipping. These heuristics are retained for inspection rather than silently labelled clean.
Acquisition rate comes from verified dataset metadata, not a new hardware clock test.
UniBo source files contain many labelled intervals; its five-second plot is not an isolated
gesture trial. This is sample-level QC, not a full-population quality or accuracy guarantee.

## Discovery delivery review, 2026-09-16

Fourteen candidate rows now have transparent A–H scorecards and unweighted reviewed
totals, with rationale and uncertainty in DATASET_CANDIDATES.csv/SCORE_REVIEW.md.
They are retrospective metadata judgements. Original total scores, decisions, reasons
and frozen experiments remain unchanged; the review does not prove scoring preceded
training. Native repetition limitations lower wearing/MANUS calibration scores;
metadata-only secondary datasets are not assigned sampled-QC evidence.

DISCOVERY_DELIVERY_AUDIT.json rechecks all seven required files, candidate fields,
eight-point score vectors, current six-archive sizes, six source sanity-report hashes
and 36 captioned plot hashes. The six-core archive total is 15,935,850,342 bytes.
The original seed/population draw also reproduces three distinct subjects and two
specified conditions for every dataset, including the restricted complete MANUS
cohort used by the native sanity producer. This verifies sample selection evidence,
not a fresh reread of the entire raw population. Archive bytes do not represent
total raw/extracted/processed disk usage. Multi-GB raw digests are the recorded
acquisition digests, not freshly recomputed. DISK_USAGE.json retains its dated
2026-09-15 snapshot and must not be presented as current total disk usage.

The candidate DS2 publication explicitly lists CC-BY4.0, now corrected in the
candidate license field. Its protocol match does not establish the identity of old
DS2 experiments. A fresh primary-source search found the same publisher/Kaggle
reference, but direct publisher fetch was rate-limited and the browser did not open
the dataset URL. This does not establish a new HTTP404 or recover the archive.
The prior404 evidence and missing old artifacts remain separate facts.
Secondary paper/license/layout provenance and exact historical baseline recovery
remain incomplete. File/hash/field checks do not prove full Phase-1 acceptance.


## Optional secondary acquisition (2026-09-16)

The electrode re-placement archive is now downloaded and officially checksum-verified.
See ELECTRODE_REPLACEMENT_REVIEW.md and electrode_replacement_native_sanity.json.
The separate secondary archive contributes 315,604,904 bytes; six core plus this
secondary archive total 16,251,455,246 bytes (16.25 decimal GB), excluding README,
extracted subsets and processed artifacts. This is not a current disk-usage scan.
Two of six sampled records contain missing numeric fields and are rejected by the
native reader; interval labels and repetition boundaries remain unavailable.
The frozen secondary selection decision and six-core classification scores are unchanged.
