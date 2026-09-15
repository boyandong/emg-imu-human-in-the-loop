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

Six Tier-1 archives are complete (approximately 16.06 GB combined). Exact sizes and
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
