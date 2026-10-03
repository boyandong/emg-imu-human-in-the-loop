# Feature Bank + Personal Calibration — interim evidence

The current project decision is to use the independently implemented, versioned
feature families. Exact equivalence to unavailable earlier B0/X1-H/X2 runs is
retired from the requested forward-work scope; historical references below remain
only to distinguish prior claims from new evidence. New own-device participants,
days and physical live sessions are deferred until recordings exist. Under the
available public and one-person Song evidence, no new bank is approved for live
deployment: validation-selected additions have inconsistent independent or
descriptive final macro-F1 and condition-level effects. The existing live model
remains unchanged, without a new accuracy claim. The formula audit now indexes
the already implemented explicit
Unknown decision and family-specific session-shift summary separately from
their base APIs.

This is an incomplete execution of the two supplied specifications, not a completion claim.
The publisher's 2026-09-25 DS2 v9 release now supplies verified public trial-level
subjective force codes; the prior v8-only statements that DS2 has no force labels
are historical snapshots. The new [force join](../benchmarks/discovery/DS2_V9_FORCE_LABEL_AUDIT.json)
and [fixed cross-user screen](../benchmarks/discovery/public_ds2_force_v9/REPORT.md)
are separate from still-unproven exact historical B0/X1-H/X2 replication.
Predictive evaluations use source OOF or held-out subjects/sessions; descriptive target
diagnostics are explicitly labelled. Model compositions and calibration rules are frozen
from source evidence, validation evidence or prespecified controls; final scores do not tune them.
Seed: 20260915. Classical logistic regression runs use CPU; no neural training is needed
for these representation comparisons.
The dated `results/validation.json` log records 224 tests (223 passed, one
skipped). The current classifier suite was rerun after the independent
new-v1 wearing-family delivery: 270 passed, one skipped and 14 subtests passed.
Older section-local test counts below are dated snapshots. The collection application's documented
default `collection/emg_meta/emg_meta/data` directory did not contain a session
at that earlier review. The user has since supplied four Song sessions at a
separate path; their offline evaluation appears at the end of this report.
They do not yet support a live-accuracy claim.
The collection app's nine targeted UI simulator, formal-protocol and UniBo
adapter tests pass in the existing `emgforce` Python environment with Qt
offscreen. This checks software readiness only; simulator data cannot establish
real-device classification performance.
The formal collection readiness gate now rejects valid trials whose stable
intervals overlap, lie outside their own trial or recorded EMG, or lack a
contiguous 200 ms EMG window. The previous synthetic fixture had allowed 144
trials to share a tiny interval; its repaired fixture uses separate recorded
intervals, and an overlap regression is rejected. The collection test suite
passes in `emgforce`; this remains a software integrity check, not device data.
The gate also reapplies the existing continuous signal-quality rule to each
valid formal interval and rejects a flat or clipped channel even when the
saved pre-session quality report says `passed`. A flat-channel regression and
the full collection test suite pass.
The same gate now checks valid formal prompts and all 26 calibration blocks
against their recorded protocol durations with 200 ms scheduling tolerance.
Stable-interval minimums subtract the two 300 ms transition guards and the
largest 200 ms arm/hand onset offset for formal prompts. The synthetic HDF5
fixture now spans those durations; shortened prompt and calibration intervals
fail regression checks. This proves recorded timing consistency only, not
physiological compliance with cues.
Formal readiness additionally checks IMU coverage for every valid formal and
calibration stable interval using the recorder's approximate EMG-index mapping.
It requires an IMU window, coverage near both interval edges, matched stream
lengths and finite gyro/accel values. Missing-window and NaN regressions fail;
the full collection suite passes. This does not prove precise EMG–IMU clock
synchronization or task-specific motion quality.
Fusion now has a separate `late_fusion_decision` API: scoreable probabilities
remain unchanged, while an all-zero effective quality weight or a prespecified
confidence cutoff returns explicit `Unknown`. Invalid quality values are
rejected. A fixed zero-cutoff replay of saved force quality probabilities
(`results/quality_unknown_replay.json`) covers 16 phase/scenario cells and
matches every saved fused probability array. No validation window is rejected;
one of 588 final trials is rejected in seven scenarios, including clean, and
none in synthetic saturation. The same trial drives all seven cells, so this
rule does not establish useful corruption detection or higher selective
accuracy. No threshold was tuned on final data or deployed to the live app.

The previous 28-row `REQUIREMENT_AUDIT.csv` is a triage summary, not an exhaustive
acceptance checklist; its broad locators must not be treated as exact source references.
`DOCUMENT_SCOPE_QUEUE.csv` now preserves all 2,586 nonblank extracted specification
lines (786 prerequisite, 1,800 goal), with exact source line numbers and source hashes.
Read-back validation proves no nonblank source line was omitted. Every queue row remains
unverified until contextual requirements and authoritative evidence are inspected;
formulas, examples and separators are included and are not independent requirements.
`DOCUMENT_SCOPE_AUDIT.json` proves index coverage only, not scientific completion.
`DOCUMENT_SECTION_AUDIT.csv` groups the same source lines into 78 contextual
sections, each with an exact line span, evidence hash, status and remaining
boundary. The companion JSON verifies all 2,586 nonblank lines belong to one
section: 15 sections have narrow verified evidence, 61 remain partial and two
are context. These counts describe audit coverage, not a completion percentage;
partial sections still require their named evidence and full acceptance checks.

Discovery delivery review now supplies 14 retrospective A–H scorecards while
preserving original selection totals/decisions and frozen experiments. Fresh
small-file evidence verifies six source sanity reports and 36 captioned plot hashes.
Complete archive size is 15,935,850,342 bytes (15.94 decimal GB), distinct from
raw expansion and the dated old disk snapshot. Historical DS2 publication match
does not establish archive identity; secondary source provenance remains incomplete.
`benchmarks/discovery/DISCOVERY_DELIVERY_AUDIT.json` records these boundaries;
it does not prove complete prerequisite acceptance or original phase ordering.

`FORMULA_IMPLEMENTATION_AUDIT.md` and CSV now record 31 reviewed formula/API
boundaries in the precise-definition appendix, each with exact heading line,
source AST symbol span and SHA-256. Twenty runtime classes have measured named
dimensions on explicit synthetic fixtures (original G5 only on its native four
channels). These fixtures prove interface dimensions, not dataset validity or
scientific completion. Candidate formula status is distinct from validated reuse.
Noise-only F0 thresholds and document uncentered CSP now have independent candidate
implementations and native held-out evidence, detailed below. Calibrated body-frame
IMU now has a tested explicit API; native calibration metadata/evaluation remain
unavailable. Generic short-window DTW is guarded. Original source/result recovery remains necessary for historical
X1-H/RLCS/CES/Frequency. No full requirement is automatically accepted by this audit.

## Explicit calibrated IMU context API

Calibrated IMU API evidence: `CalibratedBodyContextFamily` requires explicit >=1s
neutral IMU, a guided/measured forearm-forward vector, separate IMU sampling rate,
units and calibration trial identities. Neutral gravity plus projected forward
axis establishes an orthonormal calibration-relative body basis. Transform outputs
10 accel/gyro magnitude summaries, three mean lowpass gravity-direction coordinates
and two movement RMS values: 15 dimensions. Causal EMA tau=.5s is source-fixed;
each independent window initializes gravity from source neutral calibration.
Linear acceleration is accel minus this lowpass gravity estimate. No stable
absolute yaw or continuous world orientation is estimated. Short-window gravity
transients can enter the movement estimate; this is not validated motion tracking.

Explicit evaluation trial identities reject calibration/evaluation overlap;
training transforms require explicitly selecting evaluation=False. Unit/rate/window
checks prevent silently using EMG rate for IMU. Synthetic rigid device rotations
preserve calibrated context; static gravity and acceleration-step lowpass residual
oracles pass, with immutable source state. Suite now runs 169 tests, one skipped.
Native EPN/MANUS loaders do not supply a verified neutral/guided-axis calibration
record; their earlier 13-dimensional device IMU context is preserved as reference,
not upgraded to calibrated body frame. Native calibrated-frame performance is N/A
until real calibration records are available. No new classifier training or
performance gain is claimed for this API addition.

## Document F0 noise thresholds and uncentered CSP candidates

`document_signal.py` preserves old reference implementations while adding two
independent candidates. `RestNoiseLocalDetailFamily` freezes adjacent-difference
thresholds from explicitly labelled native Rest windows only, without joining
boundaries; changing all active amplitudes by 10,000 leaves thresholds identical.
Six local metrics have 48 dimensions at eight channels. Historical extra R0
features remain unavailable, so this does not complete mandatory R0 retention.

`DocumentCspFamily` implements XX transpose/(trace+epsilon) without centering or
covariance shrinkage, one-vs-rest source means, fixed gamma=1e-5 generalized
eigensystem and top2/bottom2 per native class. Variance normalization adds epsilon
inside log as specified. Independent second-moment, eigen-residual and variance
oracles pass. Five native classes give 20 CSP dimensions. All fit quantities
remain within each source training fold.

Frozen `wearing_core_ring --definition-mode document` uses F0-noise+reference
F1+document CSP (76 dimensions), plus optional raw F3c (100). It retains five
source-only Before repetition folds, source OOF temperatures and personal Before
source fit, zero target calibration. Old reference runs remain intact. Both F0
and CSP definitions change together; this comparison does not isolate either's
causal mechanism and does not recover missing historical validated X1-H.

| Cohort | Mean user Core F1 | Core+raw F3c F1 | Core LL | Core+F3c LL | Delta Brier |
|---|---:|---:|---:|---:|---:|
| Validation 15–17 | .466001 | .623405 | 4.556547 | 2.786742 | +.051554 |
| Final 18–20 | .619935 | .669772 | 3.007709 | 1.671127 | +.021902 |

Mean user results pool four after-wearing domains per user, whole-native-trial
averages of eight sparse windows. Three users/cohort, five classes without Pinch,
native 200Hz; no current-device or streaming accuracy claim. Absolute LogLoss
remains high despite source OOF calibration: source calibration does not establish
target reliability. No target probability fitting or final-score selection occurs.
Both phases replay 42 arrays exactly. Ninety-six family/condition descriptive
nuisance-distance/gesture-separation/J rows use source-only per-family diagnostic
scalers; labelled target centroids never update models or weights. Source model
states remain immutable. Runs:
`feature_bank_wearing_document_core_{validation,final}_20260916`.
Suite: 167 tests, one skip, remaining passing; compile checks pass. Full original
baseline recovery, complete document acceptance and hardware validation remain open.

## Raw ring covariance formula correction and wearing conditional increment

Current DTW input guard: `CompleteSequenceBatch` requires explicit full coverage
and a finite >=1-second physical duration for every sequence, independent of its
compressed path-bin rate. Fit and prediction both reject ordinary short/sparse
FeatureBatch inputs; subset selection retains durations and mutable metadata is
revalidated at use. Native UniBo producers supply contiguous bout durations.
Legacy MANUS sparse-window DTW refuses new execution before data loading/training;
saved sparse results remain retrospective proxies, not full-sequence evidence.
No native full-sequence MANUS replacement or streaming detector is claimed.

`results/complete_sequence_contract_replay.json` reloads 10,139 native UniBo bouts,
including 1,700 Day6 targets. Twenty-eight source/target probability arrays and
14 source temperatures match saved results exactly; source states and all five
parent artifacts remain unchanged, no classifier refit. Reproducer:
`PYTHONPATH=src python benchmarks/replay_complete_sequence_contract.py <dataset> <parent>`.
The 165-test suite passes with one skip. Complete-sequence eligibility is now
enforced; producer boundary provenance and stream segmentation still require
their own evidence. Later historical appendices retain their original test counts.

Goal text lines 1480–1533 define F3c from F2a raw-signal covariance.
Existing `RingGeometryFamily` uses envelope covariance for its ringcov block;
previous evidence remains a reference proxy, not exact F3c reproduction.
Separate `RawRingCovarianceFamily` applies channel centering, fixed .05 shrinkage
and trace normalization to raw windows. Four circular lags each output mean,
median, std, q25, q75 and early/late mean displacement: 24 dimensions at eight
channels. Verified ring topology must be explicitly supplied. Independent np.cov
oracle and cyclic rotation/random permutation checks pass; old results are preserved.

`wearing_core_ring.py` freezes actual concatenation Core=F0+reference F1+CSP
(66 dimensions) versus Core+F3c (90), balanced C=1 logistic regression.
Five Before-wearing repetition-held-out folds refit every family/scaler/classifier,
then fit source-only temperatures for complete models before opening target data.
F1 is not unavailable validated historical X1-H; F3c is not historical RLCS/CES.
Each cohort has three personal-source users/four wearing domains, no target calibration.

| Cohort | Mean user Core F1 | Core+F3c F1 | Mean delta LL | Mean delta Brier |
|---|---:|---:|---:|---:|
| Validation 15–17 | .430623 | .554775 | +1.358402 | +.035953 |
| Final 18–20 | .545346 | .645852 | +.614565 | +.028270 |

Means use each user's pooled whole-native-trial results across four domains;
they are not stream accuracy or the differently aggregated fixed-bank robustness
vector. Results support conditional predictive value for this fixed Core, with
identical target-independent choices in both cohorts. Native Myo 200Hz, five
classes without Pinch; current 250Hz device remains unvalidated. Per-user/domain,
per-class results and paired errors appear in the two
`feature_bank_wearing_core_raw_ring_{validation,final}_20260916` runs.
Saved source-fold/full models replay 42 probability arrays per phase exactly,
including source OOF temperatures; source states remain immutable. Archive hashes
and trial splits are checked. Suite: 163 tests, one skipped, remaining pass.
Integrated copied-value audit: 200 artifacts, 51,035 rows, 1,182 explicit partitions;
canonical provenance: 49,552 rows. These checks do not prove full document acceptance.

## Current interpretation of the evidence

The full bank has local gains and failures. It does not improve every task or the minimum
of the complete available robustness vector. The following seven-axis table describes
fixed full-bank algorithms at cal0; it does not substitute the best observed specialist.
The separately frozen EPN F0+Ring+CSP+IMU shortlist also failed to raise its
independent final-user minimum (full .3965 versus F0 .4263), as detailed below.
Different tasks have different classes, users and aggregation. The values are descriptive
and cannot be read as accuracy of one universally fitted classifier.

| Axis | F0 F1 | Full bank F1 | Evidence boundary |
|---|---:|---:|---|
| force | 0.4929 | 0.4516 | source users Ramp only; independent users and unseen force conditions |
| wearing | 0.4912 | 0.5917 | same-user before/after electrode shift |
| day | 0.7044 | 0.6992 | UniBo native four muscles, Days 7/8 |
| user | 0.4370 | 0.4792 | EPN selected 21-user subset, three final users |
| posture | 0.7044 | 0.6992 | same observations as day; posture strata, no fabricated IMU |
| speed/session | 0.4152 | 0.4582 | MANUS six users, six finger classes; session and speed confounded |
| synthetic quality | 0.4473 | 0.3875 | seven paired fixed corruptions, mean individual-user/scenario F1; clean excluded |

Authoritative values, run IDs, condition minima and aggregation are in
`results/full_system_robustness_vector.csv`. Across these unlike axes, descriptive means
are 0.5275 -> 0.5381, but minimum axis F1 is 0.4152 -> 0.3875. Quality/force share native
trials and day/posture share observations. The earlier six-axis vector omitted synthetic
quality and cannot support a claim that the complete available minimum improved.

### Answers to questions A–H, with remaining uncertainty

| Question | Current answer | Boundary / remaining requirement |
|---|---|---|
| A: Missing information or poor organization? | Both remain plausible. Feature organization and calibration alter performance substantially; extra dimensions and stronger nuisance deletion can hurt. | No experiment isolates the physical cause of current eight-channel hardware failure. |
| B: Which families add conditional information? | EPN fixed late-fusion Core receives log-loss gains from spectral/temporal/quality providers, but final F1 gains are inconsistent. Actual concatenated force Core receives spectral final F1 gains with worse log loss; other additions reverse validation gains or reduce minima. | Force Core has 0/1/2-shot controls; Core coverage on other eligible tasks remains incomplete. This is predictive information, not estimated mutual information. |
| C: Which families are specialists? | Ring helps average wearing/force performance in some compositions; spectral helps some load/force cells; temporal helps the native UniBo shortlist. | The EPN development Ring gain does not survive its independent shortlist final-user ablation: removing Ring improves pooled F1 and log loss. All force Core additions reduce Core's condition minimum. Historical RLCS/CES/Frequency equivalence is unverified. |
| D: Which need personal calibration? | MANUS session models can recover strongly with a small own-session budget; current EPN anchors can harm performance. Ramp-only force Core anchors recover only a small amount. | No universal anchor benefit or device calibration prescription follows. |
| E: Does Personal Anchor reduce cross-user variation? | No for the tested EPN branch: all nonzero budgets lower observed mean/minimum F1 and raise standard deviation relative to matched no-anchor controls. | Three final users, descriptive variation; this does not reject every anchor design. |
| F: Does Session Signature help cross-day/re-donning? | MANUS session profiles measure shifts. Matched current-session prototypes outperform the fixed long/current blend in the tested controls. | Targeted before/after wearing-domain controls now exist: local prototypes improve F1 with worse LL; cosine context weighting has no final F1 gain. Calendar-session, broader cross-day and own-device verification remain open. |
| G: Does the bank improve R_min? | Not for the complete seven-axis vector. Frozen concatenated force Core improves its own force-condition minimum; adding families can raise average F1 while lowering that minimum. The independent EPN shortlist also lowers its worst-user F1 from .4263 to .3965. | No global robustness recovery; unlike/correlated tasks and synthetic-quality scope remain explicit. |
| H: How much product calibration is needed? | Offline budgets range from limited Ramp-only recovery to large MANUS own-session recovery; EPN can fail even at five-shot. | Trial duration estimates exclude preparation/transitions. No measured device-level latency, accuracy or calibration duration claim. |

### Calibration recovery and model-composition limits

Force concatenated Core final mean-user F1 at cal0/1/2 is 0.4902/0.4943/0.5005;
its minimum force-condition mean-user F1 is 0.4199/0.4328/0.4328. Calibration uses
7/14 own-user Ramp trials and never the evaluated force conditions. Five-shot is
unsupported because only four Ramp trials per native class exist.

Selected-policy EPN full-bank F1 at cal0/1/2/5 is 0.4725/0.4053/0.4346/0.4115,
versus matched no-anchor 0.4725/0.4708/0.4670/0.4952. The negative result identifies
the current anchor-probability mixing branch; it must not be hidden behind a different
fine-tuned model. Calibration trials are excluded, so budgets have different test trials.

MANUS selected-policy full F1 at cal0/1/2 is 0.4631/0.5126/0.5926. Supplemental
source-scale matched local prototypes reach 0.4631/0.6633/0.8241, while fixed
long/local blending reaches 0.4582/0.5398/0.6176. This supplemental control is not
a final-selected deployment replacement. Two-shot leaves only one trial per class
per user (36 evaluation trials); high apparent recovery has a narrow evidence scope.

Synthetic-quality mean-user/scenario F1 is F0 0.4473, original uniform fusion 0.3875,
without the quality classifier 0.4209, quality routing with that classifier 0.3384,
and routing without that classifier 0.3884. Quality observation can detect corruption
without identifying a reliable gesture provider. These fixed diagnostic controls do
not justify promoting a routing policy based on final scores.

### Historical priors, delivery status and next experiment

Historical conclusions about X1-H, RLCS, CES, nRLCS and Frequency remain historical
priors. Exact original DS2 raw data and validated implementations/results are missing;
new reference formulas do not reproduce or refute them. Original UniBo G0/G5 formulas
are reused through an explicit native-four-channel compatibility adapter and tested.
The current temporal comparison shares most errors (G5/reference-F5 model correlation
0.9045 around F0); it is a common-baseline model comparison, not standalone G5/DTW.

Six new benchmark archives are complete. Five canonical CSVs, schema/provenance audits,
source run manifests, trial lists, dimensions, per-user/per-class diagnostics, calibration
burden estimates and SVG curves are available. The latest suite ran 223 tests (222
passed, one skipped); consolidated integrity covers 226 artifacts, 51,635 rows
and 1,248 explicit partitions, while canonical record verification covers
50,006 rows. These are narrow
integrity/implementation checks, not proof that every scientific requirement is complete.
The copied-result check was rerun against the archived processed-run directory
`D:/emg-imu-benchmarks/data/processed` plus source roots recorded per entry;
the canonical check used `benchmarks/canonical_delivery.py --verify`.

The highest-priority remaining work is the exact historical dataset/algorithm audit,
remaining named complementarity pairs and Core comparisons on other eligible datasets,
followed by an exhaustive formula/requirement audit. Current device failures still require
labelled eight-channel recordings and a held-out device-specific evaluation; UniBo's
four named muscles cannot establish that product's live accuracy. New results stay in
local commits; the specifications prohibit automatic GitHub push.

## Earlier specialist test results (supplementary)

| Dataset / failure | F0 macro-F1 | Frozen bank macro-F1 | Bank |
|---|---:|---:|---|
| UniBo / days 7–8 | 0.7044 | 0.7102 | F0 + temporal |
| Electrode Shift / subjects 18–20 | 0.4912 | 0.5478 | F0 + ring geometry |
| MANUS / session 3 | 0.4453 | 0.4718 | F0 + SPD tangent |
| EMG-FMG / loads 250–1000 g, subjects 4–6 | 0.6239 | 0.6530 | F0 + spectral |
| EMG-FMG / positions 2–8, subjects 4–6 | 0.6112 | 0.6307 | F0 + SPD tangent |

UniBo posture 2, its hardest posture cell, improves from 0.6638 to 0.6788.
EMG-FMG load log-loss improves from 0.8356 to 0.7420 and position log-loss from
1.1270 to 0.9889. MANUS SPD improves F1 but worsens log-loss (1.5920 → 1.7535),
so it cannot be called an unconditional improvement.

## Personal calibration

| Final protocol | 0 trials/class | 1 | 2 | 5 |
|---|---:|---:|---:|---:|
| LibEMG force ProductMode / F0+CSP+X1H with anchor fusion | 0.5860 | 0.5856 | 0.6063 | unsupported |
| EPN612 / F0+ring weighted fine-tune | 0.4386 | 0.4298 | 0.4419 | 0.4653 |
| MANUS / F0+SPD session fine-tune, mean per-user F1 | 0.4161 | 0.6861 | 0.6065 | unsupported |

Force has four trials/class/condition (MVC two); MANUS has three speed trials/class/session.
Unsupported budgets are explicit rows, not estimated scores. Calibration trials are removed
from evaluation and their IDs are saved in calibration run manifests. MANUS two-shot leaves
only one trial/class and has considerable sampling uncertainty. Its per-user mean F1 is a
different aggregation from the pooled screening F1 and should not be directly subtracted.

EPN prototype-anchor calibration failed on validation: 0.4169 → 0.4142 / 0.3996 /
0.3706 for 1/2/5 trials. This negative result is retained. Weighted fine-tuning was subsequently
frozen on validation (C=0.1, total calibration weight 15/class) before final evaluation.

## Screening and interactions

LibEMG force validation F0 is 0.5478. F0+CSP reaches 0.6256; adding X1H reaches
0.6446. Adding all four screened specialists yields 0.6430 with worse log-loss.
The full bank and leave-one-family-out evidence therefore do not support indiscriminate stacking.
Conditional deltas and pairwise error complementarity are in the accompanying CSVs.
They are held-out validation comparisons, not nested cross-validation estimates.

F9 quality weighting under synthetic motion bursts improves log-loss 1.4638 → 1.2203
but lowers F1 0.5109 → 0.5016. Synthetic corruption is not evidence of real hardware robustness.
No public benchmark proves the current eight-channel device will recognize open/pinch/fist
reliably without device-specific labelled calibration and live evaluation.

## Scope, dimensions and reproducibility

The six Tier-1 archives are complete and hashes are recorded in the discovery manifest.
EPN experiments use trainingJSON users 1–21; labelled testingSamples are unavailable and
Myo detections are never treated as ground truth. MANUS uses users 3–8 and six flexext gestures.
Electrode Shift uses validation subjects 15–17 and final 18–20; EMG-FMG uses validation
1–3 and final 4–6. These are bounded studies, not full-cohort evaluations of 612/27 users.
UniBo uses native four named-muscle channels and excludes cyclic-ring assumptions.
EMG-FMG uses columns 9–16 (EMG), excludes FMG, and samples eight windows from the central
nine seconds of each 18-second recording; this crop is an analysis assumption.

Feature dimensions are recorded per model in `results/feature_family_results.csv` and
run manifests. F2 uses covariance trace normalization, CSP train-fit filters and shrinkage
SPD tangent references. X1H retains normalized channel power and cross-channel structure.
F3 combines relative log-channel structure, cyclic energy statistics and covariance.
F4 summarizes spectral bands; F5 uses within-window temporal summaries. These implementations
are new reference implementations; equivalence to unavailable historical DS2 formulas is unproven.

`benchmarks/consolidate_feature_bank.py` rebuilds the five result tables with source run IDs
and SHA-256 provenance. Small JSON manifests are copied alongside the tables. The audited
force validation run additionally stores family pickle states, held-out probability arrays,
split trial IDs, and train-standardized nuisance/gesture centroid distances. Treat pickle
files as trusted local artifacts only. Raw archives remain outside Git.

## Remaining specification work

Historical DS2 remains blocked for reproduction: no exact historical artifacts were found.
The publisher-linked Kaggle version 8 archive is now complete at
`work/datasets/historical_ds2_candidate/ds2_kaggle_v8.zip`: 1,123,505,003
compressed bytes, 102 files and a recorded SHA-256. All ZIP members pass CRC.
Native raw-signal checks now include an exact [raw-to-MAV window join](../benchmarks/discovery/DS2_MAT_TRIAL_WINDOW_JOIN_AUDIT.json):
all 996,324 published MAV values reconstruct from the corresponding raw trial.
Of 2,863 gesture-code blocks, 2,862 are uniform and one is mixed and retained
as ambiguous. An additional [exact TDMS waveform join](../benchmarks/discovery/DS2_TDMS_RAW_EXACT_JOIN_AUDIT.json)
verifies subject-folder identity for 2,833 trials. Thirty consecutive trials
have no exact TDMS match, including at nonzero offsets, and remain unassigned.
This permits bounded public-v8 gesture-code and verified-subject-subset analysis,
but not complete subject/force stratification or reproduction of historical B0/X1-H/X2;
current-version identity with that old input is still unproven.
A separate [public-v8 subject-held-out gesture-code study](../benchmarks/discovery/public_ds2_subject_gesture/REPORT.md)
now uses only the 2,832 verified-subject, uniform-label trials for fixed F0
versus F0 plus F1/F2a/F2c/F4 increments. On final subjects 17–20, F4 raises
macro-F1 from 0.4413 to 0.5183 but worsens LogLoss from 1.7615 to 1.8344;
F1 raises macro-F1 only to 0.4475 while improving LogLoss to 1.7167.
The 5,725 held-out trial-probability rows and paired metrics pass independent
read-back checks. This adds bounded reference-family cross-user evidence, not
historical force-family reproduction or Song eight-channel validation.
The [DS2 force-annotation audit](../benchmarks/discovery/DS2_FORCE_ANNOTATION_SOURCE_AUDIT.json)
also checks the publisher-linked v8 description against every MAT variable and
the TDMS metadata inventory. Three subjective force conditions are described,
but no per-trial force key or validated condition order is supplied; the
description itself has sample-count and window-duration inconsistencies.
The exact subject and gesture joins cannot justify low/medium/high labels,
so the historical force requirement remains missing rather than silently
substituting a guessed order.
NinaPro requires access; secondary datasets remain explicitly deferred.
EMG-FMG and UniBo now each have six-trial QC and six raw/envelope/PSD plots.
F8 session signature now has a bounded MANUS reliability-fusion comparison, while full-bank
session-context integration remains incomplete. Personal normalization now has an EPN comparison;
DTW templates now have a bounded ordered-RMS trial comparison. Required prediction/state persistence
is currently strongest for audited force screening; other runners need the same coverage.
Per-family calibration gain, diagnostics across all failures and full-system comparisons
across datasets remain incomplete. Family robustness vectors and bounded force/EPN full-bank
ablations are now available; they do not establish a complete multi-session system.

Verification so far: 100 unittest tests passed, one skipped. This report and the active goal
remain open until those requirements and Git delivery are verified.

## Shrinkage late fusion follow-up

EPN uses independent F0 and ring classifiers with uniform population weights. Personal
weights use calibration-trial prototype separation divided by within-class dispersion,
softmax temperature 1 and population shrinkage n0=8. Both classifiers and all family/scaler
states are fit on users 1–15; reliability is fit only on explicit calibration trials.
The validation and final runs store fitted states, selected trial IDs and held-out predictions.
Final mean per-user F1 for uniform/personal fusion is 0.4690/0.4711 at two trials and
0.4780/0.4729 at five trials. Five-trial personal fusion improves log-loss 1.4425 → 1.4258
while lowering F1. This is a limited specialist-fusion experiment, not the full-bank solution.
Uniform scores change with calibration budget because the same calibration trials are removed
from both methods; those changes alone must not be interpreted as calibration gains.

## Session context follow-up

MANUS session signatures are computed per family from each user's session-1 class profile and
explicit session-2/session-3 calibration trials. Cosine agreement modifies personal reliability
weights; no evaluation labels, evaluation variance or unlabelled evaluation stream updates enter
the signature. All calibration/evaluation trial intersections are asserted empty. Validation
one-shot F1 is 0.3622 for uniform fusion, 0.3474 for personal weights, and 0.3593 with session
context. Two-shot methods are equal at 0.3537. These results do not justify enabling this rule
by default. Full result cells, vectors and split IDs are preserved with the result manifests.
Final one-shot F1 is 0.4518/0.4212/0.4527 for uniform/personal/context fusion; two-shot
methods are equal at 0.3444. Context recovers the personal weighting loss but barely exceeds
uniform fusion, reinforcing the limited value of this particular rule. Future runner outputs
also include source trial IDs and aligned target labels in their prediction artifacts;
the existing runs preserve evaluation IDs and probability arrays separately.

## Personal amplitude normalization follow-up

Each source user's rest median and active absolute-amplitude 95th percentile are fit using
that user's training trials only. Source-normalized F0+ring features, family states, scaler
and classifier are then frozen. Target normalization uses explicit labelled calibration trials;
zero-shot uses a source-population normalizer. Raw and normalized models are compared on
exactly the same remaining trials. All states, source IDs and calibration/evaluation IDs are saved.
Validation one-shot mean per-user F1 falls from 0.4208 raw to 0.1886 normalized. Independent
users confirm degradation: raw/normalized F1 is 0.4389/0.3205 at one trial, 0.4407/0.3672
at two and 0.4509/0.4355 at five. This normalization rule is not suitable as a default repair.
The experiment changes source normalization as well as target normalization, so it measures
the complete normalization protocol rather than isolating a single target-side transformation.

The detailed formula divides by `Q95 + ε`, while those frozen runs use a
`max(Q95, ε)` floor. A separate opt-in `DocumentPersonalNormalizerV2` now
implements the stated denominator and passes a near-zero known-signal test;
the older normalization results keep their original identity. This numerical
correction has no separate native performance screen or default promotion.

A [native EPN input audit](../benchmarks/new_bank_v2/DOCUMENT_NORMALIZER_EPN_REPORT.md)
compares 34 source and target calibration states without retraining. Every
active Q95 exceeds 2.0 archive units; the largest relative denominator
difference is 5.01 × 10⁻¹¹. This bounds the numerical effect on those
inspected inputs, without claiming identical model predictions.

## Ordered-RMS DTW follow-up

Each trial becomes an ordered sequence of channel RMS values from its eight sampled windows.
This is a sparse summary, not a continuous full-trial trajectory. Source-user templates use
session 1; personal templates use only selected target calibration trials. Temperature is
the median source-template distance and fusion weight is fixed at shots/(shots+2).
Independent session-3 mean per-user F1 for baseline/DTW/fusion is 0.4040/0.5739/0.4040
at one trial and 0.3102/0.4769/0.3565 at two trials. Personal DTW is promising in this small
cohort, while the fixed fusion fails to capture most of its gain. Different remaining trials
across budgets prevent interpreting raw cross-budget changes as pure calibration gains.
Templates, probability arrays and all calibration/evaluation IDs are preserved. Two new
tests verify trial/window order and reject unequal path lengths without implicit alignment.

## Audited force ablation and table integrity

The audited force selection run now includes the full five-family shortlist and all five
leave-one-family-out models, including removal of F0. Each is reported for ALL and all eleven
target-intensity/subjective conditions (72 ablation rows). Validation predictions, fitted
family states and source/validation trial IDs are saved. This is complete for that shortlist,
not an all-dataset or all-F0–F9 system ablation.
`benchmarks/verify_feature_bank_results.py` checks source SHA-256 hashes, copied values and
available explicit trial-list intersections. The current audit passes 31 source artifacts,
2230 result rows and 122 explicit split checks. It does not prove missing fit-state coverage,
all scientific requirements or live-device performance.

## EPN shortlist interactions and ablation

The original single-family validation screen selects ring geometry, CSP and real IMU context
as the three strongest additions to F0. On the same held-out development users 16–18, the
four-family full shortlist reaches 0.4498 macro-F1, compared with baseline 0.3838 and the
best two-specialist combination 0.4245. Removing F0/ring/CSP/IMU gives 0.3294/0.4126/
0.4245/0.4240 respectively. This supports a development-set interaction, not a new independent
test claim. Twelve models and all subject cells are stored, together with conditional deltas,
pairwise error complementarity, fitted classifier/family states and trial-aligned predictions.
Result integrity now passes 35 source artifacts, 2376 rows and 123 explicit split checks.

## Force protocol and probability-scale correction

Force-ZeroShot now calibrates held-out users using Ramp trials only and evaluates exclusively
at the other eleven intensity conditions. Its final mean per-user F1 is 0.5582/0.5630/0.5675
for 0/1/2 trials; log-loss is 2.6319/1.2182/1.2066. Five trials are unsupported (four Ramp
trials/class). Calibration IDs and target IDs are asserted disjoint and saved.
Force-ProductMode uses target-intensity calibration and is reported separately. It must not
be described as unseen-force calibration. Its corrected pooled final F1 is
0.5860/0.5856/0.6063; the aggregation differs from the ZeroShot per-user mean.
The original ProductMode temperature used median evaluation distances, creating dependence
on other evaluation rows. The corrected runner fits this scale from calibration distances
only; corrected result tables replace the original runs. A regression test verifies that
adding extreme evaluation rows cannot change an existing query probability. Original run
manifests remain historical artifacts and are not the source of current consolidated scores.
Current integrity checks pass 37 source artifacts, 2592 rows and 135 explicit split checks.

## Family robustness vectors

`results/robustness_vectors.csv` reports ten family-specific validation vectors and
`robustness_cells.csv` preserves 293 matched condition deltas for F1, log-loss and Brier.
Each vector coordinate is candidate-minus-F0 on its own benchmark; no cross-dataset average
is taken. Quality remains N/A because the synthetic quality-weighting study does not provide
matched additive-family screening. MANUS speed is confounded with target session, UniBo day
with reapplication, and EMG-FMG position with mixed loads; these limits are recorded in the
definition JSON. X1H's force delta is +0.0425 but day delta is −0.0246. Trace covariance's
wearing delta is +0.1617 but force delta is −0.1010. The vector therefore supports specialists,
not a universally invariant family. These development deltas do not replace independent tests.

## Unavailable family handling

Late fusion now skips absent probability providers and renormalizes remaining weights. If
all remaining providers had zero configured weights, it falls back to equal weights among
available providers. Missing personal calibration families receive zero reliability weight;
if none has complete class coverage, an explicit error requests a population-mode fallback.
No unavailable family is imputed. Tests cover missing providers, zero surviving weights,
partial calibration coverage and the all-unavailable calibration case.

## Multi-specialist EPN late fusion and full removals

The fixed bank contains F0, X1H, CSP, ring, spectral, temporal, real IMU and quality providers.
Independent source-trained logistic classifiers supply probabilities. Calibration-only
reliability uses n0=8 shrinkage toward uniform population weights; each family also has a
calibration-only personal anchor with fixed shots/(shots+2) probability mixing. All eight
provider removals and removal of the complete anchor stage are evaluated on identical
remaining trials. F8 is unavailable in EPN; F9 is a classifier provider in this experiment,
not sample-level quality gating. This remains a bounded cross-user system comparison.

Final mean per-user F1 for full/no-anchor/uniform is 0.4332/0.4647/0.4773 at one trial,
0.4524/0.4755/0.4770 at two and 0.4732/0.4949/0.4829 at five. At five trials, no-anchor
fusion also has better log-loss (1.4394 versus full 1.5723). Personal anchors in this fixed
mixing rule hurt; reliability alone shows a small five-trial benefit. This does not justify
default deployment of the full system. The current integrity audit passes 41 source
artifacts, 3264 rows and 159 explicit trial split checks.

## Multi-session full fusion follow-up

The same eight-family fusion is now evaluated on MANUS session 1→2 development and
1→3 independent sessions, users 3–8. F8 uses source-user class profiles and calibration-only
class cosine agreement. F9 uses source-fit signal quality: F0/CSP receive minimum channel
quality, other signal providers mean quality, and IMU/context providers retain unit quality.
Provider removals are for the anchor/reliability bank; the additional F8 and F8+F9 variants
separately assess context and quality weighting. The v2 follow-up below completes combined
system component removals using the same fixed rules.

Final one-shot F1 is 0.4609 for anchor/reliability fusion, 0.4055 without anchors, 0.4509
with F8 and 0.4972 with F8+F9. The F8+F9 log-loss is 1.2687 versus 1.2613 before context,
so F1 and probability quality disagree. Two-shot F8+F9 F1 is 0.4231, leaving only one
trial/class/user. Five-shot is unavailable and recorded in the manifest. Session vectors,
trial IDs and fitted states are saved. Per-family dimensions extracted from actual fitted
states are in `results/full_fusion_dimensions.json`. Current integrity checks pass 45
source artifacts, 4314 rows and 195 explicit split checks.

## Combined-system component removal audit

MANUS v2 evaluates the combined anchor + reliability + F8 + F9 system and individually removes
all eight providers, F7 anchors, F8 context and F9 quality weights. The old variants are retained
as comparisons; unsupported five-shot rows now explicitly contain blank metrics. The independent
one-shot full F1 remains 0.4972. Removing F7/F8/F9 weighting yields 0.4241/0.4517/0.4509;
removing CSP increases F1 to 0.5171, while removing ring or temporal reduces it to
0.4441/0.4414. These are frozen-model diagnostics, not permission to select a different model
on the test set. This completes component removal for this bounded MANUS system only.
Current copied-result integrity checks cover 45 source artifacts, 5350 rows and 231 explicit
trial split checks. Cross-failure family diagnostics and broader reproducibility coverage
still remain open.

## Per-family calibration and session-drift diagnosis

The trusted saved MANUS states now produce validation and final diagnosis without fitting
new parameters. Across both phases, 288 calibration cells and 96 same-user class-matched
session-displacement cells are saved. Calibration gain compares anchored and unanchored
provider predictions on identical evaluation trials; it does not subtract unmatched budgets.
Final mean one-shot F1 gains are +0.0309 F0, +0.0198 X1H, +0.0126 CSP, +0.0120 ring,
0.0000 spectral, −0.0019 temporal, −0.0078 IMU and −0.0065 quality. Anchors therefore
help some spaces and harm others under the fixed mixing rule.

`cross_session_family_diagnostics.csv` reports D_nuisance (source-to-target class centroid
displacement), D_gesture (target within-user class separation) and their ratio J in source
standardized coordinates. Evaluation labels are used only for offline diagnosis and cannot
change fitted states. State serialization is checked unchanged after source/target transforms.
Selected X1H/frequency, X1H/CSP, ring/anchor, raw/ring and temporal/anchor error comparisons
are appended to the complementarity table. Current integrity checks cover 51 source
artifacts and 5914 copied rows. Equivalent diagnostics for other failures remain incomplete.

## Cross-user family diagnosis

EPN adds 192 same-evaluation-trial calibration cells and 48 source-population-to-target-user
class displacement cells across validation/final phases. Unlike same-user MANUS, the source
reference is the training population's class centroid; no nonexistent source profile is
constructed for a held-out user. All distances use frozen source-fit coordinates. Selected
raw/robust and ring/anchor complementarity cells are appended to the common table. Model
state is asserted unchanged after transforms; evaluation labels are only diagnostic inputs.
These results diagnose the existing fixed anchor rule, not a newly optimized personal model.

## Full-system prediction replay

`python -m emgimu.feature_bank.replay_fusion` independently transforms raw inputs using saved
family/scaler/classifier states, rebuilds source-only session profiles where applicable, and
replays every saved fusion variant. It asserts unchanged classifier/family state and complete
coverage of prediction-array keys. Independent MANUS replays all 450 probability arrays with
zero difference; independent EPN replays all 132 arrays with maximum error 1.67e-16. No
classifier or family fitting occurs. Validation runs have the same full replay coverage.
Replay audits are copied into result manifests. This proves these saved fusion outputs can
be reconstructed, while historical DS2 and other runners' reproducibility gaps remain open.

## UniBo validation state and drift audit

The original Day 1–5→6 validation protocol now persists family/classifier/scaler states,
aligned predictions and explicit trial partitions. Audited rerun F1 scores exactly match
the original validation scores. Nine native four-channel families generate same-user
day displacement matched by posture/class, posture displacement matched by class, and
within-posture gesture separation. Coordinates are standardized on source windows only;
centroids are window-pooled and this analysis does not control reapplication within day.
The dedicated `cross_day_posture_diagnostics.csv` contains these diagnostic cells. Test-split
intersection verification now checks every available train/validation/test partition pair.
The final-state persistence and replay follow-up below now covers independent UniBo outputs.

## UniBo independent-state replay

The audited Day 7–8 run uses the original frozen F0+temporal choice and exactly reproduces
the original final F1 cells. Both predictors and all family/scaler states are persisted.
`python -m emgimu.feature_bank.replay_unibo` loads target days only, verifies sample labels,
trial IDs, users, days and postures, and replays every saved predictor without fitting.
Independent replay covers 48818 windows with zero probability difference; validation replay
covers all nine saved predictors. State serialization is asserted unchanged. No model
selection uses the replay results.

## Wearing-state persistence and replay

Electrode Shift audited validation/final runs persist per-subject family/scaler/classifier
states, trial-aligned predictions and before/after trial partitions. F1 cells exactly match
the original runs. `wearing_family_diagnostics.csv` adds 132 per-family/subject/after-domain
centroid-displacement and gesture-separation cells in source-standardized trial coordinates.
`python -m emgimu.feature_bank.replay_wearing` reproduces all 27 development and six
independent predictors with zero probability difference and checks unchanged states and
disjoint before/after trial identifiers. The original before-trained subject-specific protocol
remains fixed; this is not a new cross-user wearing experiment.

## External load and limb-position state audit

EMG-FMG audited runs persist native eight-channel EMG family/scaler/classifier states,
trial-aligned predictions and scenario-specific train/evaluation partitions. All original
F1 cells remain unchanged. The source-standardized class-centroid diagnostic table adds
363 family/subject/load-or-position cells. External grasped load is kept distinct from
voluntary contraction intensity. Validation replay reproduces all 54 predictors with zero
probability difference. Nested subject/scenario split checks are now covered by the integrity
verifier and two regression tests (including a deliberate train/test overlap).
Independent replay also reproduces all twelve predictors with zero probability difference.

## Calibration burden

`calibration_burden.csv` and the standalone `performance_vs_calibration_budget.svg` report
0/1/2/5 budgets, supported trial counts, signal-duration estimates and force/session needs.
EPN five-shot needs 30 six-class trials (about 150 signal seconds). MANUS one-shot needs
six task-class trials (about 60 seconds) each target session; two-shot needs twelve, with
very few remaining test trials. Force one-shot needs seven trials (about 21 seconds);
ProductMode uses target intensity, whereas source-only calibration uses Ramp. Estimates
exclude setup, transitions and rests. These native task costs cannot be transferred directly
to the current four-gesture device. A duration estimate is not a validated live onboarding time.

## Research questions and family roles — current evidence

A. Both information and organization limit performance: specialist views improve particular
failures, while indiscriminate stacking and anchor probability mixing can lose useful evidence.
B. CSP/X1H contribute on force development; ring/CSP/IMU contribute conditionally in EPN;
combined MANUS removals show context-dependent temporal/ring/anchor value.
C. X1H is a force specialist; ring and covariance are wearing specialists; SPD is a position/
session specialist; temporal summaries help UniBo days. No family is universally invariant.
D. EPN X1H/CSP anchor views show positive five-trial gains, and personal DTW improves the small
MANUS cohort. F0/quality anchor mixing can harm EPN. Calibration value depends on feature space.
E. The anchor-coordinate comparison below shows selective improvement of relative separation,
not universal reduction of harmful cross-user variation.
F. F8 alone is unstable; the combined MANUS system shows one-shot context contribution.
Re-donning is confounded with session/day, so an isolated physical re-donning claim is unproven.
G. The frozen expert-package envelope below improves the paired mean and minimum dimension;
a universal full system evaluated across all seven failures remains incomplete.
H. MANUS one-shot fine-tuning gives substantial bounded-cohort recovery at roughly six trials;
EPN needs more calibration for a smaller gain. Current-device recovery and onboarding cost
must be established with its own labelled recordings.

| Family | Evidence-supported role in the evaluated protocols |
|---|---|
| F0 local detail | Backbone reference; indispensable in EPN shortlist |
| F1 X1H | Specialist for force; calibration amplifier in EPN |
| F2 covariance / CSP / SPD | Specialists with different wearing, force and session strengths |
| F3 ring | Specialist for wearing; complementary expert in EPN |
| F4 spectral | Specialist for external load; auxiliary force evidence |
| F5 temporal / DTW | Day specialist; calibrated temporal expert in MANUS |
| F6 real body context | Complementary expert in EPN; oracle posture variant is an upper bound |
| F7 personal anchor | Calibration amplifier in selected spaces; harmful mixing in others |
| F8 session signature | Context-dependent complementary expert; no stable isolated gain |
| F9 quality | Observability specialist; F1/log-loss gains can disagree |

These roles concern the implemented variants. Historical X1H/RLCS/G5 priors are retained;
missing historical DS2 artifacts prevent claiming formula or baseline equivalence.

## Cross-user anchor-coordinate comparison

For the same evaluation trials, class centers are measured across the three held-out EPN
users in source-standardized feature coordinates and in personal anchor-distance coordinates.
`anchor_variation_diagnostics.csv` preserves 96 validation/final cells. Coordinate dimensions
and units differ, so only the dimensionless gesture-separation / cross-user-distance ratio J
is compared directly. At five trials, independent X1H J improves 0.572→0.749, CSP
0.595→0.811 and IMU 0.678→0.781. F0 falls 1.125→0.142, ring 0.991→0.640, spectral
0.663→0.204, temporal 1.110→0.344 and quality 0.916→0.152. This supports a selective
anchor view for particular spaces, not a universal invariant coordinate system. Three users
and centroid-level diagnosis limit the claim; raw distance magnitudes are not comparable
across these spaces. Current table integrity checks cover 65 sources and 7019 rows.

## Frozen expert-package robustness envelope

`system_robustness_envelope.csv` reports six available native-task coordinates, matched
baseline coordinates where available, and separate worst-condition cells. The descriptive
candidate mean is 0.5701 and minimum available coordinate is 0.4718. On five paired
coordinates, baseline/candidate means are 0.5378/0.5670 and minimum coordinates are
0.4370/0.4718. Force lacks an independently persisted final F0 reference; real-quality
performance remains N/A. Means use different datasets, tasks and metric aggregations and
must not be interpreted as one population's accuracy or a universally evaluated model.
The minimum dimension is different from the worst condition within a dimension; both are
kept separate. This is supplementary evidence for the fixed benchmark-specific expert
package, not completion of the original seven-failure full-system comparison.

## Independent force reference and harder-condition caution

The final source-only F0 reference now uses the original source-fit feature state and
prespecified C=1 logistic regression on the same Ramp subjects 1–6. It evaluates precisely
the same target subjects 9–10 and intensity trials as the frozen bank, whose probability
outputs are verified unchanged. Independent F0/bank pooled F1 is 0.5788/0.5860. The
worst intensity-cell F1 slightly declines 0.4853→0.4841; wearing's worst after-shift domain
also declines 0.3938→0.3638 despite its higher pooled F1. Overall improvement therefore
does not imply improvement of every hard condition.

The expert-package envelope now has six paired coordinates: descriptive baseline/candidate
means are 0.5447/0.5701 and minimum aggregate dimensions 0.4370/0.4718. Individual
worst-condition cells remain separate and expose the regressions above. Real quality is N/A;
the original universal-system comparison remains open. Current result integrity covers
66 sources, 7091 rows and 252 explicit split checks.

## Full specialist fusion under source-force-only calibration

`feature_bank_force_full_fusion_{validation,final}_20260915` evaluates seven
independent providers (F0, X1-H, CSP, ring, spectral, temporal, quality) on trial means.
Providers fit only Ramp subjects 1–6. Validation uses subjects 7–8; final uses 9–10.
Target-user calibration uses only separate Ramp trials; all 11 target intensity
conditions remain evaluation-only. F6/F8 are unavailable without real IMU/session keys.
This trial-classifier experiment is distinct from the earlier window-classifier bank;
its scores must not be silently substituted into that earlier benchmark.

| Final mean per-user macro-F1 | 0/class | 1/class | 2/class |
|---|---:|---:|---:|
| Independent F0 trial classifier | 0.4929 | 0.4929 | 0.4929 |
| Uniform population bank | 0.4665 | 0.4665 | 0.4665 |
| Personal reliability without anchors | 0.4665 | 0.4636 | 0.4748 |
| Full reliability plus anchors | 0.4665 | 0.4688 | 0.5051 |

Validation full-bank F1 is 0.5935/0.6041/0.6116; the final zero-shot bank regresses
against F0. Two-shot calibration improves final F1 by 0.0386 over the population bank
and 0.0122 over F0. Full-bank final log loss is 1.4691/1.3380/1.3238 versus F0 2.1005.
All seven provider removals are exported, alongside explicit unsupported five-shot
rows (only four Ramp trials/class exist). Temperatures use calibration distances only.
These fixed-provider parameters were not selected on the final users.

Both phases replay all 66 saved probability arrays with zero absolute error using
persisted family/classifier/anchor states; replay performs no family/classifier fit and
verifies calibration temperatures, identifiers, disjoint partitions and state immutability.
Current integrity audit covers 70 source artifacts, 7499 copied rows and 264 explicit
split checks. The full unit suite runs 102 tests, passing with one skip. These checks
support this bounded experiment, not completion of every original research requirement.

## Anchor similarity batch-independence correction

The F7 distance and margin coordinates were already row-wise, but its auxiliary
similarity columns used the median distance of the current transform batch. This made
one query's similarity depend on unrelated evaluation rows and violated the fixed
calibration-state requirement. New anchors now freeze that scale from calibration
distances during fit. Legacy pickles lacking the scale derive a fixed fallback from
their calibration prototypes without state mutation; that fallback is not asserted to
reconstruct the original calibration-distance median.

A regression test compares a query alone versus with 100 extreme unrelated rows for
Euclidean, standardized Euclidean and cosine metrics, including legacy states. All
output coordinates remain consistent and fitted states unchanged. Current full-fusion
experiments consume only distance columns: replay after this change checks 264 EPN,
900 MANUS and 66 final-force probability arrays, unchanged within 2.23e-16. Existing
distance-based performance and diagnostics therefore remain valid; historical auxiliary
similarity outputs should not be treated as valid fixed-state features. Full tests now
run 103 cases, passing with one skip. Source-only probability calibration and strict
OOF evidence remain separate open requirements; this fix does not satisfy them.

## Nested source-subject OOF probability calibration

`feature_bank_force_nested_oof_20260915` supplies strict grouped OOF evidence for all
seven force-bank providers. Outer held-user pairs are (1,2), (3,4), (5,6). Every outer
training partition is split into two inner subject folds. Families (including supervised
CSP), feature scalers and trial-mean logistic classifiers are refit inside each inner
and outer training fold. A scalar probability temperature, constrained to [0.25,4], is
fit only to inner OOF probabilities and labels, then applied to the outer held users.
Target users 7–10 are not opened. All 168 source Ramp trials have exactly one outer
prediction. Persisted predictions contain raw and calibrated probabilities, original
trial identifiers, users, labels and fold IDs. The complete 21-pair error-complementarity
matrix uses these calibrated outer OOF predictions.

| Family | Raw OOF log loss | Calibrated OOF log loss |
|---|---:|---:|
| F0 | 2.2446 | 1.3471 |
| X1-H | 1.7950 | 1.4408 |
| CSP | 2.1794 | 1.5289 |
| Ring | 2.9042 | 1.8616 |
| Spectral | 2.2009 | 1.3126 |
| Temporal | 3.3012 | 1.6810 |
| Quality | 3.0045 | 1.6042 |

Temperatures preserve class order, so family F1 is unchanged. Better log loss here
demonstrates correction of source-domain overconfidence, not improved target-force
discrimination. This does not retroactively calibrate previously published full-fusion
target probabilities. Applying a source-only calibrated bank to the frozen target
protocols and obtaining OOF evidence on the other native benchmarks remain open.

Replay verifies all 42 outer probability blocks with zero error, exact label/user/fold
alignment, nested trial partition coverage and state immutability without refitting.
Inner calibration fitting is evidenced by executable run code and explicit inner
partitions; inner fitted states/probabilities are not retained in this run. Current result
integrity checks 72 source artifacts, 7534 rows and 273 explicit partitions. Full tests
run 105 cases, passing with one skip.

## Source-only probability calibration applied to target force

`feature_bank_force_probability_{validation,final}_20260915` reuses the exact source-fit
providers from the earlier full-force validation run. No family, scaler or classifier is
refit. Each provider's temperature is fit once to its raw source-user OOF predictions
(subjects 1–6, Ramp only). Source trial identifiers and labels are checked against the
raw adapter output; source OOF SHA-256, fitted temperatures and all fitting identifiers
are retained in `probability_calibration.json`. Both target phases use byte-equivalent
calibration metadata. Target-user Ramp trials still supply only the permitted personal
anchors/reliability; target-force trials remain evaluation-only.

| Final mean per-user macro-F1 | 0/class | 1/class | 2/class |
|---|---:|---:|---:|
| Source-calibrated F0 | 0.4929 | 0.4929 | 0.4929 |
| Source-calibrated uniform bank | 0.4516 | 0.4516 | 0.4516 |
| Source-calibrated full personal bank | 0.4516 | 0.4802 | 0.5563 |
| Earlier uncalibrated full bank | 0.4665 | 0.4688 | 0.5051 |

Calibrated full-bank validation F1 is 0.5994/0.6277/0.6793. Final two-shot F1 gains
0.0512 over the earlier full-bank result, but zero-shot F1 declines 0.0149. Calibrated
F0 final log loss improves 2.1005→1.2853 without changing its classification. Full-bank
final log loss is 1.4021/1.4264/1.4250: better at zero shot than the earlier 1.4691,
but worse at two shots than the earlier 1.3238. Thus calibrated classification benefits
and probability-quality benefits do not move uniformly together. Some source-fitted
temperatures approach the prespecified upper bound 4; the bound is retained rather
than retuned after seeing target results.

These are prespecified supplemental comparisons on previously opened final subjects,
not a newly untouched final test. Both phases replay all 66 saved probability arrays
with zero error; reused source state remains unchanged. All provider removals and
unsupported five-shot rows are retained. Current table integrity covers 76 source
artifacts, 7942 rows and 285 explicit partitions. The 105-test suite passes with one skip.
Other native benchmarks still require comparable grouped OOF probability calibration;
the original DS2 release and universal-system completion remain unresolved.

## EPN nested source-user OOF and auditable inner calibration

`feature_bank_epn_nested_oof_20260915` extends the grouped protocol to the eight-provider
EPN bank, including its real IMU provider. Source users 1–15 are divided into three
outer folds of five users each. Every outer training set has ten users, divided into
two internal folds of five. All family/scaler/classifier fitting remains within the
corresponding training subjects. Temperature bounds and source-only fitting rules
are identical to the force study. No target users 16–21 are loaded.

The run covers all 2250 labelled source trials and exports raw/calibrated OOF predictions
and the 28-pair error-complementarity matrix. Calibrated/raw log loss is 1.4517/1.5180
for F0, 1.5340/1.7712 for spectral and 1.4485/1.5585 for quality; all eight providers'
log loss improves while F1 remains unchanged. These are source cross-user results,
not a target-user personalization gain or evidence that all eight providers should be
used at deployment.

Unlike the earlier force run, this run retains every inner fitted state and inner OOF
probability block. Replay checks all 48 outer and 48 inner blocks with zero error,
recomputes each outer temperature from replayed inner predictions/labels, verifies
nested trial partitions and exact identifier alignment, and confirms fitted-state
immutability without classifier/family fitting. The previous force artifact remains
replayable with its original, more limited inner evidence; it is not overwritten.
Current table integrity checks 78 source artifacts, 7986 rows and 294 explicit partitions.
All 105 tests pass with one skip. Applying these source-only temperatures to the fixed
target EPN fusion protocol remains the next experiment; other failure protocols and
the original historical DS2 requirement remain open.

## EPN source-only calibrated target fusion: anchor negative transfer

`feature_bank_epn_probability_{validation,final}_20260915` reuses all eight original
source-fit EPN providers, with temperatures fitted only to raw source-user OOF
probabilities from users 1–15. Source trial identifiers and labels match the raw adapter;
fitting IDs, temperatures and source-OOF hash are saved. Validation/final use identical
calibration metadata, and no family/scaler/classifier fitting occurs. Legacy EPN source
manifests lack a dataset field; compatibility accepts that legacy format only alongside
exact source-trial matching. Users 16–18 and 19–21 remain target validation/final groups.

| Final mean per-user macro-F1 | 0/class | 1/class | 2/class | 5/class |
|---|---:|---:|---:|---:|
| Uniform population fusion | 0.4792 | 0.4818 | 0.4818 | 0.4909 |
| Personal reliability without anchors | 0.4792 | 0.4762 | 0.4770 | 0.4791 |
| Personal reliability plus anchors | 0.4792 | 0.4124 | 0.4240 | 0.4214 |

Target calibration trials are fully removed from evaluation. Rows at different budgets
therefore have different evaluation trial sets; within each budget every method uses
exactly the same remaining trials. This is target-user product personalization, not
source-only zero-shot robustness once target labelled examples are provided.

The final full-bank log loss is 1.5069/1.5523/1.5897/1.6092, while population fusion at
five shots is 1.4966. Anchor mixing causes both classification and probability-quality
regression in this EPN setting, contrasting with force two-shot improvement. This
supports family/task-specific calibration rather than a universal fixed anchor mixture.
It does not justify tuning the mixture on final users. Original and corrected runs remain
separately identified, with all provider removals retained. These supplemental results
use previously opened final users and are not a new untouched test.

Both phases replay 132 saved probability arrays, with maximum error 2.23e-16, exact
trial/user/label alignment and unchanged fitted source state. All 105 tests pass with
one skip. The full-system seven-failure evaluation and remaining dataset-specific OOF
evidence still require further work.

## Full-force condition envelope and individual failure cells

`feature_bank_force_condition_{validation,final}_20260915` analyzes saved calibrated
force-bank predictions without training, calibration refitting or raw-signal processing.
Each phase exports 1089 subject/pooled-condition cells across all eleven intensity
conditions, three supported calibration budgets and all eleven model/removal variants,
plus 33 robustness summaries. Native trial identifiers are checked against saved subject
and gesture labels before conditions are recovered. Prediction SHA-256 and original
evaluation trial IDs are recorded. Label/user tampering is covered by a regression test.

| Final method | Mean condition F1 | Minimum pooled-condition F1 | Minimum subject-condition F1 |
|---|---:|---:|---:|
| F0 | 0.5103 | 0.4102 (MVC) | 0.0830 |
| Uniform bank | 0.4767 | 0.3778 (Hard) | 0.1211 |
| Full bank, 1/class | 0.4960 | 0.4100 (Hard) | 0.1465 |
| Full bank, 2/class | 0.5673 | 0.5007 (Hard) | 0.0873 |

The condition mean here averages pooled-user per-condition F1. It differs from the
earlier average of per-user ALL-intensity F1 and must not be substituted for it.
Two-shot calibration raises the pooled worst-condition F1 by 0.0904 over F0 and 0.1229
over uniform fusion, but the worst individual user-condition score declines 0.1211→0.0873
against uniform fusion. One-shot calibration's worst individual cell is better at 0.1465.
Thus neither aggregate condition improvements nor larger budgets ensure protection of
the hardest individual cells. These supplementary diagnostics preserve all negative
results and do not select a new policy using final-user scores. The minimum condition
is not the requested minimum across seven failure dimensions. Full tests run 106 cases,
passing with one skip; remaining failure-system coverage is still incomplete.

## MANUS source-session OOF and calibrated session fusion

`feature_bank_manus_nested_oof_20260915` uses only session 1 of users 3–8, with
outer held pairs (3,4), (5,6), (7,8) and two inner subject folds per outer fold.
All 108 source trials have one raw and calibrated OOF prediction for each of eight
providers, including real IMU. The 28-pair complementarity matrix and all inner/outer
fitted states and probabilities are retained. Replay checks 48 outer and 48 inner blocks
with zero error, recomputes all outer temperatures from inner OOF predictions, and
confirms partition integrity/state immutability. F0 source OOF log loss improves
2.4965→1.5089; temporal 2.0439→1.5062; all eight providers improve log loss without
changing their class order or F1. These are source-session cross-user calibration
results, not an OOF test on new sessions.

`feature_bank_manus_probability_{validation,final}_20260915` then reuses the original
session-1 full-bank states, fitting source-only temperatures to the saved raw OOF
predictions. The known users are shared between source and target sessions; target
sessions 2/3 do not enter probability fitting. The OOF manifest now states that target
evaluation data is unopened, rather than suggesting source and target user identities
are disjoint. No family/scaler/classifier refit occurs. Target-session calibration
trials supply only the allowed personal anchors, reliability and session signatures,
and are completely removed from evaluation. Source fitting metadata is identical in
both phases. Unsupported five-shot budgets remain explicit.

| Session-3 mean per-user F1 | 0/class | 1/class | 2/class |
|---|---:|---:|---:|
| Uniform population | 0.4440 | 0.4052 | 0.3704 |
| Combined without anchors | 0.4582 | 0.4271 | 0.4398 |
| Combined anchors/context/quality | 0.4582 | 0.5264 | 0.5574 |

Within each budget, methods share evaluation trials; different budgets remove different
trials. Combined log loss is 1.4287/1.4256/1.4342 versus uniform 1.4187/1.4123/1.4334,
so higher F1 is not a universal probability-quality gain. Session-2 combined F1 is
0.3822/0.3922/0.5019. All combined component/provider removals remain available.
Both target phases replay 450 probability arrays with zero error and unchanged source
state. This is supplementary evaluation on previously opened final sessions, using
the native six finger flexion-extension classes, not wearable pinch/fist/open accuracy.
Current integrity checks 92 source artifacts, 13032 copied rows and 363 explicit
partitions; all 106 tests pass with one skip. Full-bank wearing, day/posture, load/position
and real-quality failure-system coverage still requires work; DS2 remains unresolved.

## Full wearing-bank fusion with before-wearing OOF calibration

`feature_bank_wearing_full_fusion_{validation,final}_20260915` evaluates the fixed nine
non-IMU providers, including trace covariance, CSP and SPD alternatives, on the native
five-class task. Validation subjects are 15–17, final 18–20. Each subject's classifiers
fit only 25 before-wearing trials. Five repetition-held-out source folds refit every
family/scaler/classifier; source OOF probabilities alone fit each provider temperature.
All forty after-wearing trials remain evaluation-only, with zero target calibration.
Known subject identities are shared across wearing domains, while trial IDs are disjoint.
The new known-user trial-fold option retains the trial-leakage guard and the default
subject-disjoint guard; both are covered by a regression test.

Each phase exports 280 subject/pooled-domain cells covering all four after domains,
ALL, F0/raw-F0, raw/uniform-calibrated/full-quality fusion and all nine full-minus-family
variants. All source OOF states/probabilities, temperatures and target probabilities are
retained. No IMU or target anchors/session signatures are invented for this zero-target-
calibration comparison. Quality is a source-fit heuristic, not hardware clipping truth.

| Final pooled method | ALL F1 | Worst after-domain F1 | ALL log loss |
|---|---:|---:|---:|
| Source-calibrated F0 | 0.4912 | 0.3938 | 2.4877 |
| Raw uniform nine-bank | 0.6147 | 0.5136 | 0.9773 |
| Source-calibrated uniform bank | 0.5907 | 0.4729 | 1.0250 |
| Source-calibrated quality-weighted full bank | 0.5917 | 0.4477 | 1.0412 |
| Full minus ring | 0.5566 | 0.4525 | 1.1769 |

All methods' worst domain is trial_2. The full calibrated package improves ALL and
worst-domain F1 over F0, but source-only probability calibration regresses against raw
uniform fusion, and quality weighting further reduces worst-domain F1. Removing ring
hurts ALL F1 while slightly improving the full system's worst domain. Final outcomes
are reported without replacing the fixed package using final-subject selection.
This is supplementary evaluation on previously opened final subjects, not a new blind
test. Before-domain calibration cannot be assumed to improve shifted-domain probabilities.

Replay checks 69 source-OOF/target arrays per phase, recomputes all temperatures from
replayed source OOF predictions, verifies source trial partitions and target identifiers,
and confirms state immutability without fitting. Target probability error is zero in
both phases. All 107 tests pass with one skip. Other failure-system comparisons and
historical DS2 remain incomplete.

## UniBo full four-channel package: day/posture negative transfer

`feature_bank_unibo_full_source_20260915` supplies eight non-ring, non-oracle providers.
Internal calibration classifiers/families fit only days 1–4 and predict day 5. Those
source-day probabilities fit temperatures in [0.25,4]. Final provider classifiers fit
days 1–5, reusing the exact previously audited source-fit family states. The internal
and final classifiers use the original hierarchical segment weights; temperature fitting
uses unweighted day-5 windows, as explicitly recorded. This source-day heldout scheme
is not a full nested OOF experiment. Source fitting covers 50607 windows, calibration
24039; target days 6–8 are never opened during source-package construction.

Actual provider dimensions are F0 24, X1-H 4, trace covariance 10, CSP 8, SPD 10,
spectral 40, temporal 29 and quality 29 (154 total). Input remains native four-muscle
topology; the processed windows are resampled to 200 Hz from verified 500 Hz acquisition.
Neither synthetic eight-channel ring geometry nor oracle posture enters classification.

`feature_bank_unibo_full_fusion_{validation,final}_20260915` evaluates day 6 and days
7–8, respectively, with no additional fitting or target calibration. All eight provider
removals, raw/calibrated uniform fusion, quality-weighted fusion and raw/calibrated F0
are saved, including subject-day-posture cells. Validation/final export 533/910 metric
rows over 24338/48818 windows, using the original hierarchical metric weighting.

| Final method | ALL F1 | Minimum posture F1 | ALL log loss | Minimum subject-day-posture F1 |
|---|---:|---:|---:|---:|
| Raw F0 | 0.7044 | 0.6638 | 0.4359 | 0.3442 |
| Raw uniform bank | 0.6944 | 0.6555 | 0.6999 | 0.3474 |
| Calibrated uniform bank | 0.6994 | 0.6600 | 0.7042 | — |
| Calibrated quality-weighted full bank | 0.6992 | 0.6594 | 0.6860 | 0.3280 |

Posture 2 is the weakest pooled posture. Full-bank day-7/day-8 F1 is 0.7082/0.6905
versus F0 0.7160/0.6927. The full package underperforms the original F0 backbone
overall, by posture and in its worst individual cell; its probability loss is substantially
higher. The earlier frozen F0+temporal expert remains stronger (F1 0.7102). Source-only
calibration therefore does not remove harmful averaging of weak providers. This result
is retained rather than selecting providers on the final days. These are supplemental
comparisons on previously opened final days, not a new untouched final test.

Source replay checks all eight heldout probability arrays, recomputes temperatures,
verifies source partition coverage and immutable state, with zero error. Target replays
check all 13 variants per phase, exact labels/trials/users/days/postures and unchanged
source state, also with zero error. Full tests remain 107 passing cases with one skip.
Current table integrity covers 96 sources, 15035 rows and 401 explicit partition checks;
source calibration uses an additional dedicated partition audit. Load/position full-bank
evaluation and the historical DS2 requirement remain open.

## EMG-only load/position full-bank source-condition OOF

`feature_bank_load_position_full_{validation,final}_20260915` evaluates a fixed nine-
provider package on validation subjects 1–3 and final subjects 4–6. Load-shift training
uses only 0g across eight positions; source probability folds hold out position pairs.
Position-shift training uses only position1 across five loads; source probability folds
hold out individual loads. Every family/scaler/window classifier is refit inside each
source fold. Source OOF trial-mean probabilities alone fit temperatures, which are
applied to final trial-mean provider predictions. Target load/position trials never fit
models or probabilities, and no target personal calibration is used.

Only EMG columns 9–16 enter the adapter; FMG is excluded. Acquisition is 2000 Hz,
with the original central-nine-second crop and eight sparse windows per trial. External
loading remains distinct from voluntary contraction intensity. Provider dimensions are
F0 48, X1-H 8, trace covariance 36, CSP 8, SPD 36, ring 40, spectral 72, temporal 57,
quality 53 (358 total), verified consistent across subjects/scenarios. Providers use
independent L2-regularized classifiers, not one concatenated high-dimensional classifier.

Each phase exports 728 subject/mean-subject condition cells, raw/calibrated F0,
raw/calibrated uniform fusion, quality fusion and all nine provider removals. The
reported F9 provider removal retains its separate quality routing; removing both F9
roles jointly remains a distinct component-ablation requirement. Uniform-calibrated
fusion supplies the full-bank routing-off control.

| Final mean per-user method | Load ALL F1 | Worst load F1 | Position ALL F1 | Worst position F1 |
|---|---:|---:|---:|---:|
| F0 | 0.6239 | 0.5837 | 0.6112 | 0.2978 |
| Raw uniform bank | 0.7648 | 0.6756 | 0.6652 | 0.2846 |
| Calibrated uniform bank | 0.7547 | 0.6505 | 0.6514 | 0.2745 |
| Calibrated quality full bank | 0.7400 | 0.6128 | 0.6622 | 0.2800 |

Raw F0 reproduces the original final baseline exactly. The raw bank substantially
improves load F1 but slightly worsens its log loss (0.8356→0.8390); source calibration
improves bank log loss to 0.7791 while reducing F1. Quality routing further reduces
load F1/worst-load F1. For position shifts, raw-bank log loss improves 1.1270→0.9665,
but its worst-position F1 declines. Source-calibrated F0 sharpens the wrong shifted
predictions, increasing position log loss to 2.0938; calibration is not uniformly safe
under nuisance shift. These supplemental results use previously opened final subjects
without selecting a new package from their scores.

All source-fold states/predictions, temperatures and evaluation identifiers are retained.
Replay checks 138 source-OOF/target arrays per phase, recomputes source temperatures,
verifies source/target trial partitions and confirms immutable fitted state without
refitting. Target probability error is zero in both phases. The 107-test suite passes
with one skip. Disk usage has been remeasured in `benchmarks/discovery/DISK_USAGE.json`:
raw 17169777002 bytes, processed 347367384, external manifests 6838214 (timestamped
snapshot, excluding this repository's own small files). Joint quality-role removal,
remaining full-system evidence and historical DS2 still require work.

## Joint F9 routing/provider removal

`feature_bank_{wearing,load_position,unibo,manus}_quality_roles_{validation,final}_20260915`
separates four controls: full system, routing off, provider off, and both roles off.
Earlier `without_F9_Quality` rows removed its classifier provider while retaining
quality routing and must be read as provider ablations, not removal of the entire F9
module. The new joint control fixes that evidence gap for zero target-calibration
budgets. MANUS uses its original zero-shot combined system, where personal anchors
and session-context weight updates are inactive; calibrated nonzero budgets are outside
this new ablation's scope. Real IMU providers retain unit quality weight when EMG quality
is rejected. Regression tests verify both IMU preservation and joint prediction invariance
to changes in the removed F9 provider and routing inputs.

| Final native task | Full F1 | Joint F9 removal F1 | Full worst condition | Joint worst condition |
|---|---:|---:|---:|---:|
| Wearing | 0.5917 | 0.5827 | 0.4477 | 0.4829 |
| External load | 0.7400 | 0.7755 | 0.6128 | 0.7045 |
| Limb position | 0.6622 | 0.6329 | 0.2800 | 0.2404 |
| UniBo day/posture | 0.6992 | 0.6925 | 0.6594 | 0.6532 |
| MANUS session/speed | 0.4582 | 0.4439 | 0.3552 | 0.3690 |

Condition minima use the reported native wearing/load/position/posture/speed factors;
they are not minima across failure dimensions. UniBo/wearing metrics pool users with
their original weights; load/position/MANUS use mean per-user metrics. Joint removal
helps the external-load benchmark and wearing/speed condition minima, but hurts limb-
position and posture performance. F9 therefore cannot be categorized as universally
helpful or universally harmful. These controls remain diagnostic and are not selected
as deployment policies using the final scores.

The eight runs export 1120 metric rows and 176 role-variant probability arrays. All 44
full-system subject/scenario probability blocks match their original saved predictions
exactly; model state is unchanged, with source-state/prediction hashes retained. No
classifier, family or source probability fitting is repeated. These sample-level quality
controls do not constitute a labelled real-hardware noise benchmark. Current result
integrity checks 106 sources, 17611 rows and 467 explicit trial partitions. Full tests
run 109 cases, passing with one skip. Nonzero-budget joint F9 controls, remaining
research evidence and historical DS2 are still open.

### Validated UniBo implementation reuse audit

The authoritative documents prioritize existing validated implementations. A new
adapter calls the existing `PhysiologyFeatureTransformer` G0 and G5 directly, preserving
their feature names, weighted source fitting and numerical outputs. Both remain native
four-channel implementations at the established processed 200 Hz; neither implies
validation on the current eight-channel hardware. Two compatibility tests compare
both groups exactly with the original implementation and reject eight-channel input.

The reference F0 and F5 are distinct implementations. In particular, original G5 has
20 dimensions and uses early-minus-late RMS and raw-waveform slope; reference F5 has
29 dimensions, late-minus-early RMS, envelope slope and additional temporal summaries.
Equal dimensions for G0 and F0 do not establish formula equivalence. Historical DS2
X1H/RLCS/CES/Frequency source equivalence remains unverified without its original files.

`feature_bank_unibo_validated_reuse_validation_20260915` fits only days 1–5 and evaluates
day 6, without opening days 7–8. All six predictors replay exactly on 24338 windows,
with unchanged fitted state and no fitting during replay.

| Source implementation | Dimensions | Day-6 macro-F1 | Log loss |
|---|---:|---:|---:|
| Reference F0 | 24 | 0.6554 | 0.5032 |
| Validated G0 | 24 | 0.6323 | 0.5054 |
| Reference F0 + reference F5 | 53 | 0.6733 | 0.4904 |
| Reference F0 + validated G5 | 44 | 0.6599 | 0.5020 |
| Validated G0 + validated G5 | 44 | 0.6374 | 0.5050 |
| Validated G0 + reference F5 | 53 | 0.6504 | 0.4964 |

These 72 validation rows document reuse and its measured differences; they do not
replace historical results or establish final-day improvement. The full test suite
now passes 111 tests with one skip. The complete document-level objective remains open.

### Quality-role controls with target calibration

`feature_bank_manus_calibrated_quality_{validation,final}_20260915` extends the joint
quality removal to supported cal0/1/2. F7 anchors, source-only reliability priors and
F8 context are held fixed across controls; each budget removes the same whole
calibration trials from every compared method. No classifier/family or source
temperature fitting is performed. All 450 original saved probability arrays per phase
replay exactly; 18 full user/budget blocks per phase also match before control export.
The two runs retain 664 metric rows, 144 variant probability arrays and explicit trial
partitions. Native speed rows with no evaluation trials after calibration are omitted,
not interpreted as zero performance. Cal5 remains unsupported (three trials/class).

| Final mean-user macro-F1 | cal0 | cal1 | cal2 |
|---|---:|---:|---:|
| Full F7/F8/F9 system | 0.4582 | 0.5264 | 0.5574 |
| Without quality routing | 0.4440 | 0.4744 | 0.5315 |
| Without quality probability provider | 0.4296 | 0.5364 | 0.5593 |
| Without routing and provider | 0.4439 | 0.5031 | 0.5593 |

At cal1, removing the quality classifier helps while removing both roles hurts;
at cal2, joint removal has a small positive F1 difference. Full log loss remains
worse than joint removal at each budget (1.4287/1.4256/1.4342 versus
1.4191/1.4205/1.4276). These final results describe frozen controls, without selecting
new policies on final data. Supported nonzero MANUS budgets now have joint controls;
historical DS2 and further document-level evidence remain open.

### Frozen full-bank robustness vector (cal0)

The previous expert-package envelope uses different selected feature combinations.
`benchmarks/full_system_robustness.py` now separately reports the frozen full-bank
algorithms, preserving native family availability, tasks, source-only probability
calibration and source/evaluation partitions. It does not select the best method in
each final cell. Source CSV hashes and explicit run references are retained in
`results/full_system_robustness_vector.{csv,json}`.

| Failure axis | F0 reference | Full bank | Difference |
|---|---:|---:|---:|
| Force (LibEMG unseen-force) | 0.4929 | 0.4516 | -0.0414 |
| Wearing (same-user before/after) | 0.4912 | 0.5917 | +0.1005 |
| Day (UniBo days 7/8) | 0.7044 | 0.6992 | -0.0052 |
| User (EPN held-out users) | 0.4370 | 0.4792 | +0.0422 |
| Posture (UniBo posture strata) | 0.7044 | 0.6992 | -0.0052 |
| Speed/session (MANUS session 3) | 0.4152 | 0.4582 | +0.0430 |
| Real labelled quality | N/A | N/A | N/A |

The requested descriptive mean over six available axes is 0.5409 → 0.5632 and
minimum axis performance is 0.4152 → 0.4516. Day/posture share the same observations,
so their overall values coincide and are not independent measurements. The vector
mixes native tasks and documented aggregation rules; these means are not pooled
accuracy or evidence for a universal classifier. F7/F8 are inactive at zero target
calibration, and quality routing is absent in force/EPN full-bank runs. Current-device
validation still requires controlled local trials.

Worst observed conditions are recorded separately from the minimum axis: wearing
0.3938 → 0.4477, day 0.6927 → 0.6905 and posture 0.6638 → 0.6594. Speed's full-bank
minimum is 0.3552; its paired F0 speed-stratum minimum is unverified and remains N/A.
Force's eleven-condition minima come from the matching frozen probability run's
condition report. This evidence establishes a mixed outcome: average and minimum-axis
improvement coexist with declining force/day/posture coordinates. It does not prove
that the entire robustness envelope has improved. Remaining source reuse, scientific
diagnostics and document-level deliverables must still be audited.

### Source-user reliability hyperparameter selection

The earlier `n0=8`, reliability temperature `1` and uniform population weights are
prespecified baselines, not cross-validation-selected parameters. The documents
require source-user selection. `reliability_selection.py` now uses retained outer
source-user OOF family/classifier states, without refitting them or opening EPN
target users or MANUS sessions 2/3. Each held user's whole cal1/2 trials are excluded
from its evaluation trials. Population priors in each outer fold come exclusively
from that fold's inner-training OOF log losses, with weights proportional to
`exp(-LogLoss_k)`. No held outer user's labels enter these population priors.

The fixed grid is `n0 ∈ {2,8,32}` and `τ ∈ {0.5,1,2}`. Mean per-user/budget log loss
selects the rule, comparing reliability fusion without personal anchor probability
mixing so these two components remain distinguishable. MANUS source session 1 and
EPN source users 1–15 both select `n0=2, τ=0.5`; source selection losses are 1.4835
and 1.5253 respectively. These are tuning scores, not unbiased final estimates.
Deployment population priors are then derived from all source outer-OOF calibrated
probabilities. Frozen fold feature matrices, probabilities, priors, source hashes and
calibration/evaluation trial lists are retained outside Git; small metrics and audits
are committed. All 108 MANUS and 270 EPN selection probability arrays replay exactly.

This closes source selection evidence for the two full-bank calibration protocols;
the selected policies are not yet applied to their held-out target runs. Historical
fixed-rule results remain unchanged. Long-term/session-local prototype blending and
the remaining requirement audit also remain open.

### Applying the selected reliability policy

`feature_bank_{epn,manus}_selected_{validation,final}_20260915` applies the same
source-selected policy in both phases, reusing frozen source family/classifier states
and source OOF temperatures. Policy loading verifies exact source trials, native
dataset, family order, source OOF hash, no target access during selection and finite
normalized population weights. Two regression tests reject wrong dataset/trials,
target-data access and modified OOF provenance, and preserve selected parameters.
The population-only branch is explicitly named `population_only` because its source
prior is nonuniform. Historical `uniform_population` outputs remain unchanged.

| Final mean-user macro-F1 | cal0 | cal1 | cal2 | cal5 |
|---|---:|---:|---:|---:|
| EPN selected full anchor fusion | 0.4725 | 0.4053 | 0.4346 | 0.4115 |
| EPN selected without F7 anchor | 0.4725 | 0.4708 | 0.4670 | 0.4952 |
| EPN selected population-only | 0.4725 | 0.4752 | 0.4749 | 0.4802 |
| MANUS selected full F7/F8/F9 | 0.4631 | 0.5126 | 0.5926 | N/A |
| MANUS selected population-only | 0.4650 | 0.4315 | 0.3861 | N/A |

Against earlier fixed-rule full systems, MANUS improves at cal0 and cal2 but declines
at cal1; EPN improves only at cal2 and declines at cal0/cal1/cal5. EPN's anchor branch
still harms held-out-user transfer relative to the same selected system without F7.
Source CV scores therefore must not be treated as guarantees for new users. Selected
MANUS full log losses are 1.4186/1.4101/1.4190; EPN full losses are
1.4968/1.5399/1.5951/1.6074. Calibration budgets use different evaluation trial sets,
while all compared methods within a budget share the same remaining whole trials.

All 1164 saved prediction arrays replay with maximum error ≤1.67e-16 and unchanged
family/classifier state. The four runs add 1416 calibration metric rows plus their
leave-family tables. Full tests pass 113 cases with one skip. This completes application
of these source-selected reliability policies; long-term/session-local prototype
blending, remaining historical reuse and document-level evidence remain unfinished.

### Explicit long-term/session prototype fusion

`SessionPrototypeAnchor` preserves the source-session long-term prototypes and returns
a new adapted anchor. For each class, `β=N_long/(N_long+N_cal)` and
`μ_adapt=β μ_long+(1-β) μ_cal`. Source session 1 supplies three whole trials/class,
so β is 1/0.75/0.6 for cal0/1/2. This rule depends only on source counts and budget,
not target performance. Long-term, local and blended controls keep the same source
distance scale and source-only median-distance temperature. Their source/classifier
mixture uses `α=(N_long_min+shots)/(N_long_min+shots+2)`, or 0.6/0.667/0.714.
The local-only control has no available local prototype at cal0 and uses the source
classifier then. Long-term and blended controls can use the known user's long-term
profile at cal0. This is session protocol B, not new-user protocol A.

Each anchor outputs 14 coordinates (six distances, six similarities and two margins)
per family. Prototype storage is six times each original family dimension; the
adapted version keeps a separate copy, preserving the long-term profile. Source scale
is deliberately held fixed to isolate prototype displacement; this is distinct from
the older cal-MAD/local-scale anchor. Two tests prove the blend equation, missing-class
rejection, zero-budget behavior, preserved source profile and query-batch independence.

`feature_bank_manus_session_blend_{validation,final}_20260915_v2` reuses the selected
full-bank source models and source OOF temperatures. It compares all prototype modes,
no-anchor, all eight provider removals, F8 removal and F9 routing/provider controls.
All 36 original full-system user/budget blocks match exactly. The 576 newly exported
fused variant arrays also replay exactly from retained provider probabilities and
fixed weights/quality. No classifier/family fitting occurs. Explicit trial partitions,
prototype coefficients, source hashes, separate adapted anchors and 2656 metric rows
are retained. The initial validation export without fusion inputs is superseded by
v2 and excluded from consolidated evidence; its metrics are unchanged.

| Final mean-user macro-F1 | cal0 | cal1 | cal2 |
|---|---:|---:|---:|
| Long-term prototypes, source scale | 0.4582 | 0.5213 | 0.4139 |
| Local prototypes, source scale and matched α | 0.4631 | 0.6633 | 0.8241 |
| Blended prototypes, source scale | 0.4582 | 0.5398 | 0.6176 |
| Original local cal-MAD anchor and α | 0.4631 | 0.5126 | 0.5926 |
| No anchor | 0.4631 | 0.4538 | 0.3815 |

Blending helps over long-term-only at cal1/2 but underperforms the matched local-only
control. Its final log losses are 1.5241/1.4848/1.4324, versus local-only
1.4186/1.4222/1.3218. The fixed count-based rule may retain too much outdated session
information; final scores are not used to change β. Cal2 leaves only six evaluation
trials per user (36 total), and speed/session confounds remain. The strong local
result is a frozen control, not an independently selected and confirmed deployment
policy. Cal5 is unsupported. This six-active-class subset has no rest class, so a
rest-center channel normalization branch cannot be formally tested here. Full tests
pass 115 cases with one skip. Historical reuse and further complete-document audit
remain open.

### Required delivery schemas and missing evidence

`feature_bank/delivery` now contains all five required CSV filenames with explicit
required fields and provenance for every one of 22704 rows. The original source
tables remain unchanged. Recorded values take precedence; aliases and same-run
dataset identities are mapped only when supported. Manifest family lists can describe
the available bank, while method/removal fields specify the actual ablation. Domain
metadata inherited through source reuse is restricted to matching phases so a final
run cannot acquire a validation target session. Metadata notes identify every derived
field; unrecoverable values remain N/A.

`delivery/SCHEMA_AUDIT.json` audits missing fields by artifact/run, including old
unrecorded pooled-subject/domain metadata and unavailable unsupported-budget metrics.
`PROVENANCE_AUDIT.json` verifies exact source record coverage, hashes and all recorded
required values. This is explicitly `schema_complete_evidence_partial`: it does not
certify scientific completion. Two regression tests reject changed canonical values
and prevent inheritance of another phase's target session. The full suite now passes
117 tests with one skip. Source repairs, historical DS2 reuse, remaining diagnostics
and the complete requirements audit remain open.

### Explicit calibration-relative spectral coordinates

`LogBandEnergyFamily` reuses the existing spectral band's frozen Nyquist-safe
definitions and tapered-periodogram convention. It outputs 32 log-band energy
coordinates for native eight-channel EPN (16 for four channels).
`RelativeSpectrumCoordinates` explicitly fits a calibration-only mean log-band
reference and returns `log(E_current+ε)-μ_cal`. It never updates the reference from
evaluation samples and rejects channel/rate mismatch. Two tests verify the exact
log-power gain behavior, frozen reference and query-batch independence.

`feature_bank_epn_relative_spectrum_source_20260915` trains raw log-band and
source-user-centered log-band logistic classifiers on source users 1–15 only.
Source user references use five labelled trials/class. Three outer source-user folds
refit both classifiers/scalers and exclude simulated cal5 trials completely from
OOF evaluation. Each fold's held user's spectral reference uses only its calibration
trials. Source-only OOF probabilities fit classifier temperatures; all six source
probability blocks replay exactly and both temperatures recompute exactly. Final
source classifiers and references are frozen before target phases.

`feature_bank_epn_relative_spectrum_{validation,final}_20260915` then compares the
raw and relative branches at cal0/1/2/5 on users 16–18 and 19–21. Cal0 uses a source
population reference; nonzero references use remaining-target-excluded calibration
trials only. The two target runs add 64 metric rows and 48 probability arrays, all
replayed exactly from retained trial features and frozen references/models.

| Final mean-user macro-F1 | cal0 | cal1 | cal2 | cal5 |
|---|---:|---:|---:|---:|
| Raw log bands | 0.2607 | 0.2625 | 0.2549 | 0.2702 |
| Calibration-relative log bands | 0.2577 | 0.2470 | 0.2723 | 0.2984 |

Final relative-branch log losses are 1.6973/1.6967/1.6890/1.6689 versus raw
1.6976/1.6951/1.6956/1.6872. Cal2/5 show local benefits, but validation relative F1
declines at every nonzero budget (cal5 0.2683→0.2405). This isolated log-band view
also remains much weaker than the broader existing spectral/full-bank views. It is
not a replacement selected on final results, a fatigue estimate or a claimed exact
CCA reproduction. Source temperature calibration simulates cal5 and may transfer
imperfectly to other budgets. Full tests pass 119 cases with one skip. Historical
reuse and remaining complete-document evidence remain unfinished.

`PersonalSessionSpectralShift` now separately freezes an equal-trial-mass
long-term log-band reference and a current-session calibration reference. It
returns both F4d `window−long` and `session−long` coordinates, and rejects
overlap among source, session calibration and evaluation trial identities.
The [Song diagnostic](../benchmarks/song_real8/F4D_SESSION_SHIFT_AUDIT.json)
uses S01/S02 initial-rest blocks for the long reference and disjoint S03/S04
initial-rest blocks plus held-out still-neutral formal trials. Mean absolute
session-minus-long log-band shift is 1.048 in S03 and 1.022 in S04 (32
coordinates). These are one-person/day, unfiltered periodogram context values;
they neither establish fatigue nor show a classification gain. S01–S03 whole
sessions did not pass collection readiness, although the selected individual
blocks were marked valid and source file hashes were checked.

### Calibration activation range and within-gesture pattern spread

`PersonalActivationProfile` now records the document's signal-range quantities:
`A_raw=sqrt(mean_channel(RMS_channel²))`, q10/q50/q90, and within-gesture spatial
pattern spread. Trial weights give each whole calibration trial equal mass;
quantiles use the frozen inverse weighted empirical CDF without interpolation.
Pattern coordinates use the new reference `RMS/global_RMS`, not a claimed historical
X1-H reproduction. The pooled within-gesture spread is the mean of per-gesture
spreads, explicitly separate from pooled between-gesture variation. Zero-activation
windows are excluded from pattern calculations rather than treated as meaningful
spatial patterns.

`benchmarks/export_activation_profiles.py` reuses exact cal1/cal2 Ramp trial IDs from
the force full-bank probability runs (validation users 7/8, final users 9/10).
It opens no unseen-force conditions, fits no classifiers and uses no evaluation
samples. Original `Info.txt` identifies No Movement as the first native class, matching
C1/adapter label0; these windows are excluded from active-range calculations.
The source metadata and split hashes are retained in `activation_profile_audit.json`.
`personal_activation_profiles.{csv,json}` retain eight profiles and 64 records,
including explicit unavailable cal0 and unsupported cal5 entries.

| Final subject / Ramp calibration | q10 | q50 | q90 | Mean within-gesture pattern spread |
|---|---:|---:|---:|---:|
| User 9 / cal1 | 0.1267 | 0.3186 | 0.5738 | 0.4381 |
| User 9 / cal2 | 0.1297 | 0.3206 | 0.5738 | 0.5373 |
| User 10 / cal1 | 0.0848 | 0.1618 | 0.3063 | 0.3804 |
| User 10 / cal2 | 0.0870 | 0.1641 | 0.2973 | 0.4976 |

These are archive signal units, using eight sparse 200 ms windows per trial. Ramp
in this public dataset uses 20–80% MVC feedback: the profile describes that observed
calibration signal range, not measured mechanical force, unconstrained natural
product use or fatigue. It does not change existing classification results. Three
tests verify rest exclusion, equal-trial weighting under window duplication, rejection
of mixed-label/rest-only trials and separation of within/between-gesture variation.
Full tests pass 122 cases with one skip. Remaining complete-document evidence and
historical baseline reuse remain open.

### Family-specific session shift summaries

`FamilySessionShiftSummary` extends the generic residual/cosine/geometry descriptors
with explicit class-matched changes in log global activation, scale-pattern vectors,
mean absolute log-band energy, ring vectors, trace-normalized covariance and channel
variance/quality scores. Covariance change uses the affine-invariant SPD distance
`||log(C_long^(-1/2) C_cal C_long^(-1/2))||_F`, with positive-definiteness checks.
Source class profiles use equal trial mass; calibration transforms never mutate them.
Native ring summaries require verified eight-channel topology. New pattern/ring views
do not establish historical X1-H/RLCS formula equivalence.

`export_session_shift_summaries.py` reuses exact cal1/2 trial IDs from the selected
MANUS full-bank runs. Source session 1 builds each known user's long-term profile;
session 2/3 calibration portions supply current profiles. No evaluation samples are
used for the signatures and no classifiers are fitted. The six-active-class subset
has no rest data, so rest-noise changes remain unavailable. Quality differences are
relative rule scores; absent ADC limits prevent physical clipping interpretation.

`family_specific_session_shifts.{csv,json}` retains 24 signatures and 144 class-matched
rows, source/split hashes and sensor contracts. Each class exports six scalar shifts
plus eight channel variance changes (14 numeric values); generic F8 residual norms,
cosine agreements and class geometry remain separate existing descriptors.

| Mean across class/user signatures | Validation cal1 | Validation cal2 | Final cal1 | Final cal2 |
|---|---:|---:|---:|---:|
| Log global activation shift | -0.2368 | -0.2334 | -0.3119 | -0.3307 |
| Mean absolute log-band residual | 0.9850 | 0.8727 | 0.9956 | 0.9050 |
| Affine-invariant covariance distance | 2.2237 | 2.0804 | 2.5719 | 2.5188 |
| Ring-vector residual norm | 0.1957 | 0.1614 | 0.2008 | 0.1495 |
| Mean quality-score shift | -0.0161 | -0.0117 | -0.0235 | -0.0180 |

These are diagnostic context descriptors, not newly evaluated gating policies or
fatigue/force estimates. Two tests check known SPD geometry and show that uniform
gain changes log amplitude and log power while leaving pattern/covariance shape
unchanged; source state remains immutable. The full suite passes 124 tests with one
skip. Integration/performance evidence for these additional descriptors and the
remaining complete-document audit remain unfinished.

### Active-only force mechanism audit

The force task's native seven-way classification keeps No Movement, but the documents
exclude rest from force-level physiology/mechanism comparisons.
`benchmarks/active_force_diagnostics.py` now separately computes these diagnostics
for active native labels 1–6, reusing the seven full-bank source family/scaler/classifier
states without fitting or changing historical results. Original `Info.txt` and source
state hashes are retained. `active_force_family_diagnostics.csv` contains 336 rows
for validation/final users, seven providers and eleven force conditions plus ALL.

Within each user, D_nuisance is mean same-active-class centroid distance across
force-condition pairs; D_gesture is mean active-class centroid separation per
condition. Both use source-standardized coordinates. J is their ratio, not mutual
information. Target labels appear only in offline class-matched diagnosis, never in
transforms or model fitting. Active-only F1 evaluates six active classes, while log
loss keeps original seven-way probabilities so mistaken rest mass is penalized.

| Final mean-user active diagnostic | D_nuisance | D_gesture | J | Active-six macro-F1 |
|---|---:|---:|---:|---:|
| F0 | 8.0294 | 9.1959 | 1.1314 | 0.4482 |
| New reference X1H view | 1.6681 | 3.1495 | 1.8839 | 0.3853 |
| CSP | 2.0679 | 4.3287 | 2.0851 | 0.4123 |
| Ring candidate | 4.8407 | 5.6328 | 1.1641 | 0.2229 |
| Spectral state | 7.3773 | 9.5281 | 1.2882 | 0.3836 |
| Temporal form | 7.8406 | 8.3478 | 1.0613 | 0.3275 |
| Quality provider | 7.7737 | 7.8040 | 0.9946 | 0.3429 |

This explicitly demonstrates why lower nuisance sensitivity alone is insufficient:
the scale-pattern view has much smaller drift and higher J than F0, yet worse active
standalone F1. These are new-reference LibEMG results, not a reversal or reproduction
of historical DS2 X1-H evidence. Force-condition/session confounds and source-coordinate
scaling remain limitations. Historical seven-class scores are preserved and must not
be compared directly with the six-class macro-F1 above. This audit covers the seven
retained providers; it does not certify every internal candidate or all document-level
requirements.

### Quality observation formula and capability audit

The legacy `QualityFamily.flatline_fraction` is a fraction of near-equal adjacent
samples, not the document's longest consecutive flatline run. Historical transforms
must remain reproducible, so `QualityObservabilityFamily` is a separate candidate.
It appends longest consecutive near-equal-difference run divided by window length,
per-channel correlation anomaly and availability-aware low-frequency power ratios.
Flatline tolerances are frozen from source median absolute adjacent differences,
with an explicit numerical floor; correlation median/MAD are source-only. Ring-neighbor
mode requires verified native eight-channel topology. ADC clipping, usable spectral
line bins and pre-highpass low-frequency observation each have explicit availability
flags. Unknown metrics cannot be interpreted as clean measurements.

The native eight-channel candidate has 80 dimensions: legacy 53, three appended
eight-channel observations, and three availability flags. Input channel, sample rate
and window length must match the fitted sensor contract. Low-frequency power is
computed only with an explicit available pre-highpass observation and a usable EMG
denominator band. Original force `Info.txt` documents 450 Hz lowpass filtering but
does not prove absence of upstream highpass filtering; native low-frequency ratio
therefore remains N/A, as does clipping without ADC range metadata.

`quality_observability_controls.csv` retains 16 diagnostic controls on the exact
recorded Ramp cal1 windows for users 7–10. Source thresholds fit users 1–6 Ramp only;
source state remains unchanged, and no evaluation recordings or classifiers are used.
For contiguous versus fragmented channel-3 flatline injections, legacy fractions
are almost identical (0.4008/0.4006), while longest-run ratios separate them
(0.3950/0.0083). Uncorrupted longest-run ratio is 0.0028. These are synthetic
observation diagnostics, not labelled real-hardware noise results or evidence that
new gating improves recognition. Native low-frequency measurements remain unavailable
even in the exported injection table; a separate known-generated-waveform unit test
verifies that an added 5 Hz component increases the implemented ratio.

Three tests verify longest-run geometry, explicit availability/source immutability
and controlled low-frequency contamination response. Full tests pass 127 cases with
one skip. The legacy quality feature implementation and every historical prediction
remain unchanged. Candidate quality fusion performance and remaining document-level
evidence are not certified by these observation controls.

### Per-subject variation and class summaries

`benchmarks/per_subject_analysis.py` derives 3342 subject/condition summaries from
unchanged family, calibration and full-bank tables. Each retains run, method, budget,
native condition, scenario, protocol and hyperparameter identity; conflicting duplicate
subject metrics are rejected. It reports mean/population-standard-deviation/min/max
user F1, mean user log loss, user IDs and actual mean per-class F1 JSON where every
underlying subject supplies compatible numeric class values. Missing class summaries
remain N/A. This supplies interpretable class summaries without overwriting historical
ALL rows that contain textual aggregation placeholders. Source CSV hashes are retained
in `results/per_subject_analysis_audit.json`.

The direct answer to question E for current EPN anchor fusion is negative:
`results/anchor_cross_user_variation.csv` compares full and no-anchor branches on the
same observed users and remaining evaluation trials within each budget. Reliability,
source classifier probabilities and calibration budget are held fixed.

| Selected-policy final EPN | No-anchor mean/std F1 | Anchor mean/std F1 | No-anchor/anchor minimum F1 |
|---|---:|---:|---:|
| cal0 | 0.4725 / 0.0303 | 0.4725 / 0.0303 | 0.4307 / 0.4307 |
| cal1 | 0.4708 / 0.0290 | 0.4053 / 0.0994 | 0.4329 / 0.2697 |
| cal2 | 0.4670 / 0.0290 | 0.4346 / 0.1429 | 0.4433 / 0.2376 |
| cal5 | 0.4952 / 0.0658 | 0.4115 / 0.1008 | 0.4194 / 0.2701 |

All nonzero budgets increase observed cross-user variation and lower both average
and minimum F1. The earlier fixed policy shows the same pattern, with std increases
of 0.0934/0.1001/0.0474 at cal1/2/5. These are descriptive results on three final
users, not statistically significant population claims or proof that all personal
anchor methods fail. They specifically identify the current distance-to-prototype
probability mixing branch as harmful relative to its otherwise identical no-anchor
control. Different budgets exclude different calibration trials and cannot be treated
as repeated measurements on one identical test set. No target-score tuning or model
fitting is performed for this analysis. Complete-document audit remains unfinished.

### Multi-family Core conditional analysis and finite source OOF interactions

The previous conditional tables used F0 alone as Core. The new EPN source-only
analysis freezes Core = F0 + Ring + CSP + IMU and tests four additional providers.
All 2,250 source-user trials are outer-held OOF predictions; provider temperatures
come from the corresponding inner-source OOF folds. No development/final users are
opened, no classifiers are refitted, and no combination weights are optimized.
Core averages four six-class probability views (24 input coordinates); each addition
averages five views (30 coordinates). This is a predictive conditional proxy for
fixed uniform late fusion, not a concatenated-feature classifier or mutual information.

| Added provider | Delta log loss (positive is improvement) | Delta macro F1 | Delta Brier | Users with improved log loss / 15 |
|---|---:|---:|---:|---:|
| X1 reference | -0.029259 | -0.011332 | -0.001952 | 0 |
| Spectral | 0.014360 | -0.000379 | 0.001358 | 12 |
| Temporal | 0.009591 | 0.005373 | 0.000740 | 14 |
| Quality | 0.033351 | 0.009405 | 0.002712 | 14 |

Error complementarity compares Core directly against each individual provider,
using the same outer-held trials. Two prespecified combinations are tested around
F0: X1/frequency and X1/CSP. Their negative-log-loss interaction scores are
0.041973 and 0.041554; macro-F1 interaction scores are 0.011326 and 0.014663.
Positive interaction measures model-composition synergy under probability averaging;
it does not establish a physiological factor interaction. Original historical X1-H
and Frequency equivalence remains unverified. Other documented pairs and target-held
multi-family Core comparisons remain open; this supplement does not close Stage 2–4.

The run is `feature_bank_epn_core_incremental_20260915`. Eleven saved probability
arrays replay exactly. Source-user coverage, nested trial separation and fold IDs
are checked; regression tests reject overlapping outer trials and target-user input.
The suite ran 129 tests with one skip. Consolidated integrity verification covers
127 source artifacts, 24,539 rows and 678 explicit partition checks; canonical
record verification covers 23,072 rows. These checks establish recorded integrity,
not full-document scientific completion or live-device accuracy. New work stays local.

### Independent-user confirmation of the fixed multiview Core

The exact source-prespecified eleven compositions were evaluated on EPN validation
users 16–18 and final users 19–21, each 450 trials at cal0. Providers reuse frozen
family/scaler/classifier states and original source-user OOF temperatures. Target
labels are used only for scoring; no target calibration or weight selection occurs.
ALL rows pool trials, while all three individual-user results are also recorded.
They must not be confused with earlier mean-per-user full-system summaries.

| Added provider | Validation delta LL / F1 | Final delta LL / F1 |
|---|---:|---:|
| X1 reference | -0.027571 / -0.003192 | -0.027002 / -0.002668 |
| Spectral | 0.032392 / 0.023522 | 0.013504 / -0.004316 |
| Temporal | 0.010141 / -0.011169 | 0.019364 / 0.000872 |
| Quality | 0.033513 / 0.016462 | 0.040057 / -0.009444 |

The X1 addition remains harmful in both phases. Spectral, temporal and quality
providers improve log loss in both phases, but their macro-F1 gains do not transfer
consistently. In particular, the source/validation quality-provider F1 gain reverses
on final users. A better probabilistic score does not imply more correct gestures.
No provider or composition is promoted based on these final results.

Runs `feature_bank_epn_core_incremental_validation_20260915` and
`feature_bank_epn_core_incremental_final_20260915` record identical fixed interaction
pairs, per-user complementarity and explicit source/evaluation trial partitions.
Each phase reproduces three original population fusion arrays; all eleven new
composition arrays replay exactly from retained individual-provider probabilities.
The suite ran 131 tests with one skip; integrity checks cover 135 artifacts,
24,707 rows and 680 explicit split checks. Canonical verification covers 23,224 rows.
Multi-family Core evidence now includes source OOF and independent EPN users;
other datasets, calibrated Core comparisons and original historical algorithms
remain incomplete. This is still a late-fusion proxy, not a concatenated-feature
classifier experiment or an end-to-end live-device improvement claim.

### Requirement triage and next scientific gap

`REQUIREMENT_AUDIT.csv` and `.json` bind 27 primary requirements and A–H questions
to current evidence files and hashes. This is a triage, not the exhaustive final
numbered/formula audit. File presence does not prove scientific completion. Historical
DS2 reproduction remains missing; most requirements retain partial status. Narrow
verified observations retain their dataset/user/method boundaries.

The next available substantive gap is labelled synthetic-corruption classification.
Current quality controls demonstrate signal-observation behavior, but do not measure
whether a frozen classifier or quality-routed fusion maintains gesture discrimination
under those corruptions. Consequently the quality robustness axis remains unavailable.
Other priorities are remaining named complementarity pairs, Core evidence beyond EPN,
exact historical algorithm reproduction and the final coherent requirement audit.

### Paired synthetic quality classification closes the missing synthetic axis

Eight fixed scenarios (clean plus seven synthetic perturbations) were applied to
identical native LibEMG force trial windows for independent users 7/8 and 9/10.
Noise scales use median source-window RMS; synthetic saturation uses source absolute
sample q99. Gaussian scales 0.25/0.5, 50 Hz line noise, saturation, channel-3 dropout,
half-window channel-3 flatline and channel-3 gain 2 are prespecified. Native source
users 1–6 Ramp family/scaler/classifier states and source-only OOF temperatures remain
frozen. There is no target calibration, corruption-specific refit or final-score tuning.

Both phases reproduce four original clean F0/uniform-fusion probability arrays.
Thirty-two fusion arrays per phase replay from retained provider probabilities and
quality scores with maximum absolute error below 4e-16. Both fitted state immutability
and source artifact hashes are checked. Regression tests check perturbation input
immutability, frozen source scaling and unaffected coordinates.

| Final pooled macro F1 | F0 | Original uniform full bank | Exploratory quality routing |
|---|---:|---:|---:|
| clean | 0.5118 | 0.4737 | 0.4520 |
| Gaussian 0.25 | 0.5205 | 0.4186 | 0.3857 |
| Gaussian 0.5 | 0.4018 | 0.3589 | 0.3350 |
| 50 Hz line | 0.4906 | 0.4128 | 0.3900 |
| source-q99 saturation | 0.5538 | 0.5094 | 0.4619 |
| channel-3 dropout | 0.3586 | 0.2941 | 0.1516 |
| half flatline | 0.4602 | 0.4253 | 0.3724 |
| channel-3 gain 2 | 0.4615 | 0.4177 | 0.3955 |

The quality robustness vector uses mean **individual-user/scenario** F1 across seven
perturbations, excluding clean: F0 0.4473, original uniform bank 0.3875. Worst scenario
mean-user F1 is 0.3632 versus 0.3100. Exploratory routing uses legacy F9 min quality
for F0/CSP and mean quality for remaining providers; it makes performance worse and
is not promoted to deployment. Detectable channel failure does not prove that these
routing weights identify the provider that retains gesture information.

With the synthetic quality axis included, the seven-axis descriptive means are
F0 0.5275 and bank 0.5381, but the minimum axis is F0 0.4152 versus bank 0.3875.
The earlier six-axis summary excluded quality and is superseded for available-axis
coverage. The bank does **not** raise the complete observed minimum envelope.
These are unlike tasks with correlated axes: quality/force share trials and
posture/day share observations. No independent-axis statistical claim is made.

Runs `feature_bank_force_quality_validation_20260915` and
`feature_bank_force_quality_final_20260915` contribute 528 rows. The suite ran
133 tests with one skip. Integrity checks cover 137 source artifacts, 25,235 rows
and 682 explicit partition checks; canonical record verification covers 23,752 rows.
Synthetic source-q99 saturation is not known ADC clipping; sparse-window corruption
is not a continuous hardware simulator or measured re-donning/noise evidence.
Historical algorithm reproduction, remaining named pairs, Core coverage beyond EPN
and complete-document audit still remain open. New results remain local.

The matched quality-role controls reinforce this failure: mean user/scenario F1
is 0.3875 with the quality classifier, 0.4209 when that provider is removed,
0.3384 with quality routing and provider, and 0.3884 with routing but no quality
classifier provider. Removing the provider improves this particular fusion while
remaining below F0 0.4473. These fixed controls are diagnostic, not final-selected
replacement algorithms.

### Original UniBo G5 versus reference temporal model complementarity

Four prespecified model pairs use the existing independent Day-6 predictions:
reference F0+validated G5 versus reference F0+reference F5; the same temporal
comparison around validated G0; and reference F0 versus each temporal augmentation.
The same concatenated-feature classifier protocol is retained. This compares
common-baseline models, not standalone temporal providers or DTW specialists.

Established hierarchical subject/day -> trial -> truth segment -> window weights
are reconstructed from saved metadata. Thirty original metrics across all six
models reproduce before new analyses are emitted. Sixty-four complementarity rows
cover pooled windows, seven users, four postures and four native gesture classes.
Source Days 1–5 and held-out Day 6 trial lists are disjoint and reproduced; final
Days 7/8 are not opened. No classifier fit or prediction changes occur.

For the reference F0 temporal pair, error correlation is 0.9045 and prediction
disagreement 0.0387. Validated-G5 model correct/reference-F5 model wrong mass is
0.0121; validated-G5 wrong/reference-F5 correct mass is 0.0203. In the OPEN class,
these masses are 0.0285 and 0.0478, with correlation 0.7436 and disagreement 0.0804.
Around validated G0, overall correlation is 0.9231 and recovery/loss masses are
0.0174/0.0100. The two models share most errors; the reference temporal branch
recovers more errors than it introduces in these comparisons, including OPEN.
This does not prove that adding both feature families improves a joint classifier,
that either standalone family is redundant, or that current eight-channel hardware
will recover the same errors. Native four named muscles and processed 200 Hz remain
distinct from the product eight-channel ring.

Run `feature_bank_unibo_temporal_complementarity_validation_20260915` records
per-class/per-user evidence and source artifact hashes. Weighted asymmetric-error
and undefined constant-error-correlation tests pass. The suite ran 135 tests with
one skip; integrity checks cover 138 artifacts, 25,299 rows and 683 explicit trial
partitions. Canonical verification covers 23,816 rows. Standalone G5/temporal/DTW
comparisons, exact historical RLCS/anchor comparisons and other remaining document
requirements are still incomplete.

### Concatenated-feature force Core and four conditional additions

The existing source-selected Core (`calibration_study.FROZEN_FAMILIES`) is frozen
as F0 + CSP + X1 reference. Source users 1–6 Ramp trials fit five new balanced
logistic classifiers after source-only scaling: Core 70 dimensions, +Ring 110,
+Spectral 142, +Temporal 127 and +Quality 123. Original source-fitted family states
are reused without modification. C=1 and max_iter=2000 are prespecified; every new
fit converges. Original standalone F0 and four additional-provider classifiers are
retained for matched error complementarity. No target data enters these fits.
Historical X1-H equivalence is still unverified; this Core contains the explicitly
labelled reference formula.

Validation users 7/8 and final users 9/10 are evaluated at zero target calibration
across eleven unseen force conditions. Unlike the earlier EPN late-fusion proxy,
these are actual concatenated-feature classifier comparisons. Each phase records
360 model-score rows, 144 Core conditional increments and 144 Core-versus-provider
complementarity rows across users and force conditions. ALL pools trial means;
it must not be confused with earlier mean-per-user or window-level aggregates.

| Addition | Validation delta LL / F1 | Final delta LL / F1 |
|---|---:|---:|
| Ring | -0.411435 / -0.050955 | 0.308815 / 0.063290 |
| Spectral | 0.143674 / 0.041118 | -0.117984 / 0.019426 |
| Temporal | 0.109059 / 0.014460 | -0.657338 / -0.036093 |
| Quality | -0.130708 / 0.017601 | -0.374717 / -0.056856 |

| Final model | Pooled trial macro F1 | Minimum force-condition pooled macro F1 |
|---|---:|---:|
| F0 | 0.5118 | 0.4102 |
| Core | 0.5176 | 0.4642 |
| Core + Ring | 0.5809 | 0.3944 |
| Core + Spectral | 0.5370 | 0.4355 |
| Core + Temporal | 0.4815 | 0.3894 |
| Core + Quality | 0.4607 | 0.3641 |

Core improves the observed force-condition minimum relative to F0. All four
additions lower that minimum relative to Core, even when pooled F1 rises. Ring's
large final average gain does not justify promotion after its validation decline.
Spectral's validation gain partly transfers to final F1 while final log loss worsens;
Temporal's validation gain reverses. Thus the source-selected representation retains
useful organization, while extra dimensions can reduce the worst-case envelope.
No family selection or weight adjustment is made using these final scores.

Runs `feature_bank_force_concat_core_source_20260915`, `_validation_20260915`
and `_final_20260915` preserve source fitting evidence and explicit trial splits.
Twenty classifier probability arrays replay exactly from retained target feature
coordinates and immutable source scalers/models; source fit and manifest hashes
are checked. The suite ran 137 tests with one skip. Integrity checks cover
144 artifacts, 26,595 rows and 685 explicit partitions; canonical verification
covers 25,112 rows. Calibrated Core comparisons, other dataset Core studies,
remaining named pairs and exact historical reproduction still remain open.

### Ramp-only calibration of the concatenated-feature force Core

Frozen source scalers/classifiers are reused for own-user whole-trial Ramp anchors
at 0/1/2 shots per native class. Calibration selection is nested and deterministic
(seed + user), covers all seven classes and excludes every unseen-force evaluation
trial and every source-fitting trial. The same evaluation trials remain at every
budget. No target-force calibration, scaler fit, classifier fit or target-score
rule selection occurs. Five-shot is explicitly unsupported, with blank scores and
blank actual calibration counts: each user has only four Ramp trials per class.

Mean prototypes, calibration-MAD standardized distances and the calibration-distance
median temperature follow the existing anchor protocol. Population and personal
probabilities mix with fixed alpha = shots/(shots+2). Every Core/extended model has
a matched no-anchor branch on identical evaluation trials. At cal0 the predictions
are exactly the prior concatenated-feature study. The calibrated ALL rows average
individual-user metrics, whereas that prior study's ALL rows pooled trials; macro F1
is nonlinear, so these summaries differ even with identical predictions.

| Final mean-user F1 | cal0 | cal1 | cal2 |
|---|---:|---:|---:|
| F0 | 0.4929 | 0.5062 | 0.5087 |
| Core | 0.4902 | 0.4943 | 0.5005 |
| Core + Ring | 0.5491 | 0.5531 | 0.5598 |
| Core + Spectral | 0.5037 | 0.5136 | 0.5192 |
| Core + Temporal | 0.4512 | 0.4545 | 0.4596 |
| Core + Quality | 0.4455 | 0.4507 | 0.4668 |

Calibration is helpful but limited for the frozen Core: two-shot recovery is only
0.0102 macro F1. Minimum force-condition mean-user F1 for Core rises from 0.4199
to 0.4328, while F0 remains 0.3614. Ring's higher average remains accompanied by a
lower minimum (0.3107/0.3107/0.3622); Spectral minimum is 0.3853/0.3884/0.3892.
Neither extra family therefore restores Core's observed force envelope. All variants
are reported; final scores do not select a deployment model or anchor weight.

Both phases retain own-user/model anchor states and source-coordinate inputs.
Each reproduces 60 anchored probability arrays and 60 matched no-anchor arrays
exactly. Source and target prediction hashes are checked. Tests verify nested own-user
selection, zero-shot behavior, rejection of unsupported budgets and duplicate whole
trial identities. Runs `feature_bank_force_concat_core_calibration_validation_20260915`
and `_final_20260915` contribute 7,260 rows including explicit unsupported budgets.
The suite ran 139 tests with one skip; integrity checks cover 150 source artifacts,
33,855 rows and 697 explicit partitions. Canonical verification covers 32,372 rows.
This closes the current force Core's offline calibrated comparison, while other
Core datasets, exact historical algorithms, remaining named pairs and final exhaustive
requirements audit remain incomplete. Current hardware performance is not established.

### Actual MANUS concatenated Core, temporal and real IMU increments

Core = F0 + SPD is the previous frozen MANUS shortlist. Source Session 1 for users
3–8 fits the SPD reference and three source-only logistic classifiers: Core 84
coordinates, Core+Temporal 141 and Core+real IMU 97. Existing source-fitted F0,
Temporal and IMU family/standalone classifier states are reused unchanged. Source
trial IDs and dataset match are verified. New classifier scaling and fitting never
open Sessions 2/3. Both target sessions reuse the exact same frozen source states.

| Addition | Session-2 validation delta LL / F1 | Session-3 final delta LL / F1 |
|---|---:|---:|
| Temporal | 0.319669 / -0.059015 | -0.048232 / 0.038887 |
| Real IMU | 0.047456 / -0.048124 | 0.077183 / -0.022569 |

| Final pooled trial model | Overall macro F1 | Minimum speed-cell macro F1 |
|---|---:|---:|
| F0 | 0.4453 | 0.3234 |
| Core | 0.4718 | 0.4339 |
| Core + Temporal | 0.5107 | 0.4903 |
| Core + IMU | 0.4492 | 0.3977 |

Core exactly reproduces the earlier standalone F0/SPD screening results. Temporal
has a final F1/minimum-cell gain with slightly worse final log loss, but validation
F1 declines. IMU improves log loss in both sessions while decreasing F1 and the
final minimum relative to Core. These demonstrate score/decision differences and
session-dependent increments; final-only gains do not select a deployment variant.
This study does not prove that all measured IMU features lack useful information.

Each target session has 108 complete trials, six known users and six finger flexext
classes without REST. There is no OPEN/pinch benchmark in this selected MANUS task.
Speed and session changes are confounded; the speed-cell minima are descriptive,
not isolated causal speed effects. ALL pools trials, while previous personal/session
calibration tables often average individual-user F1. No target calibration is used
for these cal0 comparisons; calibrated MANUS Core increments still remain open.

Runs `feature_bank_manus_concat_core_source_20260916`, `_validation_20260916`
and `_final_20260916` preserve source fits, dimensions, real context provenance,
trial partitions and 560 score/increment/complementarity rows. Twelve classifier
probability arrays replay exactly from retained feature coordinates and source fits.
Tests reject target-session source partitions and source-phase held-out reporting.
The suite ran 141 tests with one skip; integrity checks cover 156 artifacts,
34,415 rows and 699 explicit partitions; canonical verification covers 32,932 rows.
The complete robustness vector remains unchanged because this supplement does not
replace previously frozen full-bank methods with a favorable final-only model.

### Whole-session-trial calibration of actual MANUS concatenated Core

Frozen Session-1 models are compared with and without mean/MAD personal anchors
on identical remaining own-user Session-2/3 trials. Calibration is nested at 0/1/2
whole trials per class with seed + user. At two-shot, each user retains exactly one
trial per class. Scalers, feature families and classifiers remain unchanged; anchors
use calibration labels only, calibration-distance median temperature and fixed alpha
shots/(shots+2). No final-score model or anchor-weight selection occurs.

| Final mean-user F1, matched no-anchor / anchor | cal0 | cal1 | cal2 |
|---|---:|---:|---:|
| F0 | 0.4152 / 0.4152 | 0.3792 / 0.4100 | 0.2796 / 0.3278 |
| Core F0+SPD | 0.4161 / 0.4161 | 0.4040 / 0.4159 | 0.3102 / 0.3333 |
| Core + Temporal | 0.4470 / 0.4470 | 0.4396 / 0.4381 | 0.3185 / 0.3481 |
| Core + real IMU | 0.3957 / 0.3957 | 0.4122 / 0.4307 | 0.3241 / 0.2718 |

The current Core anchor gives small matched gains at both nonzero budgets, unlike
the much stronger earlier fine-tuning or supplemental local-prototype controls.
The IMU model's two-shot anchor harms performance; Temporal's one-shot anchor also
slightly harms F1. The differences are method-specific, not universal conclusions
about personal calibration. Cal0/1/2 evaluate 108/72/36 total whole trials and cannot
be subtracted as if they used one identical test set. Every matched no-anchor branch
removes exactly the same calibration trials as its anchor branch.

Some individual speed cells are empty after two-shot calibration. Their scores
are explicitly blank; corresponding increments and complementarity are unavailable.
ALL speed-cell metrics average only users with observed trials and state their
available-user count. Remaining speed cells can lack some truth classes, with
`classes_present` recorded; six-class macro F1 retains the task's six labels.
This is not an isolated speed-generalization experiment or an OPEN/pinch/rest task.
Five-shot has explicit unsupported rows; no actual trial counts or scores are invented.

Runs `feature_bank_manus_concat_core_calibration_validation_20260916` and
`_final_20260916` contribute 3,332 rows. Each phase replays 108 anchor predictions
and 108 matched controls exactly, checks immutable source/target hashes and reproduces
whole-trial calibration/evaluation indices. Tests verify own-user nested selection,
held-out class coverage and rejection of duplicate trials or exhaustive budgets.
The suite ran 143 tests with one skip. Integrity checks cover 162 artifacts,
37,747 rows and 735 explicit partitions; canonical verification covers 36,264 rows.
Exact historical data/algorithms, remaining named pairs, other eligible Core tasks
and exhaustive scientific requirement audit are still incomplete. New work stays local.

### Calibration burden updated for actual force and MANUS Core studies

The cost table and standalone performance/budget SVG now include nine protocols,
including force Core and Core+Spectral anchors, and MANUS Core/Temporal/IMU anchors.
Model and method filters identify the exact matching curve, avoiding accidental use
of a no-anchor or different-Core row from the same run. All sixteen original cost
rows retain their previously recorded values and approximate timing assumptions.

New Core signal-time estimates use actual selected whole-trial sample counts divided
by the native nominal sampling rate (force 1000 Hz; MANUS 200 Hz). Mean user durations
for force Core at 0/1/2 shots are 0/21.12/42.272 seconds for 0/7/14 Ramp trials;
MANUS Core requires 0/6/12 trials and 0/60.45/120.9367 seconds. Minimum/maximum user
durations and split hashes are retained. Model variants use the same calibration
trials at each budget. Five-shot remains unsupported without points or invented costs.

`results/calibration_burden_audit.json` retains each selected raw recording/member's
sample count, hash, nominal rate and signal duration, plus each exact source curve's
hash. These are recording-time estimates, not measured human/device calibration wall
time. Preparation, transitions and hardware delays remain outside the estimate.
The SVG explicitly warns that tasks, aggregation and evaluation trials differ;
curves cannot establish universal budget recovery or live-device accuracy.

Validation confirms 36 cost rows, unchanged sixteen legacy rows, nine SVG curves,
matching supported point counts and valid user-duration ranges. This report-only
change requires no new model training or classifier tests. Historical reproduction,
remaining named pairs/Core tasks and exhaustive final scientific audit remain open.

### Source OOF probability calibration for the concatenated Core specifications

Exact source-user fold refitting now supplies probability temperatures for every
force and MANUS concatenated-Core/standalone model used by the anchor controls.
Force source users 1–6 Ramp and MANUS users 3–8 Session 1 each use three held-user
pair folds. Families, CSP/SPD references, thresholds, scalers and classifiers are
refitted on each fold's source-training trials before held-source prediction. No
full-source fitted feature state is reused inside OOF, and no target recording is opened.

The temperature fits raw held-source OOF probabilities and source labels only.
Force Core temperature is 3.7062; MANUS Core is 4.0000 at the existing prespecified
search bound. The bound is not expanded based on target scores. Calibrated scores
on these same source OOF labels are explicitly marked as tuning/descriptive evidence,
not nested outer-held temperature generalization; raw model predictions remain OOF.
These source temperatures are prepared for subsequent frozen target-anchor integration.
The earlier native-probability Core controls are retained as original results and
must not be represented as already using these temperatures.

Runs `feature_bank_force_core_probability_source_20260916` and
`feature_bank_manus_core_probability_source_20260916` retain source trial/fold lists,
fold-fitted family/scaler/classifier states, held-source feature coordinates, raw and
calibrated probabilities, temperatures and exact original source-fit hashes. Thirty
force and eighteen MANUS fold probability arrays replay exactly; all sixteen
source temperature fits and calibrated arrays reproduce. Source guards reject a
final-user source manifest before raw loading, and fold subset tests preserve
aligned context/trial metadata without held-window leakage.

A terminal metadata-serialization error after completed force folds was repaired
from their saved predictions and source manifest; completed classifiers were not rerun.
The producer now casts native source-user IDs to JSON-safe integers.
The suite ran 145 tests with one skip. Integrity checks cover 164 artifacts,
37,971 rows and 741 explicit partitions; canonical verification covers 36,488 rows.
Target integration and held-out confirmation of the probability-calibrated Core
anchors remain unfinished, alongside exact historical and other scientific requirements.

### Frozen source-temperature integration in independent Core anchor controls

`--probability-source` now binds source-only OOF calibration to exact dataset/users,
source-fitting trials, Core model specifications and original source-fit hashes.
OOF source coverage and temperature reproduction are checked before target mixing.
Output manifests retain calibration manifest/OOF-array hashes and exact temperatures;
replay rejects changed probability provenance. Omitting the option preserves original
native-probability behavior. No new classifier fits occur in these target runs.

Four new independent validation/final runs retain the original trials, anchor method,
selection seed and alpha; only the population probability temperature changes.
Scalar temperature leaves standalone decisions unchanged. Matched no-anchor F1 at
all budgets therefore reproduces the original matched no-anchor values, while mixed
anchor decisions can change. The original native-probability controls remain available.

| Final Core mean-user F1 | cal0 | cal1 | cal2 |
|---|---:|---:|---:|
| Force, matched no anchor | 0.4902 | 0.4902 | 0.4902 |
| Force, native probability anchor | 0.4902 | 0.4943 | 0.5005 |
| Force, source-temperature anchor | 0.4902 | 0.5308 | 0.5573 |
| MANUS, matched no anchor | 0.4161 | 0.4040 | 0.3102 |
| MANUS, native probability anchor | 0.4161 | 0.4159 | 0.3333 |
| MANUS, source-temperature anchor | 0.4161 | 0.4074 | 0.3380 |

Force average calibration recovery improves through probability organization without
adding measured signal information, but its minimum force-condition mean-user F1 is
0.4199/0.4282/0.3829. Two-shot therefore lowers the observed minimum below cal0 and
below the native-anchor cal2 minimum 0.4328, despite higher average F1. This is not
universal force robustness recovery. MANUS Core gains remain limited; Temporal
extension cal1 decreases from matched no-anchor 0.4396 to anchor 0.4210, while cal2
rises from 0.3185 to 0.3676. Budgets remove different MANUS evaluation trials, so only
within-budget matched comparisons support calibration effects.

Runs `feature_bank_force_core_temperature_validation_20260916`, `_final_20260916`
and `feature_bank_manus_core_temperature_validation_20260916`, `_final_20260916`
retain all model variants, source-temperature audits and exact calibration splits.
Both force phases reproduce 60 anchor/60 matched control arrays; both MANUS phases
reproduce 108/108 arrays exactly. Tests reject wrong source-user calibration and
verify temperature decision preservation and legacy identity behavior.

The suite ran 147 tests with one skip. Integrity checks cover 176 artifacts,
48,563 rows and 789 explicit partitions; canonical verification covers 47,080 rows.
Calibration burden now includes eleven protocols/44 cost rows, with source-temperature
Core rows using the same recorded sample-count duration evidence as native Core.
All 36 earlier cost rows preserve their recorded values. The full-system vector is
unchanged: supplemental final gains do not replace frozen full-bank algorithms.
Historical data/formulas, remaining named comparisons and exhaustive scientific
requirement audit still remain incomplete; no new GitHub push is performed.


## Matched personal-source G5 versus complete-bout DTW (2026-09-16)

Run `feature_bank_unibo_sequence_temporal_validation_20260916` compares standalone
validated G5 and DTW using identical own-user historical source Days 1-5. Inner
models fit Days 1-4; Day 5 calibrates temperature; final source models fit Days 1-5;
only Day 6 is evaluated. No Day 7/8 data is opened or parameter tuned on Day 6.

DTW uses complete contiguous canonical hand-labelled bouts of at least one second,
32 RMS bins covering every sample, per-timepoint L2 normalization, Euclidean local
cost, 0.1 Sakoe-Chiba band and path-length normalization. Each class medoid uses
at most five deterministic source-only candidate bouts; this is explicitly a
subset medoid. G5 uses the unchanged validated implementation on all complete
contiguous 200-ms windows within the same bouts, then averages window features.
Both predictors have the same personal historical source information. Cal0 means
no new target-day calibration; it does not mean no historical personal data.

On seven users and 1,700 Day-6 bouts, hierarchically weighted macro-F1 is 0.771275
for G5 and 0.373172 for DTW; log loss is 0.401166 versus 1.240488. Error correlation
is 0.116144 and prediction disagreement 0.552544. G5-correct/DTW-wrong mass is
0.488650; G5-wrong/DTW-correct mass is 0.039851. Low error correlation therefore
coexists with a weak standalone predictor and asymmetric potential corrections;
this evidence alone neither selects DTW nor proves its conditional uselessness.
A matched G5-plus-DTW classifier increment has not yet been tested in this run.

Oracle ground-truth bout boundaries and phase-normalized complete trajectories
are offline information. Scores are not directly comparable to earlier short-window
F1, do not measure unsegmented streaming recognition, and do not prove recovery on
the current eight-channel device. The run exports 32 scores and 16 named-pair rows.
Native data were reloaded and all 14 target arrays, 14 inner-source arrays and 14
source temperatures reproduced from saved states without classifier retraining.
Four regression tests enforce bout separation, rejection of short-window DTW,
whole-bout sample coverage and hierarchical user/label weights. The full suite ran
151 tests with one skip; compile checks passed. Current integrity checks cover
178 artifacts, 48,611 rows and 796 explicit partitions; canonical checks cover
47,128 records. Completion remains unproven and new work remains local only.


## DTW conditional value given G5: validation and frozen final (2026-09-16)

Runs `feature_bank_unibo_sequence_incremental_validation_20260916` and
`feature_bank_unibo_sequence_incremental_final_20260916` reuse the full-bout G5 and
DTW families. Only new source-only StandardScaler/balanced logistic classifiers
are fit for concatenated G5 (20 dimensions) plus DTW distances (4 dimensions).
Inner source Days1-4 predict Day5 to fit temperature; final source classifiers
fit Days1-5. Source parameters are frozen before opening final Days7-8. Both
baseline and added model receive identical personal historical source data and
complete oracle-labelled bouts. There is no new target-day calibration.
This is a one-family specialist conditional control, not the multi-family Core.

| Held-out phase | Bouts | G5 F1 | G5+DTW F1 | Delta LogLoss | Delta Brier | Users with LL / F1 gain |
|---|---:|---:|---:|---:|---:|---|
| Day6 validation | 1700 | 0.771275 | 0.797154 | +0.050734 | +0.001213 | 3/7 / 5/7 |
| Days7-8 frozen final | 3391 | 0.753845 | 0.735807 | +0.056390 | +0.000040 | 6/7 / 1/7 |

Positive final Delta LogLoss provides held-day conditional predictive-loss evidence
for DTW given G5 even though standalone DTW is weak. It does not establish better
classification or global robustness. Final minimum-user F1 drops 0.523621 ->
0.516288 (validation minimum drops 0.552994 -> 0.496490). Final OPEN F1 drops
0.663969 -> 0.645234, FIST 0.624028 -> 0.588226 and PINCH 0.728825 -> 0.718358.
The appropriate current role is an auxiliary conditional probability candidate;
it is not selected to replace G5. The previous standalone result is not rewritten
as proof that DTW is useless, and the validation F1 gain is not promoted into a
final or streaming gain.

Source/validation replay reproduces 14 probability arrays and seven source
temperatures, bound to parent artifact hashes. Independent saved-state final
replay reproduces both baseline and added-model predictions for all seven users
(14 arrays, maximum absolute probability error 0). No source family/classifier
is fit or mutated during final evaluation. Oracle bout boundaries, phase-normalized
full trajectories, correlated bouts within trials, native four-muscle data and
known-user historical profiles remain explicit limits. These results do not prove
recovery on the current eight-channel streaming device or improve the earlier
short-window seven-axis robustness vector by substitution.

The per-subject exporter previously accepted only integer identifiers and omitted
native `u01`-style subjects. It now preserves native identifiers and excludes
aggregate/missing subjects. All 7,343 prior summary records were checked unchanged;
241 existing native-ID groups were recovered, plus four new conditional experiment
groups, for 7,588 summaries. The source metric CSVs are unchanged by this exporter.
Four tests cover identifier preservation, aggregate exclusion, rejection of a
final-day parent and source/calibration/evaluation trial overlap. The full suite
ran 155 tests (one skip); current compile checks and replay pass. Integrity covers
184 source artifacts, 48,747 rows and 810 explicit partitions; canonical record
verification covers 47,264 rows. All new work remains local; full completion is
still unproven against the two complete specifications.


## Matched wearing-domain Session calibration (2026-09-16)

Runs `feature_bank_wearing_session_validation_20260916` and
`feature_bank_wearing_session_final_20260916` reuse frozen subject-specific before-
wearing F0/ring classifiers and their repetition-held-out source temperatures.
Validation users are 15/16/17 and final users 18/19/20, with four native after-wearing
domains each. Each source class has five repetitions; each after-wearing
class/domain has only TWO repetitions. Per-domain 0/1-shot is supported; 2/5-shot
is explicitly missing because no held-out trial remains. Other wearing domains
are never pooled into calibration. The five native classes are close/open/rest/
flexion/extension; there is no pinch class. Native recordings are eight-channel
Myo at 200 Hz; the current device is not evaluated.

Long-term prototypes/scales remain immutable. Source class count determines
beta=5/(5+shots); local prototypes are calibration means in the same source-MAD
coordinates. Session signatures contain five residual norms, five cosine
agreements and ten class-pair geometry changes per family. A fixed previously
prespecified cosine-agreement rule modifies uniform provider weights. Source-only
quality-observability diagnostics record calibration/source differences with
unknown ADC/pre-highpass availability masked; they do not gate this classifier
comparison or supply hardware fault labels. Signal rest-center/channel-scale
session updates are not tested by this prototype/context control.

Final mean individual-user/domain scores on the MATCHED one-shot remaining
trials (12 cells, five evaluation trials each) are:

| Method | Macro-F1 | LogLoss |
|---|---:|---:|
| Source provider uniform | 0.519444 | 0.990131 |
| Source provider Session Signature weighting | 0.519444 | 0.989171 |
| Long-term prototype uniform | 0.586111 | 1.423913 |
| Current-session local prototype uniform | 0.701111 | 1.272471 |
| Source-budget blended prototype uniform | 0.683333 | 1.391350 |
| Fixed source/local probability mix | 0.519444 | 1.014047 |

Method IDs contain `population`, but the reused source classifiers are personal
before-wearing models, not generic cross-user classifiers. Population denotes
uniform fixed provider weighting in these controls. Only the source-classifier
probabilities have empirical OOF temperature calibration; prototype distance-
softmax controls use fixed source similarity scale and are not claimed empirically
probability-calibrated. Current local prototypes improve F1 on these matched
trials while worsening LL relative to source classifiers. Session Signature has
no final F1 benefit; the fixed source/local mix fails to recover a final F1 gain.
Validation local-prototype F1 is 0.852778 vs matched source 0.447222, while the
validation cosine rule improves 0.447222 ->0.522222. The validation gain must not
be promoted into a final/deployment gain. Bouts/windows within a trial are not
independent observations. Calibration excludes entire trials; cal0/1 use different
evaluation sets, so causal recovery comparisons are within each budget only.

Each wearing condition requires five one-shot trials, mean 15.11 seconds of native
recorded signal (range 15.075-15.13), plus unmeasured human setup/transitions. The
cost report now has 12 protocols/48 rows and 12 SVG curves; all 44 earlier costs
were checked exactly unchanged. Raw sample/rate evidence covers 160 selected
recordings. The wearing curve shows the fixed source/local mixing CONTROL,
not a final-selected local-prototype deployment replacement. Native before/after
wearing conditions are not fabricated calendar-session/day labels, and the
reference ring is not claimed to reproduce unavailable historical RLCS.

Saved-profile replay reproduces 144 probability arrays per phase (288 total,
maximum absolute error 0) and verifies parent source hashes and immutable long-
term profiles. Three tests cover whole-trial class coverage, rejection of budgets
that empty evaluation and ambiguous duplicate source roots. The full suite ran
158 tests (one skip); compile checks passed. Integrity covers 188 source artifacts,
49,303 rows and 858 explicit partitions; canonical verification covers 47,820 rows
and descriptive per-subject analysis has 7,684 summaries. These integrity checks
are not an exhaustive scientific completion audit.

New experiment outputs live under workspace `work/benchmark_runs` rather than
writing outside the permitted workspace. Consolidation accepts `--local-root`
and records explicit source-root provenance for these runs; the verifier reads
that bound root. Existing external benchmark source rows remain unchanged.
No new GitHub push is performed; the two specifications remain incomplete.


## Integrated personal/session signal calibration (2026-09-16)

`SessionCalibrationPipeline` connects source-fitted rest/active normalization,
source quality observability, source family/scaler/classifier models and immutable
long-term prototypes to calibration-only session updates. Current rest center is
the calibration Rest median; per-channel scale is active-calibration absolute
Q95 about that center. Calibration updates quality diagnostics, low-dimensional
class residual/cosine/geometry signatures and local/blended prototypes. It keeps
the long-term normalizer, models, quality reference and prototypes unchanged.
Native Rest label is explicitly 2 in this wearing dataset. Eight channels alone
do not establish ring geometry: callers must supply the verified topology contract.
Prediction rejects source-fit trials, calibration trials, different users and
changed sample-rate/window contracts. No target evaluation transform is fit.
Quality descriptors are recorded without unvalidated quality gating.

Runs `feature_bank_wearing_session_normalization_validation_20260916` and
`feature_bank_wearing_session_normalization_final_20260916` retain the original
unnormalized source providers. Normalized source classifiers fit only before-
wearing data. Each of five source OOF folds uses three before-wearing repetitions
for every source normalizer/family/scaler/classifier/profile, the next repetition
for simulated current-session calibration, and the held repetition for evaluation.
All six branch/provider probability paths, including prototype controls, receive
empirical temperatures fit from their complete source OOF predictions BEFORE
fusion. Full personal source models fit all five before-wearing repetitions.
This temperature protocol differs from the old raw provider's four-repetition
source OOF; raw-vs-normalized comparisons are complete branch comparisons and
must not be interpreted as a pure signal-normalization mechanism isolation.

Final matched one-shot mean individual-user/wearing-domain scores are:

| Branch | F1 | LogLoss |
|---|---:|---:|
| Retained unnormalized source | 0.519444 | 0.990131 |
| Fixed historical source normalization | 0.380000 | 1.398134 |
| Current-session rest/scale update | 0.663889 | 0.986590 |
| Current-session update plus cosine context | 0.641667 | 0.983365 |
| Current-session local prototype | 0.747222 | 0.831565 |
| Current-session source-budget blended prototype | 0.563889 | 1.004060 |

On the same trial split, source/current-normalized model F1 is 0.430556/0.542778
in validation, and local-prototype validation F1 is 0.895556. Static per-channel
normalization can hurt; current rest/scale calibration recovers signal organization
in this measured wearing protocol. The context weighting loses final F1 relative
to the updated source classifier, and blending retains too much historical
prototype information in this control. These are not universal calibration laws.

Family-specific final one-shot F0 model F1 changes 0.427778 -> 0.611111 with current
normalization, while ring changes 0.263333 -> 0.331667. Uniform fusion is 0.663889,
so weak standalone ring performance does not establish uselessness. F0 local
prototype F1 is 0.772222 vs ring 0.428889, while local fusion is 0.747222: the same
uniform bank can hurt relative to a specialist. Per-family tables and paired
calibration-conditioned loss/Brier/F1 controls are exported. Those controls
compare transformed input/prototype/fusion branches, not a literal additional-
feature concatenated classifier or estimated conditional mutual information.

Only 0/1-shot is supported: two native repetitions/class/after-wearing domain leave
one evaluation trial/class at one-shot. Calibration never pools wearing conditions
or fits evaluation samples. Different budgets have different remaining trials;
all recovery comparisons above are within the one-shot split. There are three
final users and four conditions, with only five evaluation trials per user/condition
(60 final one-shot trials). Metrics aggregate whole-trial features averaged over
eight representative 200-ms windows and do not measure streaming 200-ms accuracy
or hardware latency. Native classes include no pinch; real current-device
250 Hz data, class mapping and electrode geometry are not validated by this result.

Native reload/saved-state replay reproduces 204 arrays per phase (408 total),
including every source-OOF branch/provider and normalized target fusion path;
source temperatures recompute and source/session state remains immutable. The
raw branch is bound to its unchanged parent model hashes and parent replay.
Source/class/window-trial guards are covered by three tests; the full suite ran
161 tests (one skip), and compile checks passed. The cost report now contains
13 protocols/52 rows and 13 standalone SVG curves; all 48 previous costs were
checked unchanged, including the shared 15.11-sec one-shot recording duration.
The SVG canvas grows to 635 px so the legend and evidence notes remain separate.
Current integrity covers 194 source artifacts, 50,915 rows and 1,146 explicit
partitions; canonical verification covers 49,432 rows and per-subject analysis
has 8,004 summaries. These are integrity/coverage observations, not completion.

Historical DS2 and exact old X1-H/RLCS/CES/Frequency remain unavailable. The
remaining exact family/formula and complete document-clause audits are still
required. New output stays under `work/benchmark_runs`; no new push is performed.


## EPN legacy Anchor temperature correction (2026-09-16)

The old `feature_bank_epn612_trial_calibration_validation_20260915` helper scaled
anchor logits by the median of evaluation-batch distances. Its twelve nonzero-shot
result rows are retained for history but excluded from leakage-compliant scientific
acceptance. Zero-shot population predictions do not use the faulty helper.

The corrected helper uses calibration-only prototype-distance scale. A new frozen
validation run uses exactly the same calibration trial CSV (byte-identical), seed,
source users1–15, target users16–18 and F0/reference-envelope-ring composition.
Pooled macro-F1 at budgets0/1/2/5 is now0.416870/0.406874/0.351224/0.336868;
log loss1.483118/1.499481/1.546646/1.650541. Previous nonzero-budget F1 values
0.414149/0.399633/0.370582 must not be cited as compliant Anchor performance.
The correction establishes evaluation-row independence, not recovery: negative
transfer persists and becomes stronger. This branch is not the later EPN selected
source-OOF probability pipeline, so its results cannot replace that separate branch.
Native labels remain six EPN classes; this is not own-device four-class validation.
See `results/epn_anchor_temperature_correction.json` for boundaries and hashes.

The frozen corrected protocol was then evaluated once on final users19–21. Pooled
macro-F1 at0/1/2/5 shots is0.438608/0.440676/0.422205/0.446555 and log loss is
1.556775/1.530031/1.513385/1.555759. The final-set pattern differs from validation:
1 and5 shots slightly exceed zero-shot F1, while2 shots declines; no monotonic or
universal calibration benefit follows. These final results were not used to change
the calibration rule, family composition, shrinkage, trials, or model settings.

The saved source family objects, scaler and classifier were also replayed directly
against native EPN target recordings. All 12 validation and 12 final user-budget
probability arrays match the saved predictions exactly (maximum absolute error 0).
This checks target preprocessing, trial aggregation, frozen source inference and
calibration-only Anchor application without refitting a model. The 5.48-GB ZIP
was not freshly hashed in full; relevant members were read with ZIP CRC checks.
`native_replay_audit.json` for each corrected run and the regenerated correction
audit record the narrower verification scope.

## F7 SPD tangent Anchor candidate (2026-09-22)

An explicit `SpdTangentPersonalAnchor` now snapshots a source-fitted F2c SPD
reference, fits gesture prototypes using only labeled personal calibration
windows, and transforms evaluation windows without refitting. Its Euclidean
distance on the sqrt(2)-weighted upper triangle equals Frobenius distance in
the fixed log-tangent space; an independent two-channel matrix oracle checks
the formula. Output dimensions are 2H+2, as for other personal anchors.
This is a candidate tangent approximation, not the exact affine-invariant
geodesic. The initial implementation had no native held-out result;
the subsequent trial-level study below supplies limited EPN evidence.
It does not replace missing historical X1-H/RLCS/CES/Frequency implementations.

The native trial extension preserves the source reference fitted only on EPN
users1–15. It averages the four available windows of each native trial before
fitting class prototypes, so each calibration trial has one vote. It reuses
the corrected Anchor study's exact selected calibration IDs (six native
classes, one/two/five trials per class) for validation users16–18 and final
users19–21; all remaining trials of the corresponding user are evaluated.
The trial-aware API stores calibration trial IDs and rejects overlap at
evaluation, including when a caller tries to use a window-only fit.
Selection and source-code hashes, per-user scores, pooled scores and every
trial prediction are saved under
`work/benchmark_runs/feature_bank_epn_spd_anchor_validation_20260922` and
`work/benchmark_runs/feature_bank_epn_spd_anchor_final_20260922`.
The compact Git-delivered score and hash record is
`feature_bank/results/epn_spd_anchor_trial_study.json`.
Zero-shot personal prototypes are undefined, so no zero-shot
SPD Anchor result is asserted.

Pooled validation macro-F1 for one/two/five shots is 0.4464/0.4550/0.4853;
accuracy is 0.4583/0.4662/0.4972. Final macro-F1 falls to
0.2907/0.3573/0.4433 and accuracy to 0.2940/0.3502/0.4472. Final log loss
is 1.7284/1.6722/1.6455. This substantial validation-to-final drop and
uneven subject results do not justify an accuracy or calibration gain claim.
The corrected F0+reference-ring Anchor scores above share the calibration
trial IDs and evaluation budgets but use a different source representation
and population classifier; any descriptive numerical difference is not an
incremental F7 contribution to a fixed multi-family Core. The candidate
also has no live-device result or historical RLCS identity.

### Fixed Core plus SPD Anchor on matched EPN trials (2026-09-23)

`epn_spd_anchor_core_increment.py` joins the saved source-frozen EPN
F0+Ring+CSP+real-IMU shortlist probabilities to the saved SPD personal
prototype probabilities by exact user and native trial ID. The input hashes,
six-class order, labels and complete calibration exclusion are checked before
scoring. Every arm sees the same 432/414/360 native trials per phase at
1/2/5 shots;
no classifier or SPD reference is refitted. The additional arm is a fixed
equal-probability mixture, chosen without using these scores. Compact per-user
and pooled results plus input hashes are in
`results/epn_spd_anchor_core_increment_{validation,final}.json`.
The same 48 score rows, 12 paired increments and 48 calibration-curve rows
per phase are exported through the two dated local run manifests into the
required consolidated CSVs. The canonical delivery checks all required fields
for these new rows and preserves their source-record hashes.

| Phase | Shots/class | Core F1 | Core+SPD F1 | Core LL | Core+SPD LL | Core Brier | Core+SPD Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 1 | .4503 | .4643 | 1.6344 | 1.4359 | .1189 | .1119 |
| Validation | 2 | .4527 | .4785 | 1.6240 | 1.4010 | .1176 | .1102 |
| Validation | 5 | .4568 | .4790 | 1.5795 | 1.3766 | .1155 | .1089 |
| Final | 1 | .4427 | .4450 | 1.5371 | 1.4400 | .1157 | .1121 |
| Final | 2 | .4460 | .4595 | 1.5458 | 1.4143 | .1159 | .1107 |
| Final | 5 | .4476 | .4613 | 1.5399 | 1.4179 | .1165 | .1111 |

This gives a narrow conditional value for SPD personal information when added
by late fusion to this fixed multi-family Core. The anchor alone is weaker at
one/two final shots (.2907/.3573 F1), and at five validation shots the anchor
alone (.4853) exceeds the mixture (.4790). Mixture gains should not be
interpreted as proof that its standalone classifier is strong or that every
subject gains. User21 loses .0356 F1 at one final shot despite a .1206
log-loss improvement. These final users were already
examined in earlier project analyses, so this retrospective replay is
exploratory, not an untouched confirmatory test. It does not establish the
value of concatenating F7 into a newly fitted Core, a tuned fusion weight,
historical RLCS, other datasets or current-device recognition.

## Frozen Quality × reference-family interaction replay (2026-09-22)

The saved LibEMG force quality-stress runs contain separate source-fitted,
source-OOF-calibrated provider probabilities for F0, reference F1, reference
Ring and F9 Quality. `quality_family_interactions.py` replays four fixed
equal-provider arms on identical native trial IDs: F0; F0+family; F0+Quality;
F0+family+Quality. No classifier, family, fusion weight or target calibration
is fitted. Validation uses users7/8, final uses users9/10; each has clean and
seven prespecified synthetic perturbations. The script verifies saved replay
audits, phase users and probability alignment; raw arm/interaction/
complementarity tables remain in `work/benchmark_runs`, and
`results/quality_family_interactions.json` carries source/output hashes.

On clean validation trials, reference F1 versus Quality has 28.2% binary
correctness disagreement and reference Ring versus Quality has 33.2%; both
directions of asymmetric correctness occur. Clean negative-log-loss interaction
S is +0.0361/+0.0709 on validation and +0.0466/+0.0572 on final for
F1/Ring respectively. Across the seven synthetic scenarios, mean final S is
+0.1031/+0.1145. These positive second differences do not imply a useful
bank: final clean macro-F1 of the F0 baseline is 0.5118, while the
F0+F1+Quality and F0+Ring+Quality arms score 0.4361 and 0.4724. Across the
synthetic scenarios, their mean pooled F1 is 0.3382/0.3298 versus baseline
0.4639. Neither full arm merits promotion. This is probability-composition
behavior under synthetic quality changes, not physiological synergy or
measured device noise; neither reference F1 nor Ring establishes historical
X1-H/RLCS equivalence.

## UniBo validated G5 × reference Temporal interaction (2026-09-22)

The named Stage 4 G5×TemporalShape question now has a four-arm *candidate*
comparison on identical native UniBo windows. B is the existing validated G0;
the other arms add validated G5, current reference TemporalForm, or both.
Source Days1–4 fit family/classifier states for probability calibration, held-out
source Day5 fits each arm's temperature, and source Days1–5 fit the frozen
evaluation states. Day6 validation and Days7–8 final evaluations never fit a
family, scaler, classifier or temperature. Native four-channel trials are
disjoint by day, and hierarchical segment weights enter source fitting and
evaluation. This is not a 0/1/2/5-shot personal-calibration curve.

| Pooled macro-F1 | G0 | G0+G5 | G0+Temporal | G0+G5+Temporal |
|---|---:|---:|---:|---:|
| Day6 validation | 0.6323 | 0.6374 | 0.6504 | 0.6502 |
| Days7–8 final | 0.6634 | 0.6662 | 0.6772 | 0.6773 |

The negative-log-loss interaction S is -0.00448 on validation and -0.00119
on final; macro-F1 interaction S is -0.00540/-0.00275. Positive user-level
log-loss interactions occur in only 2/7 validation and 4/7 final subjects.
Adding G5 to G0+Temporal changes final pooled F1 by less than 0.0001 and
slightly worsens log loss. These data do not establish that the two temporal
families are synergistic in a common bank. All four arms are source-day
calibrated; the validated G5 formula is reused, while the current TemporalForm
remains a reference candidate rather than proven historical TemporalShape.

The first three Day6 arms, after applying the recorded source temperatures,
exactly replay the prior validated-reuse probabilities on all 24,338 matched
windows (maximum absolute difference zero); the fourth arm is the new matched
combination. `results/unibo_g5_temporal_interaction.json` records the frozen
source/output hashes, Day6 replay and independent final scores. Full
probabilities, split trial IDs, per-subject and per-posture cells are under
`work/benchmark_runs/feature_bank_unibo_g5_temporal_*_20260922`.
Windows from the same trial are correlated; this is an offline representation
comparison, not complete-bout DTW, streaming recognition or own-device evidence.

## Canonical complementarity delivery repair (2026-09-22)

The 60 recorded wearing Core-versus-increment native-trial rows contained the
two asymmetric correctness counts and their exact trial denominator, but the
required delivery columns were empty. The canonical builder now divides each
count by `evaluation_trials` only when `evaluation_unit` is
`whole_native_trial_mean`, records the derivation in row metadata, and rejects
invalid count/denominator combinations. It also maps the recorded
`disagreement_fraction` to `disagreement_rate`; that disagreement need not equal
the sum of asymmetric correctness rates because both predictions may be wrong.
All 60 rows now have the three required rates without retraining or inventing
observations. The regenerated provenance audit verifies 49,988 source records.
The delivery remains `schema_complete_evidence_partial`: some source runs still
lack subject, session/domain, or error-correlation evidence. This repair does
not establish a physiological complementarity claim or complete Section 15.

## Corrected Quality complementarity and robust-family replay (2026-09-22)

The earlier Quality-pair table labelled correctness disagreement as
`disagreement_rate`. Those are different quantities: two models can predict
different wrong classes on the same trial. The replay now reports actual class
prediction disagreement, keeps correctness disagreement separately, and adds
Pearson correlation of the two binary error indicators (undefined when either
indicator is constant). The F1/Quality and Ring/Quality clean validation
prediction-disagreement rates are 0.4354 and 0.6395, versus the previously
reported correctness-disagreement rates 0.2823 and 0.3316. The earlier
four-arm interaction scores replay exactly; only the mislabelled diagnostic
changed.

The same frozen source providers also permit Stage 4 reference comparisons
with CSP, Spectral and Temporal against Quality. All three have both directions
of asymmetric correctness on clean validation trials, so each was carried to
the untouched final users using the same four equal-provider arms. On final
clean trials their negative-log-loss interactions are positive, but across the
seven synthetic perturbations their full-arm mean macro-F1 values are about
0.35, 0.36 and 0.34, respectively, below the common F0 baseline of 0.46.
Thus this probability-composition replay does not support promotion of these
banks. The 240-row pair table and 960-row arm table remain under
`work/benchmark_runs`; `results/quality_family_interactions.json` records
source hashes and pooled scores. These are current reference families under
synthetic perturbations, not historically validated robust families or real
device-noise evidence.

The same semantic check found 402 older delivery rows where the source field
`disagreement` also means binary correctness disagreement. The canonical
builder no longer maps that field to prediction disagreement. For 102 EPN
selection and force screening rows, preserved held-out probability arrays
recover the actual class-prediction disagreement. The remaining 300 EPN and
MANUS diagnostic rows were replayed from frozen fitted states, original trial
splits and freshly SHA-256-verified raw archives, without fitting. Recovery
verifies every original correctness-disagreement and asymmetric-correctness
rate to 1e-12, binds each recovered rate to its original row hash and input
hashes, and changes no source row. All 402 legacy rows now have true
prediction disagreement; all 49,988 canonical records pass provenance checks.
The broader schema remains evidence-partial because other fields still lack
source support.

## Frozen EPN shortlist on independent final users (2026-09-22)

The four-family F0+Ring+CSP+real-IMU shortlist was selected on EPN development
users 16–18. `epn_shortlist_final_replay.py` loads its existing family and
classifier states fitted on source users 1–15, first reproduces all 12 saved
development-model probability arrays exactly (maximum error zero), then applies
those unchanged states to final users 19–21. Source, development and final
native trial identities are disjoint. There is no new feature/model fitting,
target calibration, weight selection or final-score-driven arm choice. The
450 final trials and 12 model probabilities, full/removal scores and split IDs
are under `work/benchmark_runs/feature_bank_epn_shortlist_final_replay_20260922`;
`results/epn_shortlist_final_replay.json` records input/output SHA-256 hashes.
The 48 final subject/model score rows and 20 full/removal rows are also copied
to `results/epn_shortlist_final_results.csv` and
`results/epn_shortlist_final_ablation.csv`; their bytes match the replay hashes.

| Final pooled macro-F1 / log loss | F0 | Full | Minus F0 | Minus Ring | Minus CSP | Minus IMU |
|---|---:|---:|---:|---:|---:|---:|
| Macro-F1 | .4407 | .4458 | .3783 | .4721 | .4736 | .4237 |
| Log loss | 1.4556 | 1.5391 | 1.6342 | 1.4152 | 1.4796 | 1.6126 |

The development full-bank gain does not generalize clearly: final pooled F1
barely exceeds F0 while log loss worsens. Removing Ring or CSP improves final
F1; removing F0 or real IMU harms it. The prespecified F0+IMU single-addition
arm reaches .4978 F1 and 1.3998 log loss, but these final outcomes cannot be
used to retroactively select a new bank and call its performance independent.
The minimum of the three final user F1 scores is .3965 for the full bank versus
.4263 for F0, so even this narrow user-robustness floor is not raised.
Only three final users are available. The final cohort is held out from this
shortlist fit/selection, although it has been used by other project studies;
this is not a pristine project-wide blind test. Reference Ring is not proven
historical RLCS, and no cross-failure or own-device generalization follows.

The frozen final replay is also included in the canonical delivery: 48
family/model score rows and 20 full-bank/removal rows. Its source audit binds
the saved source states, exact split IDs and replay outputs; the delivery
exporter verifies those hashes and keeps the prior ablation columns stable.
The additional relative log-loss value remains in the frozen source result.
Canonical verification checks all 49,988 copied records; this incorporation
does not change the original model selection or final-user scores.

## Frozen EPN Personal Anchor × reference Spatial Coordination (2026-09-23)

The remaining named Personal Anchor × Spatial Coordination question now has a
matched four-arm reference comparison on native EPN trials. The arms are source
population F0; source population F0+CSP; personally anchored F0; and personally
anchored F0+CSP. Both population providers reuse source-user OOF temperatures.
The personalized providers reuse the saved calibration-only `PersonalAnchor`
objects, their saved distance temperatures, the fixed `shots/(shots+2)` blend,
and exactly the same 1/2/5-shot calibration and evaluation trial IDs as the
existing EPN probability study. No state or trial selection is refit. Validation
uses users16–18; the unchanged code then evaluates final users19–21.

| Pooled macro-F1 | F0 | F0+CSP | F0+Anchor | F0+CSP+Anchor |
|---|---:|---:|---:|---:|
| Validation 1-shot | .3946 | .3691 | .3676 | .3540 |
| Validation 2-shot | .3981 | .3734 | .3638 | .3857 |
| Validation 5-shot | .4104 | .3829 | .3730 | .3642 |
| Final 1-shot | .4455 | .4454 | .3516 | .3905 |
| Final 2-shot | .4462 | .4460 | .4213 | .4231 |
| Final 5-shot | .4518 | .4465 | .3841 | .4087 |

The negative-log-loss second difference is positive for every pooled cell
(.0651/.0897/.1117 on validation and .0371/.0385/.0732 on final), as is the
macro-F1 second difference. This only says that the joint degradation is less
than the sum of the two individual degradations. Every full arm is below its
matched F0 macro-F1, and every full arm has worse log loss, so the interaction
does not support promotion. Different budgets remove different calibration
trials; comparisons are only within a budget. The saved probability arrays,
split IDs and per-user/pooled scores are under
`work/benchmark_runs/feature_bank_epn_anchor_spatial_{validation,final}_20260923`;
the compact hash record is `results/epn_anchor_spatial_interaction.json`.
Current CSP is a document/reference candidate and is not proven identical to a
missing historical Spatial Coordination implementation. This is offline EPN
probability composition, not causal physiology, feature concatenation, streaming
recognition or current-device performance.

## Frozen reference Ring × Session Signature wearing interaction (2026-09-23)

The named RLCS × Session Signature question now has a matched reference-family
four-arm study. To keep Session Signature identifiable after removing Ring, the
fixed base contains F0 plus reference CSP. The arms add reference Ring, the
prespecified calibration-only Session Signature provider reweighting, or both.
All family/scaler/classifier states and source OOF temperatures come from the
existing before-wearing packages. The experiment reuses the exact saved
one-shot calibration and evaluation trials for each user and wearing domain.
Validation users15–17 freeze the code before final users18–20 are evaluated.

| Pooled one-shot result | F0+CSP | +Ring | +Session | +Ring+Session |
|---|---:|---:|---:|---:|
| Validation macro-F1 | .5856 | .6309 | .5856 | .6309 |
| Validation log loss | 2.8503 | .7995 | 2.8251 | .7767 |
| Final macro-F1 | .6682 | .6967 | .6985 | .6967 |
| Final log loss | 1.3856 | .9101 | 1.3762 | .9023 |

Ring supplies the large probability-loss recovery in both cohorts. Session
Signature slightly improves log loss, but its pooled F1 gain is absent on
validation and does not survive after Ring is present on final. The pooled
negative-log-loss interaction is slightly negative (-.00238 validation,
-.00155 final); final macro-F1 interaction is -.03031. Thus the two mechanisms
do not show positive joint value for this fixed bank. Full prediction arrays,
per-user/domain cells and split IDs are under
`work/benchmark_runs/feature_bank_wearing_ring_session_interaction_{validation,final}_20260923_v2`;
`results/wearing_ring_session_interaction.json` binds their hashes.
Reference Ring is not the unavailable historical RLCS, and the four native
wearing conditions are not calendar sessions. This result cannot establish
longitudinal or current-device performance.

## Historical implementation recovery audit (2026-09-23)

The GitHub remote exposes four branches: `main`, `codex/hla-emg-testbed`,
`codex/unibo-physiology-experiments`, and the current
`codex/unibo-full-physiology-ablation`. Their remote tip commits were fetched
without switching or modifying the working branch. Exact named searches for
RLCS, X1-H, TemporalShape, DS2 and Spatial Coordination over source, config,
CSV and report extensions found no match at any remote tip and no match in the
diff history reachable from those refs. The HLA branch contributes GRABMyo/HLA
code; the older UniBo branch contributes the baseline experiment, not the
missing historical families. `results/historical_source_recovery_audit.json`
records the remote URL, exact ref commits, path filters, terms and script hash.
Therefore current reference families cannot be relabelled as recovered
historical implementations. This audit cannot exclude code that existed only
outside this repository, in inaccessible private history, or under unrelated
names without preserved documentation.
# Real 8-channel single-participant follow-up

The user-provided Song HDF5 v3 sessions have now been audited and evaluated in an exploratory split-locked offline study: [study report](../benchmarks/song_real8/REPORT.md) and [machine-readable results](../benchmarks/song_real8/RESULTS.json). Four-state S04 trial accuracy is 90.3% (macro-F1 90.0%); the 28-state endpoint is materially weaker at 64.6% (macro-F1 53.1%). These one-day cue-labelled results do not establish live UniBo recognition or formal collection acceptance. A separate [four-session cohort gate](../benchmarks/song_real8/COHORT_READINESS_AUDIT.json) confirms that S03 did not occur on a later day and S02–S04 lack re-donning attestations; S04's individual pass cannot certify the cohort.

The distinct pre-formal Song calibration blocks now provide a direct paired
personal/session follow-up: one neutral, pinch, fist and open block per class
improves S04 four-state trial accuracy from 90.3% to 93.8% and macro-F1 from
90.0% to 93.6%, correcting five trials with no new errors. Two blocks per
class give the same score. Parameters were chosen using S03 only; S04 formal
trials were scored after applying the earlier calibration blocks. The exact
paired discordance p-value is 0.0625 and this single-person/single-day study
does not establish product-level recovery or live accuracy. Details and
descriptive bootstrap uncertainty are in
[CALIBRATION_RESULTS.json](../benchmarks/song_real8/CALIBRATION_RESULTS.json).

Under a separate causal filtering replay, S04 zero-shot accuracy/macro-F1 are
91.0%/90.7%. One calibration block per class gives 91.7%/91.3% (two corrected
trials, one new error); two blocks give 91.0%/90.6%. The causal one-shot
macro-F1 difference has a descriptive paired 95% bootstrap interval spanning
zero. The larger zero-phase calibration gain therefore does not establish a
real-time-compatible recovery effect. See
[causal calibration results](../benchmarks/song_real8/CAUSAL_CALIBRATION_RESULTS.json).

The causal zero-shot F0 model is now exportable to the collection app as a
hash-checked local JSON bundle. An independent runtime replay compared 416
S03 validation windows against the source sklearn classifier (maximum
probability error `1.41e-7`) and 999 chunked S04 windows against the offline
causal filter (zero observed probability difference). The app's offscreen
model-load/probability test passes with a schema-compatible synthetic bundle.
This closes the model-format and
preprocessing-transfer gap, while real USB streaming accuracy and end-to-end
latency remain unmeasured. The learned model stays local under Git-ignored
`models/`; see [the replay audit](../benchmarks/song_real8/LIVE_EXPORT_REPLAY_AUDIT.json).

A full S04 replay with sliding 200 ms windows predicts neutral in 66.0% of
11,916 frames, which include rest and uncued time. In stable cue intervals,
frame accuracy is 80.1%; trial-mean accuracy is 89.6%, with Open Hand 36/36
and Index Pinch 25/36. The interface now keeps a debounced Song label visible
through a sustained action. These are cue-timeline findings, not verified
physiological onsets or a physical-device accuracy test; see
[continuous replay](../benchmarks/song_real8/CONTINUOUS_REPLAY_AUDIT.json).

A source-frozen F2c SPD tangent increment has now been checked on the same
Song 8-channel split. Causal S04 stable-trial macro-F1 rises from F0 90.68%
to F0+SPD 95.08%, with six corrected trials and no new errors; source models
and the SPD reference use S01/S02 only. A 1/2-block F7 personal SPD anchor
mix selected on S03 gives lower S04 F1 than the zero-shot F0+SPD model.
The F0+SPD model is available as a separate local experimental bundle and its
S03-window/chunked-causal transfer agrees with the source classifier. Full
S04 cue-timeline replay improves stable decoder-state accuracy from 73.1%
to 79.9%, while raw neutral predictions in pre-prompt rest fall from 87.1%
to 78.9%. It is not promoted as an unqualified default, and none of these
same-person/day cues prove hardware live or cross-person/day performance.
See [the SPD study](../benchmarks/song_real8/SPD_INCREMENT_RESULTS.json),
[export replay](../benchmarks/song_real8/SPD_LIVE_EXPORT_REPLAY_AUDIT.json)
and [continuous replay](../benchmarks/song_real8/SPD_CONTINUOUS_REPLAY_AUDIT.json).
The [paired cue-event readback](../benchmarks/song_real8/CUE_EVENT_REPORT.md)
finds 100/108 active events detected at least once by F0+SPD versus 92/108
by F0; nine Pinch misses are recovered and one Fist becomes missed. Late
pre-prompt rest intervals containing an active decoded state rise from
16/144 to 28/144. These are recorded-cue diagnostics, not measured USB or
physiological-onset behavior.

A separate validation-only neutral-logit offset study confirms that the SPD
model's active/rest tradeoff is visible after online decoding. S03 selects
an exploratory +1.0 offset; S04 late-rest active display then falls from
17.2% to 12.0%, still above F0's 9.3%, while stable-cue decoded macro-F1 is
80.9% versus F0's 72.3%. The offset is not deployed, because S04 had already
been inspected and one person's cue-labelled rest cannot establish a safe
live operating point. See [the bias audit](../benchmarks/song_real8/NEUTRAL_BIAS_STUDY.json).

The 28-state Song hand×arm endpoint gives a distinct condition-level result:
adding source-fitted SPD to the fixed F0+real-IMU model improves S04 overall
accuracy from 68.1% to 70.8% and log loss, but lowers joint macro-F1 from
54.9% to 51.7%. Half the S04 trials are still-arm; the backward and down
conditions lose joint accuracy, so this is not a robust multi-condition Core
increment. The run replays and matches the saved causal F0+IMU baseline only
under its original Python/NumPy/SciPy environment and records those versions.
See [the 28-state result](../benchmarks/song_real8/SPD_28_STATE_RESULTS.json).

The project collection page has additionally loaded the actual local Song
F0+SPD bundle in a Qt offscreen replay and matched direct inference on saved
S04 raw samples. Both desktop entries `EMG 数据采集` and
`EMG-IMU 项目版（Song 8通道）` now point to this repository's collection app;
the former external launcher remains under a name marked as old. This is
software-path evidence only; USB/device behavior
and human-facing recognition remain unmeasured. See
[the UI replay audit](../benchmarks/song_real8/SPD_UI_OFFSCREEN_AUDIT.json).

A further full-stream cue-response audit compares when the unchanged live
decoder first displays the prompted class. On S04, F0+SPD has three additional
within-cue hits (142/144 versus 139/144), including three additional Index
Pinch cues; both models hit all 36 Open Hand cues. Among the 139 paired hits,
SPD is earlier on 51, equal on 74 and later on 14, with 0 ms median paired
latency difference. S03 has 139/141 hits for both. This is descriptive
cue-relative output timing, not measured gesture-onset or device/UI latency.
See the [S03](../benchmarks/song_real8/CUE_RESPONSE_S03_AUDIT.json) and
[S04](../benchmarks/song_real8/CUE_RESPONSE_S04_AUDIT.json) audits.

The same S04 raw prefix has now also passed a reconstructed device-protocol
replay through the actual parser, acquisition controller, project main window
and Song worker. All 1,000 signed 24-bit 8-channel samples kept their saved
order, and the latest UI probability vector exactly matched direct runtime
inference. This narrows the software-path uncertainty; physical USB acquisition
and new-donning electrode geometry remain unverified. See the
[protocol audit](../benchmarks/song_real8/PROTOCOL_TO_UI_AUDIT.json).

The local Song source-only F0 versus F0+F2c SPD study now supplies a held-out
probability reliability curve and the requested canonical delivery schemas.
The exported bundles exactly replay the previously saved 140 S03 and 144 S04
trial scores before ECE is calculated. S04 10-bin top-label ECE is 0.1851
for F0 and 0.1029 for F0+SPD (S03 0.1600 and 0.0913). Four family rows,
two conditional increments, two paired error-complementarity rows and ten
calibration-curve rows were appended to the five delivery tables;
all prior source records remain in order and unchanged, and the canonical
verifier now checks 50,006 rows. The existing prediction-rate recovery was
rebound only after verifying an append-only source table with 4,685 unchanged
prior pair rows; the two new Song rows carry directly computed disagreement
rates. This remains a same-person/day cued-stable result, not live reliability
or a full-bank ablation. See the [calibration audit](../benchmarks/song_real8/PROBABILITY_CALIBRATION_AUDIT.json),
[Song source run](source_runs/feature_bank_song_real8_spd_delivery_20260924/run_manifest.json)
and [delivery provenance](delivery/PROVENANCE_AUDIT.json).

The six additional Song curve rows are a separately named source-F0 plus
personal-SPD-anchor method at 0/1/2 shots per class. S03 selected the 1/2-shot
mixture weights; S04 uses them unchanged and excludes calibration blocks from
formal evaluation. S04 macro-F1 rises from 0.9068 to 0.9225/0.9210, while
LogLoss worsens from 0.4272 to 0.8647/0.5357. Zero-shot source F0+SPD remains
stronger at 0.9508 F1 and 0.2376 LogLoss. Only two pre-formal blocks/class
exist, so 5-shot is N/A; this calibration method is not promoted to the app.

A separate [Song temperature tradeoff audit](../benchmarks/song_real8/TEMPERATURE_STUDY.json)
tested seven post-processing temperatures on the exported source-only F0+SPD
probabilities. S03's best eligible value improved trial LogLoss by only
0.0066, less than the fixed 0.01 adoption margin, so the selected temperature
remains 1.0. S04 continuous decoding and probability scores are unchanged;
no app threshold or bundle was changed. This one-person/day result does not
validate live recognition or cross-session generalization.

The [own-device gain sensitivity audit](../benchmarks/song_real8/GAIN_SENSITIVITY.json)
also subjects the unchanged source F0 and F0+SPD bundles to fixed synthetic
common-channel gain factors on Song causal stable windows. On S04, reducing
all channels to 0.25× changes F0 macro-F1 from 0.9068 to 0.3629 and
active-to-neutral trial errors from 4.6% to 57.4%; F0+SPD retains 0.9157
macro-F1 with 5.6% active-to-neutral errors. S04 recorded per-channel RMS
remained near the source reference (0.98–1.28×), so this is a failure-mode
probe rather than an explanation of a measured live amplitude drop. It cannot
substitute for another wearing or the required multiuser/multiday evidence.

The [Song raw-ADC F9 observability audit](../benchmarks/song_real8/QUALITY_OBSERVABILITY.json)
fits source quality references on S01/S02 and checks the recorded S03/S04
stable windows. The existing composite F9 `min_quality < 0.5` would reject
35.3%/68.0% of windows, including 62.0%/90.7% of Open Hand windows. All
flags arise from source-relative amplitude `|z| > 3`, not observed zero,
flatline or ADC-clipping faults. A separate flatline/clipping-only candidate
flags none of those recorded windows and all windows with one synthetically
frozen channel, but remains undeployed pending real fault and new-wearing
validation. This prevents treating F9 activation variation as measured
hardware quality or claiming the current quality mask is a safe live gate.

A further [Song signal-calibration audit](../benchmarks/song_real8/SIGNAL_CALIBRATION.json)
uses only pre-formal native 1/2-block-per-class windows to estimate a bounded
per-channel RMS correction for the frozen F0+SPD model. It restores much of
the loss from a synthetic uniform 0.25× gain, but on unmodified S04 the
0/1/2-shot macro-F1 is 0.9508/0.9513/0.9438 and LogLoss worsens from
0.2376 to 0.2554/0.2601; S03 also worsens. Post-hoc amplitude summaries,
excluded from correction fitting, show source calibration/formal RMS ratios
of 0.30–0.53 versus S04 one-block ratios of 0.72–0.91. This measured
protocol-context mismatch makes simple calibration transfer unreliable, so
the method remains experimental and outside the live app.

The [recent Song reproduction recheck](../benchmarks/song_real8/REPRODUCTION_RECHECK.json)
reruns the four newest diagnostics in isolated temporary outputs. Their
versioned JSON artifacts match byte for byte in the current Python environment;
the audit binds source HDF5, ignored local model bundles, script hashes and
commands. This is targeted reproducibility evidence, not a rerun of all
historical model fits or a substitute for unavailable new-user/day data.

The [Song F4 conditional-increment study](../benchmarks/song_real8/F4_INCREMENT_RESULTS.json)
now compares source-fitted F0, F0+SPD, F0+F4 and F0+SPD+F4 on identical causal
S03/S04 trials. The saved source F0 and SPD scores replay before the new
comparison. F4 added to F0+SPD lowers S03/S04 LogLoss by 0.0124/0.0222, but
S03 macro-F1 drops from 0.9714 to 0.9640 and S04 remains about 0.9508.
All 1,136 trial probabilities and paired errors pass independent read-back.
The current selectable live bundle remains F0+SPD; this one-person/day,
previously inspected S04 result does not support a deployment change.

The four supplied Song recordings also support a fixed F0+SPD
[leave-one-session-out check](../benchmarks/song_real8/SESSION_HELD_OUT_RESULTS.json):
three sessions' valid stable formal trials train each fold and the fourth contributes only cued stable
trial scores. Across 569 held-out trials, pooled macro-F1 is 0.9522;
individual S01/S02/S03/S04 values are 0.9222/0.9713/0.9722/0.9437. These
are within-day, one-person folds, not multiuser or multiday validation. The
S04 fold is slightly worse than the existing S01/S02-trained F0+SPD result,
so the live bundle is unchanged.

On those same Song folds, a fixed [14-arm F0-family screen](../benchmarks/song_real8/FAMILY_HELD_OUT_SCREEN.json)
keeps all 569 trial identities matched and exactly replays the prior F0+F2c
probabilities. Adding F2a to that core lowers pooled LogLoss by 0.0154 and
does so on all four within-day folds, while macro-F1 rises only 0.0018.
F4 lowers pooled LogLoss by 0.0104 but its fold signs split two positive/two
negative; F5 worsens pooled LogLoss. These are exploratory current-family
results for one person/day, not historical algorithm reproduction or a reason
to change the live bundle.

Stratifying those frozen Song predictions by the native
[arm-cue labels](../benchmarks/song_real8/CUE_ARM_DOMAIN_AUDIT.json) reveals a
limitation hidden by the pooled scores: 283/569 trials are `still`, versus
47–48 in each moving-arm group. Core F0+F2c macro-F1 is 0.993 on `still`
and 0.828 on `up`; adding F2a raises the latter to 0.867 but worsens
`backward` LogLoss by 0.011. F6 helps `up`/`down` LogLoss while harming
`right`/`forward`. These descriptive cue-group differences are not measured
posture robustness or evidence of new-day/device transfer.

Rechecking F2a on the original Song S01/S02 source, S03/S04 held-out split
([fixed-split study](../benchmarks/song_real8/F2a_INCREMENT_RESULTS.json))
confirms a probability-versus-decision tradeoff. F2a added to F0+F2c
improves S03/S04 LogLoss by 0.0217/0.0227 but reduces macro-F1 by
0.0071/0.0149; S04 Index Pinch recall falls from 30/36 to 28/36. The
current realtime model therefore remains F0+F2c. Neither this one-day
result nor the prior leave-one-session-out screen proves a deployable gain.

The project-specific Song desktop launcher now passes `--song-realtime`,
opens the realtime tab and loads the existing hash-checked F0+F2c bundle.
Previously, the same shortcut launched the generic page, whose first
discoverable model was UniBo. An offscreen full-window smoke test confirmed
the current shortcut's preferred model loads; the generic collection
shortcut remains unchanged. This resolves model-selection ambiguity at
launch but is not a physical-device recognition test.

The separately preregistered [GRABMyo public cross-day screen](../benchmarks/grabmyo_crossday/REPORT.md)
adds 672 verified trials from eight selected subjects across three days, using
the publisher's first forearm F1–F8 ring. A fixed day-1 model was tested on
day 2 and then day 3. On final day, F0 macro-F1/LogLoss/Brier were
0.9008/0.3007/0.1552; F0+F2a gave 0.8536/0.7640/0.2042 and F0+F4 gave
0.9002/0.4767/0.1567. Thus neither added family demonstrated a robust
cross-day improvement in this limited four-class public setting. Different
hardware, subject selection and trial-level labels keep this separate from
own-device real-time claims and historical DS2 force reproduction.

An [independent ring-family reconstruction](../benchmarks/historical_reconstruction/REPORT.md)
now separates the current code's RLCS-like lag correlation and CES-like
eigen-spectrum blocks. In held-out LibEMG Electrode Shift users, F0 plus
reconstructed CES improves pooled final macro-F1 from 0.4912 to 0.5184,
whereas reconstructed RLCS alone lowers it to 0.4875. The validation-best
combined arm does not retain its validation advantage on final users. These
new operational definitions let forward Feature Bank work continue without
the old source, but do not prove historical formula or effect equivalence.

The [new bank v1](../benchmarks/new_bank_v1/REPORT.md) independently implements
four opt-in eight-channel families rather than wrapping old reference code.
Source-only public force and wearing screens were preregistered before results.
Both validation-best combinations regress below F0 macro-F1 on independent
final subjects. The families are available for further research, but no v1
combination is enabled in the live recognizer or claimed to solve own-device
hand-open recognition.

The versioned [new-bank v2 spatial screen](../benchmarks/new_bank_v2/GRABMYO_REPORT.md)
adds independently coded source-Rest F0, trace covariance F2a and ring-relative
covariance F3c. On the public eight-channel GRABMyo Day1/Day2/Day3 split,
the new F0 reproduces 448 prior held-out baseline probabilities; Day2
selects F0 alone. Adding either spatial family lowers Day3 pooled macro-F1
and increases log loss. F3c's better Day3 minimum-subject F1 is final-only
descriptive evidence, not a selection result or own-device validation.

On the user's own 250 Hz Song recordings, the [frozen v2 follow-up](../benchmarks/new_bank_v2/SONG_REPORT.md)
selects F0v2+F2a+F3c on S03 (macro-F1 0.9714 versus F0v2 0.9277).
The same arm scores 0.9366 versus 0.8919 on S04 and achieves 36/36
open-hand recall in its stable trial intervals. Its old F0 reference exactly
replays all 284 previous held-out probabilities. These one-person, one-day
offline results do not override the public cross-day regression or justify a
live-bundle switch; S04 was previously examined and is exploratory.

The matching [28-state Song screen](../benchmarks/new_bank_v2/SONG_28_REPORT.md)
adds the real IMU to every v2 arm and scores all seven arm cues times four hand
states. The S03-selected arm is F0v2+IMU alone (joint macro-F1 0.6125); its
S04 joint macro-F1 is 0.5365, below the reproduced previous F0+IMU baseline
0.5487. F2a/F3c hand-only gains do not transfer to the joint endpoint.
The prior 28-state baseline reproduces only in its recorded Python 3.13/NumPy
2.4/SciPy 1.17 runtime; a separate Python 3.11/NumPy 1.26/SciPy 1.14
environment changes one S04 decision. The live model remains unchanged.

The [guided Song IMU arm-calibration test](../benchmarks/new_bank_v2/SONG_ARM_CAL_REPORT.md)
uses the actual pre-formal six-direction and rest blocks without fitting on
target formal trials. With identical source-trained hand probabilities, a
source-trained arm classifier scores S03/S04 joint macro-F1 0.7159/0.5789;
per-session guided arm prototypes score 0.3272/0.5386. The factorized source
model improves S04 macro-F1 versus the earlier joint classifier but lowers
S04 accuracy from 0.6944 to 0.6458. Guided blocks do not supply a measured
body-forward axis or a validated live calibration solution.
The [post-hoc shift diagnostic](../benchmarks/new_bank_v2/SONG_ARM_SHIFT_REPORT.md)
finds that only 3/7 S03 and 5/7 S04 formal arm centroids are nearest their
own guided IMU prototype; this explains the risk of deploying the guided rule
without identifying the physical cause. Separately, [source-session OOF
probability calibration](../benchmarks/new_bank_v2/SONG_28_SOURCE_CAL_REPORT.md)
selects hand/arm temperatures 0.5/0.75 from S01/S02 only. It lowers factorized
joint LogLoss on S03 from 1.0553 to 0.8858 and on S04 from 1.1887 to 1.0157;
all 28-state decisions and macro-F1 values remain unchanged. This improves
probability quality in the same-day offline study, not live recognition.

The independently sourced [Zenodo electrode re-placement check](../benchmarks/new_bank_v1/ZENODO_REPLACEMENT_REPORT.md)
adds a separate file-level P1-to-P2/P3 position-shift test of the new bank.
The P2-selected ring-lag-plus-correlation arm improves P3 mean-subject top-1
from 0.6003 to 0.6188 over F0 on 79 matched whole recordings from nine
subjects. This is limited corroboration for a new representation, not a
time-local gesture or live-device result; the deployment decision is unchanged.

## Frozen standalone G5 versus DTW on UniBo Days 7–8 (2026-09-28)

The Day-6 standalone study saved each user's source-only G5 classifier, DTW
templates, and Day-5 probability temperatures. The new final run
`feature_bank_unibo_sequence_temporal_final_20260928` loads those states without
fitting and scores both methods on exactly the same 3,391 complete bouts from
Days 7–8. Native trials, source artifact hashes, split IDs, and all 14 user-by-
method probability arrays replay exactly (maximum absolute difference 0).

| Method | Weighted macro-F1 | Accuracy | Log loss |
| --- | ---: | ---: | ---: |
| G5 | 0.753845 | 0.836147 | 0.809240 |
| Standalone DTW | 0.371166 | 0.391969 | 1.235459 |

Their prediction disagreement is 0.564798. Weighted mass with G5 correct and
DTW wrong is 0.480312; the reverse is only 0.036133. A low error correlation
(0.155461) therefore does not imply a useful standalone DTW predictor here.
The 36 score rows and 18 paired-error rows are in the canonical result tables.
All bouts use oracle ground-truth boundaries, so this is an offline sequence
comparison, not an unsegmented streaming or own-device result. Earlier project
experiments had already examined Days 7–8; this added comparison is descriptive
confirmation, not a newly untouched final test.

## Newly reconstructed RLCS by Personal Anchor on EPN (2026-09-28)

The [prespecified protocol](../benchmarks/reconstructed_rlcs_anchor_protocol.json)
tests an eight-feature circular-lag envelope-correlation family against the
frozen F0 provider, with and without 1/2/5 whole calibration trials per class.
This is `RLCS_reconstructed_v1`, not the unavailable historical RLCS. Source
users 1–15 fit one population classifier; their three user-held-out folds fit
the probability temperature. Validation users 16–18 and descriptive final
users 19–21 retain the existing exact trial splits. No population model or
temperature is refit on target users.

| Phase and shots/class | F0 F1 | F0+RLCS F1 | F0+Anchor F1 | Joint F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation, 1 | 0.3946 | 0.4106 | 0.3676 | 0.3735 |
| Validation, 2 | 0.3981 | 0.4210 | 0.3638 | 0.3657 |
| Validation, 5 | 0.4104 | 0.4344 | 0.3730 | 0.3319 |
| Final, 1 | 0.4455 | 0.4334 | 0.3516 | 0.3599 |
| Final, 2 | 0.4462 | 0.4329 | 0.4213 | 0.4120 |
| Final, 5 | 0.4518 | 0.4382 | 0.3841 | 0.3705 |

The reconstruction gives no stable final gain, and the joint arm is worse
than F0 at every budget. The final pooled log loss also rises for all added
arms. A positive second-difference interaction in negative log loss therefore
does not justify promotion: it measures combined effects relative to the two
individual losses, not an absolute win. Native-trial replay reproduced all 54
saved prediction arrays per phase with zero difference; the F0 baseline also
matches the earlier frozen EPN interaction arrays exactly. Canonical tables
contain 96 four-arm score rows, 72 conditional increments, 24 reconstructed
population-versus-anchor error rows, and 24 interaction rows. Final users were
inspected in earlier project experiments, so these results are descriptive.

The optional F7 classwise shrinkage-Mahalanobis distance now has an independent
quadratic-form test and rejects calibration sets with fewer than feature
dimension + 2 samples in any class. The existing native 1/2/5-shot high-dimensional
calibration protocols fail that eligibility rule, so this implementation adds
formula coverage without a native performance claim or a deployment change.

## Prospective Song session eligibility gate (2026-09-28)

The [frozen Song protocol](../benchmarks/song_real8/PROSPECTIVE_FREEZE.json)
pins the current live F0+SPD bundle and all four previously inspected source
session hashes. The [gate](../benchmarks/song_real8/prospective_gate.py) rejects
an earlier or reused session, a modified model, incomplete collection readiness,
or disagreement between HDF5 metadata and the sidecar records. It correctly
rejects existing S04; a synthetic fresh-format session passes, and a changed
model fails. A future session must be collected after the freeze date before
this gate can be used for blind scoring. Passing is only an eligibility check:
it does not itself validate live accuracy, new wearer generalization, measured
gesture onset, or actual electrode re-placement.

The four existing Song recordings preserve host-clock timing for their 26 guided
calibration blocks. A [dual-clock audit](../benchmarks/song_real8/CALIBRATION_CLOCK_AUDIT.json)
checks frozen HDF5 hashes and independently reads block events and EMG packet
reception times. All four show a 64.0-second first-to-last event span; received
EMG spans are 63.969–64.000 seconds. The hand-state portion through the last
post-open rest is 32.0 seconds by events and 31.969–31.985 seconds by packet
reception. From session start to calibration end is about 66 seconds. These
numbers quantify this one-day scripted collection protocol only. They exclude
electrode preparation, operator interaction before recording, and the separate
live-recognition calibration flow; S01–S03 also failed collection readiness.

## Publisher DS2 v9 force labels and fixed active-gesture screen (2026-09-28)

The publisher added `LabelForces_All.mat` in Kaggle version 9. Its 332,108
window codes are uniform within every one of the 2,863 raw-trial blocks.
Version-9 MAV and gesture files match audited version-8 bytes exactly, binding
the new force codes to the existing exact raw-to-window and TDMS subject joins.
The 30 subject-unmatched trials retain force labels but cannot enter a cross-user
split; one additional trial has mixed gesture labels. The resulting fixed
four-active-gesture study scores 2,297 subject-verified trials across two
source-training modes and three F0/F1 arms. A second run exactly reproduced
the results, and all 5,454 saved probability rows independently rescore.

With source users trained on low and average force only, final descriptive
high-force macro-F1 is 0.4767 for F0, 0.4730 for reconstructed F1 alone and
0.4845 for F0+F1. With all three source force levels, final all-force F1 is
0.3771/0.4551/0.3998 respectively. Thus F1 alone helps in the product-mode
screen, but the fixed concatenation does not reliably dominate F0 and the
unseen-high final comparison is nearly tied. These subjective, three-channel
public results neither recover historical X1-H nor validate own-device live
recognition. Earlier gesture-only analyses had already inspected final users.

## Public DS2 v9 personal force calibration (2026-09-28)

The publisher's verified trial force codes now support a separate fixed
0/1/2/5-shot personal-calibration study on public DS2. The source-user model
is replayed exactly. In unseen-high mode, calibration uses only low/average
force and the test set consists entirely of high-force trials. Product mode
reserves the same five trials per gesture at every budget; those trials never
enter evaluation. The [study report](../benchmarks/discovery/public_ds2_force_v9/REPORT.md)
and [verification](../benchmarks/discovery/public_ds2_force_v9/CALIBRATION_VERIFICATION.json)
include all individual scores and audit checks.

For reconstructed F1 alone, validation high-force macro-F1 rises from .366
at zero shots to .574 at five; descriptive final users rise from .473 to .791.
Product all-force F1 rises from .354 to .576 on validation and .461 to .738
on descriptive final users. The four final users vary substantially, and
F0+F1 concatenation does not inherit the F1-only gain. The canonical
`delivery/calibration_curve.csv` now includes 240 pooled/per-subject records
from this frozen study, with 0/40/80/200 seconds of recorded trial signal
for the four-gesture budgets. These are subjective-force, three-channel,
trial-level results; they do not set an eight-channel live recognition model
or establish historical X1-H equivalence. A subsequent frozen-prediction
paired analysis shows F0+F1 loses to F0 on validation at five shots in both
force modes even while F1 alone improves; all three arms use identical
native trial IDs. This [negative conditional-increment result](../benchmarks/discovery/public_ds2_force_v9/CALIBRATION_PAIRED_RESULTS.json)
prevents interpreting F1's standalone gain as evidence for naive concatenation.

## New v2 eight-channel electrode-shift result

A frozen new experiment uses native LibEMG CIILData at 200 Hz with each subject's
unshifted trials as source and four shifted wearing domains as target. Subjects
15–17 select the arm; subjects 18–20 evaluate that fixed choice. The selected
F0v2+F2a+F3c reaches final macro-F1 0.6560 and worst-domain F1 0.6027,
versus the exactly replayed prior F0 at 0.4912 and 0.3938. F0v2 alone also
improves, while F2a does not consistently help on final users. The unselected
F0v2+F3c arm reaches final macro-F1 0.7737; that final-only ranking cannot
change the validation choice. All 1,200 native-trial predictions are saved,
independently rescored and byte-reproducible. See the
[new wearing report](../benchmarks/new_bank_v2/WEARING_REPORT.md). This is
public-device, trial-level evidence, not a live own-device approval.

A [fixed one-shot same-domain calibration follow-up](../benchmarks/new_bank_v2/WEARING_CAL_REPORT.md)
uses each of the two native target trials per gesture once as calibration and
once as held-out evaluation in reciprocal folds. The previous F0v2-only arm
rises from final macro-F1 0.6170 at zero shots to 0.9338 at one shot/class;
the previously selected F0v2+F2a+F3c arm rises from 0.6560 to 0.9163.
The full arm has slightly lower calibrated log loss, while F0v2 has higher
calibrated F1. All source-only probabilities exactly replay the zero-shot
parent, and calibration/evaluation trial IDs are disjoint. This public
same-domain result supports the value of a short guided calibration, but does
not decide the own-device model or later-day performance.

An independently frozen [new-v2 cross-user intensity screen](../benchmarks/new_bank_v2/FORCE_REPORT.md)
on native eight-channel LibEMG force data provides the counterpoint. The
validation-selected F0v2 arm improves macro-F1 from 0.4835 to 0.6295 on
validation users but falls below F0 on final users, 0.4939 versus 0.5118;
final worst-condition F1 falls from 0.4102 to 0.3579. F2a is the final-only
best arm and cannot be selected retrospectively. The new spatial bank is
therefore condition-specific and calibration-sensitive, not a validated
universal default representation.
The frozen-trial [paired analysis](../benchmarks/new_bank_v2/FORCE_REPORT.md#paired-feature-family-analysis)
shows why: adding F2a or F3c separately reduces validation F1 by about
0.064, and their positive validation interaction still leaves the combined
arm below F0v2. On final users, F2a alone adds 0.0418 F1 but the combination
loses 0.0142, with a negative interaction. These descriptive final-user
comparisons do not alter the validation selection.

A frozen [new-v2 force personal-calibration follow-up](../benchmarks/new_bank_v2/FORCE_CAL_REPORT.md)
uses the parent validation-selected F0v2 on public eight-channel intensity
trials. For each target user and ten four-repetition conditions, it reserves
repetitions 1–2 for 0/1/2-shot class prototypes and evaluates all budgets on
the same repetitions 3–4. On descriptive final users, macro-F1 rises from
0.5187 to 0.7254/0.7369 and worst-condition F1 from 0.3320 to
0.5974/0.6718. Final minimum-subject F1 remains only 0.4836/0.4851 after
calibration, so a pooled gain does not solve all users. All source-only
probabilities replay the frozen parent; 3,360 prediction rows and 40 nested
assignments pass independent verification. MVC lacks enough native trials
for the matched 2-shot protocol and is excluded. This does not establish
new-day or own-device live reliability.

A [frozen single-reference follow-up](../benchmarks/new_bank_v2/FORCE_REF_CAL_REPORT.md)
tests the calibration burden directly. One/two labelled Medium trials per
gesture cost about 21/42 seconds of recorded signal per user versus about
211/423 seconds for separate calibration at all ten intensities. On the same
final evaluation trials, F0v2 macro-F1 reaches only 0.5781/0.5794 with the
shared reference versus 0.7254/0.7369 with per-condition prototypes. On
validation, one shared shot lowers F1 below the 0-shot control. The 10×
burden reduction therefore sacrifices much of the gain; it is not a validated
short-calibration default. All 3,360 predictions and four nested assignments
pass independent checks, and the 156 additional calibration-curve records
are provenance-verified.

The new-v2 eight-channel candidate full bank now has a matched
[leave-one-family-out study](../benchmarks/new_bank_v2/LOFO_REPORT.md) on both
public wearing and cross-user intensity trials. F0v2 is useful in all four
phase–dataset comparisons. F2a's inclusion lowers descriptive final wearing
F1 by 0.1178, while F3c's inclusion lowers descriptive final force F1 by
0.0560. The complete three-family concatenation is therefore not a stable
common default. All 5,664 arm–trial predictions, 1,416 native trials, exact
parent replays and 176 canonical ablation rows pass independent checks.

The [new-v2 Stage-1 family screen](../benchmarks/new_bank_v2/FAMILY_SCREEN_REPORT.md)
now consolidates four saved eight-channel prediction matrices. It records
macro-F1, log loss, Brier, ten-bin ECE and per-class F1 in 256 pooled,
subject and condition cells. The seven-axis robustness vector has direct
new-v2 evidence for force, wearing and day; independent user, posture,
speed and real quality remain N/A, while the one-person Song sessions are
a separate same-day supplement. F3c is a wearing specialist candidate, not
a general default; F2a is inconsistent across validation and descriptive
final cohorts. The canonical family table and all 52,012 provenance rows
pass the source-bound audit.

The independent [new-v2 paired-error and interaction study](../benchmarks/new_bank_v2/PAIR_REPORT.md)
matches all four candidate arms on 2,148 native trials before comparison.
Its 384 arm-pair cells quantify error overlap and prediction disagreement;
64 more cells test the F2a×F3c interaction without retraining. On final
force, F2a-only is correct and F3c-only wrong on 60 matched trials, with
the reverse on 31; on final electrode shift those counts are 9 and 27.
Their errors therefore differ, but combining them does not consistently
beat F0v2: final force and GRABMyo day joint macro-F1 are 0.0142 and
0.0441 below core, while electrode shift and Song same-day are above it.
The interaction sign alone is not an absolute performance gain. The 384
canonical pair rows and all 52,396 provenance records pass verification;
the 64 interaction rows retain their own versioned table and hash audit.

The [new-v2 Stage-2 conditional analysis](../benchmarks/new_bank_v2/CONDITIONAL_REPORT.md)
then asks whether F2a or F3c still adds value when the other is already in
the core. Its 128 matched cells show F3c's consistent held-out conditional
log-loss and macro-F1 benefit on public electrode shift. F2a is unstable
there and harms both metrics on the descriptive final split. Neither
addition improves pooled force log loss in both phases; F2a's cross-day
conditional loss also worsens. These contrasts support a wearing-specialist
interpretation for F3c, not a universal concatenated bank. The canonical
conditional table and all 52,524 records pass source-bound verification.

The [new-v2 robustness envelope](../benchmarks/new_bank_v2/ENVELOPE_REPORT.md)
adds the requested mean of the three observed axes and their minimum for
each candidate and phase. The full arm's validation equal-axis mean is
0.7424, but its worst axis is 0.5671 versus F0v2's 0.5775. A mean gain
therefore does not establish a raised robustness floor. Independent user,
posture, speed and real quality remain N/A for this exact candidate set;
Song is separately identified as a same-day one-person supplement.

A frozen [new-v2 GRABMyo score-anchor calibration curve](../benchmarks/new_bank_v2/GRAB_SCORE_CAL_REPORT.md)
tests a minimal same-user correction without retraining: 0/1/2/5 labelled
recordings per gesture calibrate a probability-space prototype, and the
same repetitions 6–7 remain held out for every budget. On descriptive
Day3 F0v2, 1 shot raises macro-F1 from 0.9210 to 0.9686, while the full
bank and F3c curves are non-monotonic. This is evidence that short labelled
calibration can repair some public cross-day score drift; it is not a
raw-feature F7 test or evidence for own-device live deployment. All 2,048
predictions and 288 canonical curve cells pass independent replay and
source-bound verification, now covering 52,812 canonical records.

The separate [new-v2 feature-space prototype study](../benchmarks/new_bank_v2/GRAB_FEATURE_CAL_REPORT.md)
tests the stronger F7-style question using source-fitted F0v2/F2a/F3c
recording features and labelled target-day prototypes. Its frozen 50/50
mixture harms Day2 log loss for every arm and nonzero budget; F0v2 Day3
1-shot F1 stays at 0.9210 while log loss worsens from 0.2349 to 0.4870.
This negative result rules out that fixed uncalibrated feature-anchor
mixture as a current default, even though the earlier score-space anchor
helped on the same held-out recordings. All 2,048 arm–budget–trial outputs,
official checksums, source-fitted transforms, and 53,100 canonical rows
are auditable. The fresh source classifier replays every top-1 label but
has at most 2.72×10⁻⁵ probability difference on the F3c arm; the saved
population probabilities remain authoritative for the calibration test.

The earlier independent [new-v1 scale×frequency and ring×correlation-spectrum
paired study](../benchmarks/new_bank_v1/PAIRED_REPORT.md) now fills the
same finite Stage-2–4 analysis for the freshly implemented F1/F4-like and
ring/global-coordination candidates. On force validation, frequency
direction conditionally improves log loss by 0.0261 and F1 by 0.0256,
but on descriptive final users those additions are −0.2420 and −0.0105.
On wearing, each ring candidate conditionally lowers log loss in both
phases, but each reduces final F1. Forty-four exact-trial error cells and
forty-four interaction cells show why single-metric promotion would be
unsafe. This is forward new-family evidence, not historical formula
equivalence. The canonical provenance check now covers 53,408 rows.

## Reduced new-v2 MANUS spatial and speed-stratified session study

The [frozen reduced-bank study](../benchmarks/new_bank_v2/MANUS_SPATIAL_REPORT.md)
adds an eight-channel 200 Hz public check of new F2a/F3c spatial blocks on
six finger flexion–extension gestures. MANUS has no Rest class, so the
rest-conditioned F0v2 is invalid here; source-fitted original F0 is the
explicitly separate baseline. Same users' Session 1 trains the models;
Sessions 2 and 3 provide validation and descriptive final recordings, with
slow/medium/fast conditions in each. Speed and session effects remain
confounded, and this study does not fill the full-bank speed-axis `N/A`.

The fixed validation rule selects F0+F2a: pooled macro-F1 is 0.4717 versus
F0's 0.3410. Descriptive final F1 is 0.4869 versus 0.4453, while final log
loss worsens from 1.5920 to 1.7900 and the minimum-user F1 falls from
0.2381 to 0.2222. F2a/F3c conditionals and their interaction have opposite
loss/F1 directions in several cells. All 864 saved native-trial predictions,
80 score groups, 20 paired analysis cells, and the updated 53,548-row
canonical provenance are independently checked. No live-model change follows.

## Same-day GRABMyo cross-user new-v2 axis

The [frozen subject-disjoint study](../benchmarks/new_bank_v2/GRAB_USER_REPORT.md)
now evaluates the exact F0v2/F2a/F3c four-arm candidate set on Day1 of
the verified GRABMyo eight-channel subset: users 1–4 train, 5–6 validate,
7–8 provide descriptive final scores. F0v2 thresholds use only 560 Rest
windows from source users. The validation rule selects F0v2 at 0.8054
macro-F1 versus 0.7485/0.7350/0.7256 for the other three arms. F3c's
0.0004 final F1 edge is final-only, has worse log loss, and cannot alter
the choice. The 448 saved predictions, eight source-score groups, 24
pooled/user family cells and six matched error/interaction cells are
reproducible and independently checked.

The [versioned four-axis envelope](../benchmarks/new_bank_v2/ENVELOPE_USER_REPORT.md)
adds the separately identified user axis to force, wearing and day while
retaining the frozen three-axis table. F0v2 has the highest validation
four-axis mean (0.7419) and minimum (0.5775). Day and user are correlated
because they use different splits of the same public subset; posture,
speed and real quality remain unobserved for this exact bank. The
canonical provenance check now covers 53,590 records. None of this
validates the user's 250 Hz live device.

## Public ROAM-EMG static posture, exact new-v2 bank

The [frozen ROAM-EMG study](../benchmarks/new_bank_v2/ROAM_POSTURE_REPORT.md)
adds an eight-channel, nominal 200 Hz, three-class public posture test of
the exact F0v2/F2a/F3c candidate bank. All 112 static files from 28
subjects pass native sequence and sampling checks. Source users 1–18
contribute only resting-posture training bouts; unseen users 19–23
validate and 24–28 provide descriptive final bouts across four postures.
The 1,828 F0v2 noise-calibration windows are source-user Rest only.

Validation selects F0v2 at 0.9630 pooled bout macro-F1 and 0.8723
worst-posture F1. Its descriptive final values are 0.9162 and 0.8452;
hanging is the weakest posture in both splits. F2a raises final pooled
F1 to 0.9353 but lowers validation F1 to 0.9167, so it is not promoted.
All 1,440 saved arm–bout predictions, 60 matched paired cells, and
54,010 source-bound canonical records are auditable. These bout scores do not
measure online transitions, and relative to resting source users the
unseen-user and posture shifts coexist.

The [versioned five-axis envelope](../benchmarks/new_bank_v2/ENVELOPE_POSTURE_REPORT.md)
now reports force, wearing, day, user and posture for the exact new-v2
four-arm set. F0v2 retains the strongest validation equal-axis mean
(0.7861) and minimum (0.5775); the latter remains the wearing axis.
Speed and real signal quality remain N/A for this exact bank, and none
of the public 200 Hz posture result validates the user's 250 Hz device.

## Exact new-v2 synthetic quality stress test

The [frozen ROAM quality study](../benchmarks/new_bank_v2/ROAM_QUALITY_REPORT.md)
applies 13 source-prespecified faults only to held-out resting-posture
windows; source models and 1,828 Rest calibration windows are unchanged.
Its clean predictions replay the posture study for every target bout.
F0v2 has the highest validation synthetic-quality family mean (0.9110)
versus 0.7221/0.6915/0.6856 for the F2a, F3c and joint arms. The
descriptive final ordering is the same, but F0v2's worst individual
channel dropout falls to 0.4960/0.4929 F1 in validation/final. Mean
resilience is not a worst-channel guarantee; F2a/F3c arms are particularly
sensitive to the fixed 50 Hz contamination. All 5,040 saved predictions,
168 matched groups, 55,186 canonical records and the source-only clean
replay are audited.

The [six-axis synthetic envelope](../benchmarks/new_bank_v2/ENVELOPE_QUALITY_REPORT.md)
adds an explicitly labelled quality simulation to force, wearing, day,
user and posture. F0v2's validation equal-axis mean is 0.8069 and its
minimum remains wearing at 0.5775. The next section adds a qualified
speed test using an external source Rest prior; *real* hardware-quality
measurements remain unavailable. These
synthetic failures do not measure the user's device or live recognition.

## Qualified exact new-v2 speed axis using external source Rest

The [frozen MANUS external-Rest study](../benchmarks/new_bank_v2/MANUS_REST_TRANSFER_REPORT.md)
now evaluates all four exact F0v2/F2a/F3c formulas on the native
200 Hz slow/medium/fast MANUS trials. Because MANUS has no labelled
Rest, F0v2 thresholds come only from 1,828 source-user ROAM Rest
windows; MANUS Session 1 alone fits the classifier, and Sessions 2/3
remain held out. This explicitly qualified source-prior experiment
does not silently relabel MANUS activity as Rest.

Validation selects F0v2+F2a at 0.5244 macro-F1 versus F0v2's 0.3914,
raising the worst native speed from 0.2845 to 0.4677; validation log
loss worsens from 2.1203 to 2.3949. Descriptive final F1 is
0.5304 versus 0.4600. The 864 saved predictions, 56 matched groups,
80 parent-score replays and 55,578 canonical records are audited. All
speed categories appear in every session, so this is speed-stratified
session transfer rather than a previously unseen-speed experiment.

The [qualified seven-axis envelope](../benchmarks/new_bank_v2/ENVELOPE_SPEED_REPORT.md)
shows the resulting tradeoff: F0v2 retains the highest validation
equal-axis mean (0.7476), but F0v2+F2a has the better minimum axis
(0.5244 versus 0.3914). Synthetic quality and external-prior speed
are explicitly tagged; *real* quality and own-device recognition
remain unmeasured. No single-arm universal robustness claim follows.

## Independent first-stage feature-family extension

The [frozen ROAM extension](../benchmarks/new_bank_v2/ROAM_V1_EXTENSION_REPORT.md)
screens four additional independently implemented eight-channel family
formulas against F0v2 on the same unseen-user/posture split. None beats the
0.9630 pooled validation F1 of F0v2. Ring lag raises the worst validation
posture F1 from 0.8723 to 0.8963 while reducing pooled F1 to 0.9500;
it remains a condition-specialist candidate. Scale pattern and frequency
direction improve only the descriptive final outcomes. All 360 baseline
predictions exactly replay the frozen posture experiment within `1e-10`.
This is one independent Stage 1 axis, not a complete multi-dataset family
screen or a historical formula reproduction.

The [frozen GRAB user extension](../benchmarks/new_bank_v2/GRAB_V1_EXTENSION_REPORT.md)
screens the same four independent new-v1 families at 2048 Hz on a separate
same-day, subject-disjoint public dataset. Ring lag is the validation-selected
arm at 0.8360 pooled F1 versus F0v2's 0.8054 and corrects two parent errors
without introducing a validation error. On descriptive final users it loses
0.0559 F1 and introduces three errors, so this gain does not justify a live
model switch. Frequency direction improves minimum validation-user F1 but
lowers pooled F1. Saved baseline probabilities replay the frozen GRAB study,
and the independent read-back test checks both extension studies. The ROAM
and GRAB effects together support condition-specific investigation, not a
universal addition to the bank.

The [frozen GRAB cross-day extension](../benchmarks/new_bank_v2/GRAB_DAY_V1_EXTENSION_REPORT.md)
tests the same four additions with Day1 source, Day2 validation and Day3
descriptive final sessions on eight repeated public subjects. Ring lag wins
Day2 F1, 0.9686 versus 0.9551 for F0v2, but loses Day3 F1, 0.8862 versus
0.9008. Its matched error balance reverses from six corrected/three created
to three corrected/six created. Frequency direction's Day3 F1 edge is
final-only with worse log loss. All 2,240 saved target probability rows
pass read-back metric replay; frozen baseline probabilities differ by at
most `1.62e-9` across floating-point reruns. This further weakens a claim
that ring lag should be in the default full bank.

The [matched extension analysis](../benchmarks/new_bank_v2/V1_EXTENSION_PAIRED_REPORT.md)
replays 176 pooled, subject and ROAM-posture cells from all three saved
prediction sets. Ring lag improves both GRAB validation splits but worsens
both GRAB final splits, with corrected/created error counts reversing.
Other additions also show task- and objective-dependent directions. This
supports a specialist/redundant screening interpretation, not an automatic
full-bank inclusion. The analysis is descriptive and does not refit or
select models using final labels.
The 176 matched conditional and 176 error cells are also in the canonical
delivery; its provenance verification now covers 55,930 source records.

The [frozen ring-lag × frequency-direction four-arm interaction](../benchmarks/new_bank_v2/RING_FREQ_INTERACTION_REPORT.md)
tests one finite theory-motivated combination across the same three public
splits. All three validation groups show negative F1 and log-loss
interaction and absolute joint F1/loss regressions versus F0v2. The 3,680
held-out probability rows exactly replay the earlier single-arm screens;
44 matched pooled/subject/posture interaction cells are exported. A positive
interaction in one descriptive final split still coincides with a much
worse absolute joint arm. This combination is rejected as a default bank
addition under the frozen validation rule.

The [independent candidate full-bank LOFO](../benchmarks/new_bank_v2/RING_FREQ_LOFO_REPORT.md)
uses the same three splits and removes F0v2, ring lag or frequency direction
from `F0v2+ring_lag+frequency_direction`. Keeping F0v2 improves validation
F1 and log loss on all three tasks; removing either added family improves
validation F1 on all three. The full arm was already rejected by the
four-arm interaction test, so the ablation identifies contributions within
a failed candidate rather than selecting a deployment bank. All 3,680
saved predictions and 176 pooled/subject/posture LOFO cells are verified;
the required canonical ablation delivery now covers 56,106 records overall.

The [five-arm family-screen delivery](../benchmarks/new_bank_v2/V1_EXTENSION_FAMILY_REPORT.md)
replays pooled and per-subject/posture metrics, ten-bin ECE and per-class F1
for the four additional independent formulas on all three public splits.
All 220 native-target cells are now in the required feature-family table;
canonical provenance verification covers 56,326 records. Ring lag is a
conditional research candidate, while no addition demonstrates a stable
pooled validation advantage across all three studies.

The [source-subject OOF temperature study](../benchmarks/new_bank_v2/V1_SOURCE_OOF_CAL_REPORT.md)
fits each of the five arms on held-source-subject folds, chooses one
temperature from source OOF predictions, and applies it to frozen public
targets without target shots. Source OOF log loss improves by construction,
but target log loss improves in only 10 of 30 arm–phase groups; F0v2 gets
worse in five of six. Brier and ECE improve in only 6 and 8 groups.
No additional family becomes a stable calibration amplifier across the three
validation splits. The 7,090 saved source/target predictions and 440 target
score cells are audited and delivered as zero-shot calibration comparisons;
canonical provenance verification covers 56,766 records. Personal
calibration and own-device performance remain separate evidence questions.

The [new-v1 personal score-anchor follow-up](../benchmarks/new_bank_v2/V1_PERSONAL_SCORE_CAL_REPORT.md)
now evaluates the same five frozen arms at 0/1/2/5 labelled native trials
per class on GRAB unseen-user and cross-day splits. Repetitions 1..N form
each held-subject calibration set; repetitions 6/7 are fixed evaluation.
Ring lag remains a validation specialist but fails to beat calibrated F0v2
on descriptive Day3 F1 at any budget. Other apparent improvements likewise
vary by phase and objective; no added family earns a general personal-
calibration promotion. All 3,200 predictions, 400 assignments, 480 pooled/
subject score cells and 800 exact zero-shot replays pass read-back checks.
The required calibration curve and canonical provenance now cover 57,246
records. This fixed score-space correction is distinct from F7 feature-
space Personal Anchor or a device calibration workflow.

The [new-v1 feature-space anchor test](../benchmarks/new_bank_v2/V1_FEATURE_ANCHOR_REPORT.md)
now fits each arm's source scaler/classifier on GRAB Day1 only, then builds
same-user Day2/Day3 standardized feature prototypes from held-target
0/1/2/5-shot native trials. Its source probabilities replay the frozen
parent exactly and its 2,560 evaluated probabilities, 320 disjoint
calibration/evaluation assignments and 360 score cells are independently
checked. Under the fixed prior feature-anchor rule, **every** nonzero
arm/day/budget worsens pooled log loss versus its zero-shot source model,
although isolated F1 cells improve. This rule is rejected as a default;
it does not establish that all personal calibration methods fail. The
canonical delivery now verifies 57,606 records.

The [independent new-v1 force/intensity extension](../benchmarks/new_bank_v2/FORCE_V1_EXTENSION_REPORT.md)
screens all four candidate families against F0v2 on the exact frozen
LibEMG cross-user source/validation/final trials. Baseline probabilities
replay exactly. F0v2 remains the pooled validation-F1 choice; ring lag
improves validation worst-condition F1 but loses pooled F1 and log loss.
Correlation spectrum's pooled gain appears only on descriptive final
users. All 5,880 saved probabilities support 140 family-screen, 112
conditional-increment and 280 full pairwise-error cells. They are
delivered in the required canonical tables, now 58,138 verified records.

The [independent new-v1 wearing extension](../benchmarks/new_bank_v2/WEARING_V1_EXTENSION_REPORT.md)
fits each public subject's source `training` domain and evaluates all four
shifted electrode domains with the exact frozen F0v2 baseline. Correlation
spectrum wins pooled validation F1, yet lowers descriptive final F1 by
.0731 and worsens final log loss; its four corrected/zero created errors
reverse to three corrected/ten created. Ring lag improves log loss but loses
final F1 and worst-domain F1. The 1,200 held-domain probabilities support
80 family, 64 conditional and 160 complete pairwise-error cells in the
canonical delivery, now 58,442 verified records. Neither new-v1 addition
is promoted as a general wearing improvement.

The [qualified MANUS observed-speed extension](../benchmarks/new_bank_v2/MANUS_V1_SPEED_REPORT.md)
compares the same four new-v1 additions with F0v2 on six users, three
observed speeds and source/validation/final sessions. MANUS has no labelled
Rest, so all arms share the frozen external ROAM source Rest prior. Ring lag
wins validation pooled F1 but reverses on the descriptive final session,
where its F1 is .4136 versus .4600 for F0v2 and log loss worsens. The
final-only scale-pattern F1 gain cannot drive selection. No addition earns
a general speed-robust promotion. All 1,080 native-trial probabilities and
1,064 family, conditional and pairwise-error cells are checked; canonical
provenance verification covers 59,506 records.

The [independent new-v1 synthetic-quality extension](../benchmarks/new_bank_v2/ROAM_V1_QUALITY_REPORT.md)
tests the same four candidate additions on the frozen ROAM 13-fault grid.
F0v2 probabilities replay the parent across every matched native bout and
fault exactly. F0v2 leads the prespecified validation fault-family mean
F1, .9110; scale pattern reaches .9089 but has worse clean and minimum-
fault F1. Several additions improve descriptive final cells, but no
addition earns validation-backed general quality promotion. All 6,300
predictions support 840 family, 672 conditional and 1,680 pairwise-error
cells; canonical provenance verification now covers 62,698 records.

The [seven-axis new-v1 decision](../benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_REPORT.md)
binds posture, unseen-user, cross-day, intensity, wearing, observed-speed
and synthetic-quality validation results under one conservative rule.
None of the four independent additions avoids both F1 and log-loss
regression on every available axis. The selected default within this
finite candidate set is therefore F0v2. The 70-row validation/final
table and source hashes are audited; final rows are descriptive only.
This is a public-data bank decision, not 250 Hz own-device acceptance.

### New-version answers to research questions A–H

These answers apply to the seven public-data new-v1 screens and the
separate fixed calibration controls. The [decision table](../benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_CELLS.csv)
holds 70 matched axis–phase–arm cells, and the
[subject audit](../benchmarks/new_bank_v2/V1_CROSS_AXIS_SUBJECT_AUDIT.json)
holds 310 individual subject cells. The older seven-axis historical-bank
vector above uses different model compositions and must not be merged
numerically with this newer comparison.

| Question | New-version finding | Limit |
| --- | --- | --- |
| A. Missing information or organization? | The frozen F0v2 features remain the most consistent default; simply adding four distinct feature families does not fix all shifts. Feature organization and calibration can change results, but a single causal bottleneck is not isolated. | No physical diagnosis of the current 250 Hz device or proof that additional information is unnecessary. |
| B. Conditional increments? | Ring lag improves validation pooled F1 on unseen-user, cross-day and observed-speed splits; correlation spectrum improves wearing and observed-speed. Gains often cost log loss or reverse in final splits. | These are matched predictive increments, not measured mutual information or universal additions. |
| C. Specialists? | Ring lag is a task-specific GRAB/speed research candidate; correlation spectrum is a wearing/speed validation candidate. The independent force and synthetic-quality screens do not support either as a general default. | GRAB splits and ROAM posture/quality reuse recordings; specialist promotion needs a new independent validation cohort. |
| D. Only after personal calibration? | No new-v1 family has a consistent advantage *only* after the fixed score-space 0/1/2/5-shot correction. The fixed feature-space anchor worsens pooled log loss at every nonzero budget for every arm on both tested days. | Other personalization rules remain untested. |
| E. Personal Anchor and cross-user variation? | The tested feature-space rule is not promoted; the earlier EPN personal-anchor branch can lower mean/minimum F1 and increase user variation. | This does not reject all anchors or establish own-device cross-user effects. |
| F. Session Signature for day/re-donning? | A native wearing-domain test found no final F1 benefit from the fixed signature routing, although local prototypes can improve F1 with worse loss. | Wearing domains are not documented calendar sessions or physical re-donning on this device. |
| G. Worst scenario R_min? | Ring lag raises the descriptive seven-axis validation minimum from .3914 to .4264 by improving the lowest speed coordinate, but lowers the seven-axis mean from .7476 to .7450, has eight per-axis F1/loss guard violations and falls below F0v2 on final R_min (.4136 versus .4600). It also lowers the quality minimum subject fault-family mean from .816 to .632. No robust overall envelope gain is established. | Axes have unlike tasks and correlated recordings; synthetic quality is not real fault prevalence. |
| H. Calibration needed in product? | The public GRAB 1/2/5-shot protocols consume 20/40/100 seconds of labelled signal, excluding setup and rest; no budget supplies a reliable all-axis recovery. The selected public-data default is zero target shots. | Actual setup time, cross-session benefit and live 250 Hz device accuracy cannot be prescribed without hardware and suitable recordings. |

The [new-version reproducibility audit](../benchmarks/new_bank_v2/V1_REPRODUCIBILITY_AUDIT.json)
checks all seven screen packages: protocol/result/prediction hashes,
19,060 saved prediction rows, source/validation/final native-trial
separation, eight-channel sample contracts and explicit classifier seeds.
It checks saved evidence, not a full raw-archive rerun of every experiment
or a physical-device test.

The [formula audit](FORMULA_IMPLEMENTATION_AUDIT.csv) now records the
source-Rest-fitted F0v2 backbone and four independent new-v1 family classes
beside the older reference candidates, including exact source symbol spans,
dimensions, an analytical six-block F0 fixture, and the explicit boundary
that they do not reproduce unavailable historical X1-H/RLCS/CES/Frequency
code. Identical-channel rank-one and four known-tone tests additionally
check the new F3 lag/spectrum and F4 sub-Nyquist formulas. The
[new-v1 canonical schema check](../benchmarks/new_bank_v2/V1_CANONICAL_SCHEMA_AUDIT.json)
finds no missing required values in 5,312 seven-axis family, conditional
and error rows. The global delivery still contains older evidence with
honest N/A fields; new-v1 completeness does not repair those records.

The [matched F2 wearing comparison](../benchmarks/new_bank_v2/F2_WEARING_CANDIDATES_REPORT.md)
separately adds trace covariance, uncentered document CSP, or SPD tangent
coordinates to F0v2 on identical public wearing-domain trials. Validation
macro-F1 is 0.5775 for F0v2, 0.6960 for F2a, 0.5444 for F2b and 0.6110
for F2c. The favorable F2a and F2c validation results reverse on pooled
final F1; the favorable F2b final result cannot retroactively select it.
Formula-level analytical checks, source-only fitting, trial identity and
saved probability readback are separately recorded. The
[matched MANUS session comparison](../benchmarks/new_bank_v2/F2_MANUS_CANDIDATES_REPORT.md)
adds a second public axis for all three candidates on 108 native trials in
each target session. F0v2 probabilities replay the prior MANUS study
exactly. All additions improve validation macro-F1 but worsen validation
log loss; F2a has the strongest validation F1 and minimum-user F1. The
[GRAB unseen-user comparison](../benchmarks/new_bank_v2/F2_GRAB_CANDIDATES_REPORT.md)
adds a third public axis with a different 2048 Hz sensor and disjoint users.
All three F2 additions lower its validation and descriptive final macro-F1,
raise log loss and reduce minimum-user F1 against F0v2. These three axes
therefore reject an unconditional F2 default while preserving their bounded
condition-specific evidence.

The [F0–F9 public-evidence scope table](FORMULA_FAMILY_SCOPE_AUDIT.csv)
separates native screens, bounded context/calibration evidence, synthetic
quality controls and unavailable device conditions for all ten families.
It also records the exact remaining boundary for each family. A known-path
DTW test checks Euclidean local cost, warp-band behavior and division by
path length, alongside its frozen complete-bout UniBo replay. This scope
table does not convert fixture dimensions or public proxy data into full
formula or live-device acceptance.

The [MANUS F4d session diagnostic](../benchmarks/new_bank_v2/F4D_MANUS_SESSION_REPORT.md)
now computes equal-trial long/current log-band references for six users
with 108 long-term, 72 current-calibration and 144 held-out native trials.
Its session-relative vectors are finite and trial-disjoint. The
[matched F4d predictive control](../benchmarks/new_bank_v2/F4D_MANUS_PREDICTIVE_REPORT.md)
compares F0v2, long-centered and session-centered spectra on 432 saved
predictions. Session-centering improves validation pooled macro F1 from
0.3898 to 0.4764, but final minimum-user F1 falls from 0.1111 to 0.0952.
Medium-speed calibration versus slow/fast evaluation confounds speed with
session, so this is bounded predictive evidence, not a default promotion
or a verified fatigue effect.

The [matched-speed F4d control](../benchmarks/new_bank_v2/F4D_MANUS_SAME_SPEED_REPORT.md)
uses the other five gesture trials of the same user, session and speed as
calibration for each excluded native trial. On the same frozen classifiers,
validation pooled F1 rises from 0.3898 to 0.4852 and loss falls from 2.2394
to 1.7187. Final minimum-user F1 still falls from 0.1111 to 0.0952. This
removes the medium-versus-slow/fast mismatch, but the reference omits the
held gesture and requires offline leave-one-out selection; it is not a live
calibration result or default promotion.

The [complete new-v1 bank LOFO](../benchmarks/new_bank_v2/FULL_V1_LOFO_REPORT.md)
now tests F0v2 plus all four independent additions and every
leave-one-family-out arm on all seven public axes. The [readback audit](../benchmarks/new_bank_v2/FULL_V1_LOFO_AUDIT.json)
binds 26,684 saved predictions and 98 scored cells to frozen parent
controls. Full bank loses macro-F1 against F0v2 on six of seven validation
axes, so the generic F0v2 default remains unchanged. Positive final-only
results are descriptive rather than a basis for promotion.

An independent [F4b/F4c known-signal check](../tests/test_f4_spectral_known_signal.py)
computes the periodogram from direct Fourier sums, then verifies total power,
centroid, median frequency, entropy and four non-DC cepstral mean/std pairs,
including a zero-signal channel. It verifies the implemented numerical
convention; it does not establish historical CCA identity or native benefit.

The optional F5c order-two path signature requires a producer-certified
complete sequence with a native duration of at least one second, like F5b.
A [known three-point path test](../tests/test_f5c_complete_signature.py)
checks both tensor levels, repeated-vertex invariance, scale invariance and
short-window rejection. A separate
[full-bout UniBo comparison](../benchmarks/new_bank_v2/F5C_UNIBO_BOUT_REPORT.md)
uses the entire oracle-labelled RMS-envelope trajectory, with 15,273 saved
matched predictions and exact replay of frozen G5 parent arrays. G5+F5c
raises Day6 macro F1 from 0.7713 to 0.8297 but worsens log loss; Days7-8
pooled F1 slips from 0.7538 to 0.7504. F5c remains optional, with no
live-segmentation or eight-channel transfer claim.

The generic F8 session signature also has a
[hand-computable three-class geometry check](../tests/test_f8_known_geometry.py)
for residual norms, cosine agreement and every pairwise distance change.
It verifies the frozen-source formula, not a predictive routing benefit or
same-user acquisition provenance.

The [versioned major-clause acceptance matrix](NEW_VERSION_ACCEPTANCE_AUDIT.csv)
separates completed scoped evidence, superseded historical-source demands,
and genuine open work. Remaining applicable F0–F9 formula/native-eligibility
checks are still open. Own-device session/physical validation is explicitly
deferred by data availability. The matrix does not replace the separate
78-section source audit or claim full specification acceptance.

An optional [source-calibrated F9 quality-mask candidate](../benchmarks/song_real8/QUALITY_MASK_V1.json)
now combines the F9v2 zero, flatline, ADC-clipping, amplitude, correlation,
line-noise and low-frequency observations into eight channel scores plus four
summaries. S01/S02 alone fix the thresholds; S03/S04 are read-only. Their
mean candidate quality is 0.998/0.982, and setting channel 1 to a constant
in every held-out window yields zero channel-1 quality in both sessions.
Joining the unchanged, source-S01/S02 F0 trial predictions by native trial ID
gives an adverse selective-recognition control: a minimum-channel-quality
cutoff of 0.5 rejects 4/140 S03 trials, all four previously correct, and
29/144 S04 trials, including 25 previously correct and four errors. S04
coverage falls to 79.9%; this rule does not justify a deployed Unknown gate.
This is a synthetic-fault detection check on one person's same-day raw ADC
recordings, not a measured hardware-fault rate or a selected live gate. The
existing deployed quality and Unknown decisions remain unchanged.

For F6, a constant lateral-acceleration and constant-rotation oracle now
checks every one of the 15 calibration-relative body-context outputs against
an independently calculated gravity-filter trajectory. The existing
rotation-invariance and held-out-calibration guards also pass. This verifies
the numerical formula, while native evaluation of this calibrated body frame
still needs a dataset with explicit neutral/forward calibration provenance.

The versioned formula implementations must be distinguished from retained
reference families. `RestNoiseDetailV2` implements the source-Rest F0 threshold
rule; `DocumentCspFamily` implements the uncentered top-two/bottom-two F2b
rule and has matched public-data increments; and `CalibratedBodyContextFamily`
implements the 15-dimensional F6a body-frame rule above. The older
`LocalDetailFamily`, `CspSpatialFamily` and `BodyContextFamily` remain separate
reference/context paths and their formula differences do not erase the new
implementations. None of the three versioned paths is thereby proven as a
universal default, and the calibrated F6a path lacks eligible native metadata.

For F5b, `CuedSequenceAssembler` now accepts only an explicitly named bout
with contiguous native sample indices from marked start to marked end. It
rejects gaps, overlaps, a wrong end marker, invalid samples and bouts shorter
than one second before constructing `CompleteSequenceBatch`. This makes
complete-bout DTW input auditable when a producer supplies trustworthy cue
boundaries. It does not infer biological onset/offset or establish streaming
recognition accuracy.

The F8 family-specific session summary now emits an actual per-channel Rest
noise log ratio when a Rest class is explicitly available, using median
absolute adjacent differences and equal trial mass. An isolated
[Song calibration-block audit](../benchmarks/song_real8/F8_REST_NOISE_SHIFT.json)
fits its long-term reference on S01/S02 formal trials and reads S03/S04's
separate 1/2-shot calibration blocks only. Direct trial-balanced recalculation
agrees to numerical precision; the shift norms are 0.7054/1.4173 for S03
and 1.1635/1.3283 for S04 at 1/2 shots. These same-day observations do not
establish a later-day domain shift or predictive benefit.

The [F0–F9 subsection coverage check](FORMULA_SUBSECTION_COVERAGE_AUDIT.csv)
now binds all 32 formula-bearing appendix sections to exact reviewed source
symbols; three additional headings are contextual. The formula inventory has
58 source-review rows. A two-dimensional independent F7 oracle verifies
Euclidean, robust-standardized and cosine coordinates, source-calibration
similarity scales and margins. Individual F9a–F9g rules are now separately
mapped, exposing where known-tone or measured-fault oracles remain missing.
This is source coverage and selected numerical evidence, not complete
scientific acceptance of every equation or native operating condition.

Independent F9 known-signal checks now recompute line-noise and low-frequency
ratios with direct Fourier sums rather than the implementation's FFT, and
recompute source-median/MAD amplitude and neighboring-channel correlation
anomalies on held-out signals. A separate direct trace-covariance calculation
also checks the global Frobenius anomaly. These numerical checks agree with
F9v2; physical-fault attribution remains unverified and the adverse Song
quality-gate decision is unchanged.

An independent [GRABMyo unseen-user F9 gate replay](../benchmarks/new_bank_v2/F9_GRAB_GATE_REPORT.md)
now tests the same source-frozen 0.5 trial rule with unknown ADC, mains and
pre-highpass metadata masked. Against unchanged F0v2 predictions, validation
coverage is 52/56; the four rejected trials were all correct, and none of
the eleven errors was caught. Descriptive final coverage falls to 40/56;
14 correct trials and two errors are rejected. This second public axis
reinforces the decision not to deploy the fixed quality gate. Coarse
frequency grids now return an unavailable line score without empty-bin
warnings or fabricated zero-as-measurement claims.

An exploratory [severe structural F9 replay](../benchmarks/new_bank_v2/F9_STRUCTURAL_GRAB_REPORT.md)
then separated persistent zero/flat channels and known-rail clipping from
soft amplitude/correlation shifts. It retains all 56 validation and all 56
descriptive-final GRAB trials, avoiding the parent rule's false rejections,
but catches none of the 14 existing F0 errors. Synthetic constant channels
and known-rail saturation are detected in tests. Both GRAB target groups had
already been inspected, and physical-fault labels are absent, so this rule
also remains diagnostic and is not a promoted Unknown gate.

The same severe rule was replayed on the available one-person, 250 Hz
[Song raw-ADC sessions](../benchmarks/song_real8/F9_STRUCTURAL_SONG_REPORT.md).
S03/S04 retain all 140/144 formal trials, so the parent mask's 4/29
rejections disappear; none of the 9/13 original F0 errors is detected.
Holding channel 1 constant in copied windows triggers the rule for every
trial. Exact parent source thresholds and frozen F0 trial identities are
checked. This is same-day, previously inspected, readiness-limited evidence,
not live fault sensitivity or a new-person validation.

A third [public re-wearing diagnostic](../benchmarks/new_bank_v2/F9_WEARING_STRUCTURAL_REPORT.md)
fits the same severe rule per source subject and reads four after-wearing
domains. All 120 validation and 120 descriptive-final native trials remain
unflagged across six users, while a copied constant-channel signal is flagged
in every trial. It catches none of the 49/45 frozen F0v2 errors. This bounds
false structural alerts under this selected electrode-shift data; without
measured physical-fault labels it does not validate live fault detection.

The [label-free session replay](../benchmarks/new_bank_v2/SESSION_UNLABELED_REPORT.md)
closes a deployment API gap: target evaluation labels are no longer required
to obtain personal session probabilities. On native public eight-channel,
200 Hz electrode-shift trials, 60 predictions from six branches and two
families exactly match the offline scoring wrapper. Deliberately wrong
offline truth labels do not change predictions, and source/session state is
immutable. Session states now carry their fitted source-profile identity;
cross-profile reuse is rejected, and the native result hash is stable across
two fresh-process replays. This is software parity, not an own-device session result.

An independent F4d numerical check now recomputes native eight-channel,
200 Hz log-band energy with direct Fourier sums. Unequal numbers of windows
per trial also check equal-trial-mass long/session references and immutable
held-out evaluation. This verifies the stated formula on a known signal; it
does not change the negative MANUS final minimum-user result or establish a
cross-day effect.

A [retrospective F2 three-axis decision](../benchmarks/new_bank_v2/F2_CROSS_AXIS_REVIEW_REPORT.md)
now applies the existing validation-only default-bank guard to F2a, F2b and
F2c on matched wearing, MANUS session and unseen-user GRAB trials. None avoids
both F1 loss and log-loss increase on every axis; F0v2 remains the public-data
default for this candidate set. The 24 saved cells and six source-result hashes
make the decision reproducible, but the reviewed public recordings are not a
new independent holdout.

The source-only personal reliability policy loader now verifies the fitted
family/classifier state hash as well as the OOF prediction hash before reusing
its selected temperature, shrinkage and population weights. A changed source
model can therefore no longer silently inherit an unrelated reliability
policy. The source-user CV evidence remains limited to the selected protocols.

The [historical uncentered F2a study](../benchmarks/new_bank_v2/F2A_DOCUMENT_REPORT.md)
compares an alternative second-moment candidate with the centered F2a arm.
Its earlier claim that uncentered F2a is the document's exact equation was
incorrect: the goal document explicitly subtracts channel time means. The
saved alternative's wearing gains, MANUS log-loss increase and unseen-user
GRAB decline retain their original experimental identity; F0v2 remains the
public-data default.

The [historical uncentered F2c/F3c extension](../benchmarks/new_bank_v2/DOCUMENT_SPATIAL_REPORT.md)
derives both alternative features from the same uncentered matrix. Its F2c
matched wearing/MANUS/GRAB evidence fails the cross-axis guard. Its F3c improves
both F1 and log loss on public wearing validation and descriptive-final users,
but the published data do not establish that saved channel columns follow the
Myo's physical circular order. Its result is conditional on that ordering
assumption and remains exploratory, not a default.

The [F3c channel-order control](../benchmarks/new_bank_v2/F3C_CHANNEL_ORDER_REPORT.md)
compares the saved order with 12 fixed non-equivalent reorderings on the same
public wearing split. Its strong native-order ranks show the cyclic-index
statistic is order-sensitive; they do not prove an anatomical column mapping.

The [public GRAB cross-day F8 control](../benchmarks/new_bank_v2/F8_GRAB_DAY_REPORT.md)
replays two frozen Day1 providers exactly and evaluates a previously fixed
cosine-agreement routing rule after four disjoint session-calibration trials
per subject/day. On 192 Day2 and 192 Day3 held-out trials, F8 changes no
gesture decisions. Log loss falls only 0.000219 and 0.000525. The result does
not support promoting the rule as a meaningful session recovery method; the
one-person Song Rest-noise descriptor remains calibration-only evidence.

The [document-exact session-normalization replay](../benchmarks/new_bank_v2/DOCUMENT_SESSION_REPORT.md)
now carries the specification's Q95+epsilon denominator through both the
long-term and session fits in a separate V2 pipeline. On six users and four
wearing domains, all 1,440 paired provider probabilities match the frozen
legacy route after feature conversion; the smallest calibration Q95 is 4.
This verifies integration and the absence of a numerical effect on those
inspected inputs, not general equivalence on quiet or faulty channels.

The [document F3c GRAB cross-day follow-up](../benchmarks/new_bank_v2/F3C_DOCUMENT_GRAB_REPORT.md)
adds a public ring1 electrode diagram and 448 saved trial-level candidate
probabilities to the topology audit. The visible 8–1–2 neighbors support only
part of the full circular numbering assumption. Against the same frozen F0,
the V2 uncentered F3c alternative lowers macro-F1 and raises log loss on both Day2
validation and descriptive Day3; all 448 decisions match the earlier centered
F3c arm despite changed probabilities. F3c remains a research candidate,
not a cross-day or own-device default.

The [centered V3 formula correction](../benchmarks/new_bank_v3/SPEC_SPATIAL_GRAB_REPORT.md)
now implements the goal file's actual F2a covariance and derives F2c/F3c
from that same matrix. Independent oracles verify centering, `T−1`, shrinkage,
`trace+epsilon`, source-only tangent reference, and circular-lag summaries.
On the frozen GRAB split, exact-formula F2a and F3c both reduce Day2/Day3
macro-F1 and increase log loss against F0. A separate
[V3 F2c cross-day increment](../benchmarks/new_bank_v3/SPEC_F2C_GRAB_REPORT.md)
improves Day2 F1/loss but reverses on descriptive Day3 (seven corrected F0
errors versus eleven new errors). Earlier uncentered V2 probabilities remain
archived as distinct alternative-feature results.

# Centered V3 F2c public-axis addendum (2026-10-03)

The [formula-correct centered F2c replay](../benchmarks/new_bank_v3/SPEC_F2C_CROSS_AXIS_REPORT.md)
uses matched held-out native trials for wearing shift, MANUS session transfer and
GRAB unseen-user transfer. It improves wearing validation F1 and log loss, but
MANUS validation log loss rises and GRAB unseen-user F1 and log loss worsen.
The frozen three-axis guard therefore retains F0v2 as default. The earlier
uncentered V2 F2c experiments remain separate alternative-feature evidence.
The 2026-10-03 numerical replay also aligns F2c's `1e-10` SPD ridge with its
eigenvalue floor. A zero-variance source window now maps to tangent zero;
all saved F2c probabilities were rerun without changing their frozen splits.
