# Feature Bank + Personal Calibration — interim evidence

This is an incomplete execution of the two supplied specifications, not a completion claim.
All reported evaluations use held-out subjects or sessions. Final families and calibration
hyperparameters were selected on validation data; final data was not used to tune them.
Seed: 20260915. Classical logistic regression runs use CPU; no neural training is needed
for these representation comparisons.

## Independent test results

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

Historical DS2 is blocked: no exact historical artifacts were found and the original Kaggle
release returned 404. NinaPro requires access; secondary datasets remain explicitly deferred.
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
