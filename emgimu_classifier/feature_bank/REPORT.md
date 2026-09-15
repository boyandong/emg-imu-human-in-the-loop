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
E. Predictive gains do not yet prove reduced cross-user variation in anchor coordinates;
that specific representation-distance comparison remains missing.
F. F8 alone is unstable; the combined MANUS system shows one-shot context contribution.
Re-donning is confounded with session/day, so an isolated physical re-donning claim is unproven.
G. Individual worst-condition improvements exist (for example UniBo posture 2), but the complete
system's seven-coordinate mean and minimum envelope comparison is still incomplete.
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
