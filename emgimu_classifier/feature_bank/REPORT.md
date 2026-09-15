# Feature Bank + Personal Calibration — interim evidence

This is an incomplete execution of the two supplied specifications, not a completion claim.
Predictive evaluations use source OOF or held-out subjects/sessions; descriptive target
diagnostics are explicitly labelled. Model compositions and calibration rules are frozen
from source evidence, validation evidence or prespecified controls; final scores do not tune them.
Seed: 20260915. Classical logistic regression runs use CPU; no neural training is needed
for these representation comparisons.

## Current interpretation of the evidence

The full bank has local gains and failures. It does not improve every task or the minimum
of the complete available robustness vector. The following seven-axis table describes
fixed full-bank algorithms at cal0; it does not substitute the best observed specialist.
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
| C: Which families are specialists? | Ring helps average wearing/force performance in some compositions; spectral helps some load/force cells; temporal helps the native UniBo shortlist. | All force Core additions reduce Core's condition minimum. Some FMG position minima also fall. Historical RLCS/CES/Frequency equivalence is unverified. |
| D: Which need personal calibration? | MANUS session models can recover strongly with a small own-session budget; current EPN anchors can harm performance. Ramp-only force Core anchors recover only a small amount. | No universal anchor benefit or device calibration prescription follows. |
| E: Does Personal Anchor reduce cross-user variation? | No for the tested EPN branch: all nonzero budgets lower observed mean/minimum F1 and raise standard deviation relative to matched no-anchor controls. | Three final users, descriptive variation; this does not reject every anchor design. |
| F: Does Session Signature help cross-day/re-donning? | MANUS session profiles measure shifts. Matched current-session prototypes outperform the fixed long/current blend in the tested controls. | A measured signature shift is not proof of predictive value; matched re-donning and broader cross-day signature tests remain open. |
| G: Does the bank improve R_min? | Not for the complete seven-axis vector. Frozen concatenated force Core improves its own force-condition minimum; adding families can raise average F1 while lowering that minimum. | No global robustness recovery; unlike/correlated tasks and synthetic-quality scope remain explicit. |
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
burden estimates and SVG curves are available. The latest suite ran 141 tests with one
skip; consolidated integrity covers 156 artifacts, 34,415 rows and 699 explicit
partitions, while canonical record verification covers 32,932 rows. These are narrow
integrity/implementation checks, not proof that every scientific requirement is complete.

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
