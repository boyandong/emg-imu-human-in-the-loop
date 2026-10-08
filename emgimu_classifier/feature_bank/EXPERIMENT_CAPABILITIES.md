# Feature Bank experiment capabilities

Calibrated body-frame context now has an explicit API requiring >=1 second of
neutral IMU, a measured/guided forward-axis vector, real IMU rate/units and named
calibration trials. Missing calibration metadata is N/A for native body-frame
evaluation. Current EPN/MANUS reference IMU results are not re-labelled as calibrated
body-frame results. EMG and IMU window durations must match at their separate rates;
frame calibration trials cannot become held-out evaluation. No absolute yaw claim.
The [public F6 eligibility review](F6_PUBLIC_ELIGIBILITY.md) separates raw
six-axis signals and per-session anatomical-axis calibration from orientation,
gesture direction and approximate mounting descriptions. No reviewed public
candidate is yet verified as eligible for calibrated body-frame recognition.

DTW now requires explicit complete contiguous sequences with native duration
>=1 second and full-coverage metadata. Compressed-path sample rate and sparse
window count cannot establish eligibility. UniBo complete oracle-labelled bouts
have native-data replay evidence on Day6 and frozen Days7-8, with the latter
descriptive because those days were previously examined in other experiments;
the legacy MANUS eight-sparse-window DTW runner
is disabled for new runs. Its saved historical results remain proxy evidence,
not full-sequence or streaming validation.

## Current evidence

| Dataset | Force | User | Day or session | Wearing | Posture | Speed | Quality | Supported calibration |
|---|---|---|---|---|---|---|---|---|
| LibEMG Contraction Intensity | yes; target intensity and subjective levels | yes | two acquisition days stated, but filenames do not expose a trustworthy day field | no | no | no | synthetic only | personal; force zero-shot and product mode |
| LibEMG Electrode Shift | no | yes | before/after domains | yes | no | no | synthetic plus source QC | per-wearing-domain 0/1-shot; 2/5 unsupported (two trials/class/domain) |
| UniBo-INAIL | no | yes | 8 days | reapplication is confounded with day | 4 oracle labels | no | synthetic only | personal and cross-day session calibration |
| EMG-EPN612 | no | 612 users in the archive; confirmatory low-dimensional comparisons use explicit smaller cohorts | no validated repeated-session key | no | no | no | synthetic only | versioned personal 0/1/2/5-shot screens; separate fixed eight-coordinate 10/20-shot Mahalanobis versus Euclidean confirmation on users 32–41 |
| sEMG-MANUS | no | yes | up to 3 sessions | confounded with session | no controlled posture | slow/medium/fast | source anomaly flags plus synthetic | personal and session calibration |
| EMG-FMG | external grasped load, not voluntary force | yes | no repeated day | no | 8 limb positions | no | synthetic only | personal; load/posture product mode |
| Song real 8-channel EMG + 6-axis IMU | no measured force | one participant only | four sessions on one date; S01–S03 failed individual readiness, S04 passed individually, four-session cohort gate failed | no validated re-donning split or recorded re-donning attestation | seven cued arm states with raw IMU; 28-state test exploratory | no controlled speed | reconstructed protocol software path; original packet bytes and physiological cue onset unverified | distinct pre-formal 0/1/2-shot calibration evaluated offline; causal-filter gain is weak and not live deployment evidence |
| Public DS2 v9 candidate | publisher's three subjective force codes verified for all 2,863 raw trials; no mechanical force measurement | 20 subject folders; 2,833/2,863 raw trials exact-matched to TDMS | no repeated day | no validated re-donning | no controlled posture variation | no controlled speed | v9 MAV/gesture files byte-identical to v8 exact raw join; one mixed gesture-code block; 30 subject-unmatched trials | fixed cross-user force screen plus separately verified personal 0/1/2/5-shot low/average-to-high and all-force curves on active gestures; historical B0/X1-H/X2 identity unproven |
| GRABMyo F1–F8, eight monopolar forearm-ring columns | no | 8 selected subjects; separate cross-day same-user and Day1 held-out-user protocols | three distinct days; day 1 train, day 2 validation, day 3 final in the cross-day protocol | electrode reapplication may contribute but is not separately identified | no controlled posture | no controlled speed | 1,344 selected files match official SHA256SUMS; no physical-fault labels | four-class cross-day screen, same-user 0/1/2/5-shot calibration, and separate Day1 unseen-user F0/F9 diagnostics; no isolated re-donning or live-device validation |
| ROAM-EMG static recordings | no measured force | 28 public participants; source users 1–18, descriptive target users 19–28 | no validated cross-day split in the present experiment | not an isolated re-donning experiment | four native static postures | no controlled gesture-speed test | no annotated physical-fault evaluation | source-only three-class relax/open/close classifier on full chronological files; no pinch class or target calibration; fixed two-emission confirmation is a separate descriptive control |

The new-v2 GRABMyo calibration studies use labelled repetitions 1..N of the
*same target day* and reserve repetitions 6–7 for evaluation. Their score-space
and feature-space methods have different held-out behavior. The source-only
Day1→Day2/3 screen remains a separate zero-target-calibration result; none of
these studies is an independent held-out-user benchmark or a physical
re-donning experiment.

The collection UI now records local click-to-calibration-outcome wall time separately
from the requested signal duration, including interrupted attempts. No real-device
wall-time observations have been collected for the research tables yet; electrode
placement and device setup are outside this measurement.

The existing four Song HDF5 recordings also contain host monotonic events and EMG
packet-reception timestamps. Their [clock audit](../benchmarks/song_real8/CALIBRATION_CLOCK_AUDIT.json)
measures about 64 seconds from first to last guided calibration block in each
session, with about 32 seconds through the hand-state sequence. This is recorded
in-protocol time only, not the separate recognition UI's click-to-outcome time
or electrode preparation time.

Secondary datasets are activated only when a Tier 1 capability gap remains. GREAT can confirm
posture by day, NinaPro DB6 remains a cross-day candidate, and the selected GRABMyo subset now provides one public cross-day confirmation. The three-position
dataset can confirm electrode replacement. Hyser remains a measured-force and observability
ceiling using selected subsets.

## Latest versioned evidence and sensor compatibility

The [current delivery index](delivery/INDEX.json) links distinct trial-level tables,
nominal-sample continuous tables and the current scientific conclusions. Their
denominators, class ontologies and calibration budgets are not interchangeable.

The [ROAM chronological control](../benchmarks/new_bank_v3/ROAM_CAUSAL_WINDOW_V1_RESULTS.json)
uses native eight-channel, nominal 200 Hz recordings with 40-sample windows and
10-sample updates. The first 39 samples in each recording remain unknown.
The [fixed label-confirmation control](../benchmarks/new_bank_v3/ROAM_DEBOUNCE_CONTROL_V1_RESULTS.json)
needs an additional consistent emission before initialization and reports the
extra unknown samples rather than treating them as Neutral. Offline sample-grid
results do not measure packet arrival, physical onset or device latency.

The [EPN calibration-burden extraction](../benchmarks/new_bank_v3/MAHALANOBIS_EPN_HOLDOUT_V2_BURDEN.json)
reports 60/120 independent calibration trials for six classes at 10/20 shots.
Extracted signal exposure is 48/96 seconds; the complete stored recordings
and physical session wall time are distinct. These calibrated-method comparisons
do not establish recovery versus no personal anchor, all-user benefit or the
required budget for an unseen device/session.

The [GRAB paper acquisition review](../benchmarks/discovery/GRAB_PRIMARY_TOPOLOGY_V1.json)
and [native channel census](../benchmarks/discovery/GRAB_CHANNEL_CENSUS_V1.json)
distinguish 28 recorded monopolar electrodes from the 32 stored columns.
The selected F1–F8 view contains eight signals from one ring. Bipolar pairing
across rings is a separately constructed montage; neither the eight-channel
count nor paper placement confirms our device wiring, pair sign or direction.
The original native headers and mV conversion remain authoritative.

The [official UniBo original-MAT smoke check](OFFICIAL_UNIBO_ADAPTER_ACCEPTANCE_V1.json)
now executes the previously optional conversion test on one archive-bound,
unchanged source file. It verifies adapter conversion and benchmark schema,
without training or extending a four-muscle public dataset into an eight-channel
physical compatibility claim. It complements the separately recorded full suite.

## Frozen rules

- Split subject, session/domain and trial before windowing. Overlapping windows never cross a split.
- Scalers, thresholds, PCA, CSP, covariance references, probability calibration, prototypes,
  templates and fusion weights fit only on training or the explicitly permitted calibration subset.
- Calibration trials are removed completely from evaluation.
- Force-ZeroShot uses calibration from source force only. Force-ProductMode is a separate result.
- Public posture labels are oracle context. Missing IMU, posture, force, wearing or speed is `N/A`.
- Dataset-native labels are preserved; strict common labels use the ontology in the discovery folder.
- Quality corruptions are labelled synthetic and are never claimed to reproduce real hardware failure.

## Existing baseline state

UniBo reference results and the 33-run G0-G6 physiology matrix are frozen in their dated reports.
They are not overwritten by Feature Bank runs. The user explicitly superseded recovery of
unavailable older experiments. The independently versioned F0–F9 implementations, frozen
protocols and results are the current forward-work deliverables; earlier algorithm names
do not establish implementation equivalence and are not prerequisites for further work.
