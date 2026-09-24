# Feature Bank experiment capabilities

Calibrated body-frame context now has an explicit API requiring >=1 second of
neutral IMU, a measured/guided forward-axis vector, real IMU rate/units and named
calibration trials. Missing calibration metadata is N/A for native body-frame
evaluation. Current EPN/MANUS reference IMU results are not re-labelled as calibrated
body-frame results. EMG and IMU window durations must match at their separate rates;
frame calibration trials cannot become held-out evaluation. No absolute yaw claim.

DTW now requires explicit complete contiguous sequences with native duration
>=1 second and full-coverage metadata. Compressed-path sample rate and sparse
window count cannot establish eligibility. UniBo complete oracle-labelled bouts
have native-data replay evidence; the legacy MANUS eight-sparse-window DTW runner
is disabled for new runs. Its saved historical results remain proxy evidence,
not full-sequence or streaming validation.

## Current evidence

| Dataset | Force | User | Day or session | Wearing | Posture | Speed | Quality | Supported calibration |
|---|---|---|---|---|---|---|---|---|
| LibEMG Contraction Intensity | yes; target intensity and subjective levels | yes | two acquisition days stated, but filenames do not expose a trustworthy day field | no | no | no | synthetic only | personal; force zero-shot and product mode |
| LibEMG Electrode Shift | no | yes | before/after domains | yes | no | no | synthetic plus source QC | per-wearing-domain 0/1-shot; 2/5 unsupported (two trials/class/domain) |
| UniBo-INAIL | no | yes | 8 days | reapplication is confounded with day | 4 oracle labels | no | synthetic only | personal and cross-day session calibration |
| EMG-EPN612 | no | 612 users | no validated repeated-session key | no | no | no | synthetic only | personal 0/1/2/5-shot |
| sEMG-MANUS | no | yes | up to 3 sessions | confounded with session | no controlled posture | slow/medium/fast | source anomaly flags plus synthetic | personal and session calibration |
| EMG-FMG | external grasped load, not voluntary force | yes | no repeated day | no | 8 limb positions | no | synthetic only | personal; load/posture product mode |
| Song real 8-channel EMG + 6-axis IMU | no measured force | one participant only | four sessions on one date; S01–S03 failed readiness, S04 passed | no validated re-donning split | seven cued arm states with raw IMU; 28-state test exploratory | no controlled speed | reconstructed protocol software path; original packet bytes and physiological cue onset unverified | distinct pre-formal 0/1/2-shot calibration evaluated offline; causal-filter gain is weak and not live deployment evidence |
| Public DS2 v8 candidate | three subjective force levels in protocol, but no verified per-trial force labels | 20 subject folders; 2,833/2,863 raw trials exact-matched to TDMS | no repeated day | no validated re-donning | no controlled posture variation | no controlled speed | raw-to-MAV window order exact; one mixed gesture-code block; 30 subject-unmatched trials | gesture-code-only held-out subject split available on 2,832 trials; no historical force result or personal-force calibration claim |
| GRABMyo selected F1–F8 forearm ring | no | 8 selected subjects, not held-out-user test | three distinct days; day 1 train, day 2 validation, day 3 final | electrode reapplication may contribute but is not separately identified | no controlled posture | no controlled speed | 1,344 selected files match official SHA256SUMS | fixed pooled model; no personal or session calibration; four-class trial-level cross-day screen only |

The collection UI now records local click-to-calibration-outcome wall time separately
from the requested signal duration, including interrupted attempts. No real-device
wall-time observations have been collected for the research tables yet; electrode
placement and device setup are outside this measurement.

Secondary datasets are activated only when a Tier 1 capability gap remains. GREAT can confirm
posture by day, NinaPro DB6 remains a cross-day candidate, and the selected GRABMyo subset now provides one public cross-day confirmation. The three-position
dataset can confirm electrode replacement. Hyser remains a measured-force and observability
ceiling using selected subsets.

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
They are not overwritten by Feature Bank runs. The repository contains no executable historical
DS2, X1-H, RLCS, CES or Frequency experiment artifacts; their claims remain prior evidence from
the supplied brief until the exact source artifacts are recovered or independently reproduced.
