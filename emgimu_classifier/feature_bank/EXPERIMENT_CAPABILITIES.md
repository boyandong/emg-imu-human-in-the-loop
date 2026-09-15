# Feature Bank experiment capabilities

## Current evidence

| Dataset | Force | User | Day or session | Wearing | Posture | Speed | Quality | Supported calibration |
|---|---|---|---|---|---|---|---|---|
| LibEMG Contraction Intensity | yes; target intensity and subjective levels | yes | two acquisition days stated, but filenames do not expose a trustworthy day field | no | no | no | synthetic only | personal; force zero-shot and product mode |
| LibEMG Electrode Shift | no | yes | before/after domains | yes | no | no | synthetic plus source QC | per-wearing-domain 0/1-shot; 2/5 unsupported (two trials/class/domain) |
| UniBo-INAIL | no | yes | 8 days | reapplication is confounded with day | 4 oracle labels | no | synthetic only | personal and cross-day session calibration |
| EMG-EPN612 | no | 612 users | no validated repeated-session key | no | no | no | synthetic only | personal 0/1/2/5-shot |
| sEMG-MANUS | no | yes | up to 3 sessions | confounded with session | no controlled posture | slow/medium/fast | source anomaly flags plus synthetic | personal and session calibration |
| EMG-FMG | external grasped load, not voluntary force | yes | no repeated day | no | 8 limb positions | no | synthetic only | personal; load/posture product mode |
| Own 8-channel data | only if explicitly collected | yes if multiple people exist | only if completed sessions pass readiness | only controlled re-donning trials | IMU exists | only if cued and labelled | real packet/channel audit plus synthetic | personal and session when data gates pass |

Secondary datasets are activated only when a Tier 1 capability gap remains. GREAT can confirm
posture by day, NinaPro DB6 and GRABMyo can confirm cross-day effects, and the three-position
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
