# Failure benchmark matrix

| Failure | Historical | Primary | Secondary or interaction | Evidence boundary |
|---|---|---|---|---|
| Force | DS2 subjective three-level effort | LibEMG Contraction Intensity | EMG-FMG external load; future Hyser force subset | External load, subjective effort and measured force are separate constructs |
| Wearing and electrode shift | none located | LibEMG Electrode Shift | Three-position replacement; FORS-EMG placement regions; own controlled re-donning | Cross-day drift cannot by itself identify electrode displacement |
| Cross-user | none | EMG-EPN612 | GRABMyo; NinaPro DB5 | User-independent accuracy does not prove session robustness |
| Cross-day and session | UniBo-INAIL | UniBo-INAIL | NinaPro DB6 (deferred); GRABMyo selected 8-subject F1–F8 subset measured; GREAT deferred | UniBo posture labels are oracle context and contain no IMU; GRABMyo uses different hardware and a limited four-class selected cohort |
| Posture and limb position | UniBo-INAIL | UniBo-INAIL | GREAT; EMG-FMG interaction | Orientation labels are not interchangeable with direction of motion |
| Execution speed | none | sEMG-MANUS | none selected | Speed is separate from force and signal duration must use actual rows |
| Quality | clean source data | synthetic corruption on every supported benchmark | real acquisition quality flags where available | Synthetic dropout, clipping and noise are labelled simulations |
| Multi-factor interaction | none | UniBo day by posture by subject; sEMG-MANUS speed by session by subject | EMG-FMG load by posture by subject | Aggregate scores must retain factor-specific cells |

Every future feature family reports a robustness vector over these cells. Missing real
annotations are reported as `N/A`, not estimated from filenames or synthesized labels.
