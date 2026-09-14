# GRABMyo forearm hardware-flexible representation screen (2026-09-14)

## Research question

This Round 3 screen asks whether the HLA hardware-flexible representations retain
their value when moving from the 8-channel, 200 Hz Myo view to the 16-channel,
2048 Hz GRABMyo forearm view. It is a held-out-subject zero-shot screen. It is not
the final GRABMyo benchmark and it does not evaluate personalization.

The five preregistered representations are:

- `R0`: per-channel G0 feature tokens;
- `R1-core`: G0 plus G5 temporal-recruitment tokens;
- `R2`: learned multi-rate raw-signal tokens;
- `R3`: fusion of R1-core and R2;
- `R2-wide`: a raw-signal capacity control with approximately the R3 parameter
  count.

## Protocol and provenance

- Dataset view: `grabmyo_forearm_16`, 16 forearm channels, 2048 Hz native
  sampling, 17 dataset-specific classes including rest.
- Window protocol: 250 ms windows and 50 ms hop, generated inside physical trials.
- Held-out targets: `p01`, `p08`, `p15`, `p22`, `p29`, `p36`, and `p43`.
- The target subject is absent from training and validation in every fold. One
  source subject is used for validation and the remaining source subjects for
  training.
- Each physical trial contributes 10 uniformly selected windows. A typical fold
  contains 146,370 training, 3,570 validation, and 3,570 evaluation windows.
- Training: seed 42, at most 30 epochs, patience 5, and batch size 128.
- All 35 expected runs are complete and report source commit
  `da7cbc5aa03145a94552141dc4f78fceee0a2aea` with a clean worktree.
- The same pre-run test log is attached to every run. It records 116 passed and
  1 skipped tests.
- The official GRABMyo SHA-256 check and adapted-dataset integrity check passed.
  The result archive copied to the local machine has SHA-256
  `11a87c8a50977ba5d8e196695729dfabe5dd181e5d2674f7ffff197f754c6e3c`.

The seven targets, one seed, and 10-window-per-trial cap define a compute screen.
Windows are not independent statistical replicates; comparisons below use the
seven held-out subjects as the paired units.

## Main result

| Representation | Parameters | Accuracy | Macro-F1 | Active macro-F1 | Difference vs R0 |
|---|---:|---:|---:|---:|---:|
| R0 | 27,331 | 0.5377 | 0.5081 | 0.4888 | -- |
| R1-core | 27,651 | 0.5088 | 0.4765 | 0.4561 | -0.0327 |
| R2 | 48,451 | 0.5464 | 0.5177 | 0.4993 | +0.0105 |
| R3 | 75,812 | 0.5607 | 0.5316 | 0.5140 | +0.0252 |
| R2-wide | 78,787 | 0.5531 | 0.5235 | 0.5061 | +0.0172 |

R3 has the highest mean accuracy, macro-F1, and active macro-F1. Relative to R0,
its paired active macro-F1 gain is +0.0252; 4 of 7 subjects improve and the
20,000-resample subject bootstrap interval is [+0.0032, +0.0535]. This establishes
a positive result for this fixed screen, not a population-level or multi-seed
conclusion.

The more decisive Round 3 comparison is against the better component, R2. R3 is
+0.0147 above R2 and improves 6 of 7 subjects, but its interval is
[-0.0064, +0.0408]. R3 is only +0.0079 above the parameter-matched R2-wide control,
with an interval of [-0.0193, +0.0393]. The screen therefore cannot separate a
fusion benefit from model-capacity and seed variation with confidence.

R1-core is -0.0327 below R0, improves only 1 of 7 subjects, and has a paired
interval of [-0.0623, -0.0083]. The G5 temporal-recruitment features that helped
the Myo screen do not transfer automatically to this sensor view and protocol.

## Class-level interpretation

R3 is strongest on broad wrist and forearm actions and rest:

| Class | Mean recall | Mean F1 |
|---|---:|---:|
| Wrist extension | 0.9503 | 0.9304 |
| Wrist flexion | 0.9020 | 0.8959 |
| Rest | 0.9313 | 0.8139 |
| Forearm supination | 0.8605 | 0.7809 |
| Hand open | 0.6204 | 0.6128 |

It remains weak on fine or overlapping muscle-control patterns:

| Class | Mean recall | Mean F1 |
|---|---:|---:|
| Lateral prehension | 0.2245 | 0.2275 |
| Index-finger extension | 0.1769 | 0.2325 |
| Thumb adduction | 0.2088 | 0.2812 |
| Thumb-index opposition | 0.3456 | 0.3309 |
| Hand close | 0.3544 | 0.3562 |

The largest aggregated confusions are anatomically or activation-pattern related:
lateral prehension is predicted as hand close for 28.2% of its windows; hand close
is predicted as lateral prehension for 24.5%; little-finger extension is predicted
as thumb-little extension for 24.2%; and index-finger extension is predicted as
thumb-index extension for 23.0%. These errors are compatible with similar muscles
and neighboring movement patterns producing overlapping surface EMG, but this
experiment does not measure muscle anatomy directly and therefore cannot prove
that mechanism.

Relative to R0, R3's largest mean class-F1 gains are thumb-index extension
(+0.1006), forearm pronation (+0.0929), thumb adduction (+0.0850), thumb-little
opposition (+0.0690), and hand open (+0.0459). It loses on thumb-little extension
(-0.0446), little-finger extension (-0.0192), and hand close (-0.0164). The gain is
therefore not uniform across gestures.

## Subject and uncertainty observations

R3 active macro-F1 ranges from 0.3990 for `p29` to 0.6832 for `p36`. The
between-subject standard deviation is 0.0944, much larger than the mean R3-R2
gain of 0.0147. This means wearer variation remains the dominant practical
problem even when the average representation improves.

R3's mean confidence is 0.695. Correct predictions average 0.787 confidence and
wrong predictions 0.578, so confidence contains useful ranking information. At
approximately the top 25% most-confident windows, mean risk is 0.12, compared with
0.44 when all windows are accepted. However, zero-shot calibration curves are
marked not applicable, so these confidence values must not be interpreted as
calibrated probabilities.

## Decision

- Keep R3 as the leading GRABMyo forearm candidate, but do not call it the final
  winner until seeds 43 and 44 are run on the same targets.
- Keep R2 as the lighter learned baseline and R2-wide as the capacity control.
- Do not treat G5/R1-core as a hardware-universal improvement. Investigate why it
  helps Myo but hurts GRABMyo before using it as a default branch.
- Do not claim that R3 has passed the Round 3 cross-dataset retention gate. The
  GRABMyo forearm mean clears the +0.01 numerical threshold over R2, but its
  interval crosses zero, only one seed has been run, and the GRABMyo wrist view is
  incomplete.
- Run the preregistered wrist screen from the beginning. The interrupted wrist
  output contains only an empty `R0_p01_s42` directory and no valid metrics or
  checkpoint.
- If the wrist pattern agrees, run seeds 43 and 44 only for the final two
  candidates rather than repeating the full five-model matrix.

Complete forearm models, predictions, metrics, confusion matrices, manifests,
data-integrity reports, and logs are stored outside Git under
`EMG-project-hla-data/downloads/grabmyo-hla-results-da7cbc5`. The wrist artifacts
in that package are incomplete and must not be used as results.
