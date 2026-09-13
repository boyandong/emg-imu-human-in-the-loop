# Myo hardware-flexible neural representation screen (2026-09-13)

## Research question

This screen asks whether the hardware-flexible learned representations from HLA
Round 3 add useful held-out-user information on the 8-channel Myo view. It is a
screening experiment, not the final Myo benchmark and not evidence of
cross-dataset portability.

The primary comparisons are:

- `R0`: per-channel G0 feature tokens;
- `R1-core`: G0 plus G5 temporal-recruitment tokens;
- `R2`: learned multi-rate raw-signal tokens;
- `R3`: fusion of R1-core and R2;
- `R2-wide`: a raw-signal capacity control with approximately the R3 parameter
  count.

## Protocol and provenance

- Dataset view: `myo_ring_8`, 8 channels, 200 Hz native sampling.
- Window protocol: 250 ms windows and 50 ms hop, generated inside physical
  trials only.
- Held-out targets: `female00`, `female01`, `male00`, `male05`, `male10`, and
  `male15`.
- For every fold, the target subject is absent from training and validation. One
  source subject is used for validation and the other 16 source subjects are
  used for training.
- Each physical trial contributes 10 uniformly selected windows. A typical fold
  contains 13,440 training, 840 validation, and 840 evaluation windows.
- Training: at most 30 epochs, patience 5, batch size 128.
- Single-seed representation screen: seed 42 for all five representations.
- Confirmation screen: seeds 42, 43, and 44 for R1-core and R3.
- Run artifacts report source commit
  `87a66f6fb3dd310b922f64f7701dfa352bab4371` and a clean worktree.
- The multi-seed aggregator verified that R1-core and R3 use identical
  evaluation examples before making paired comparisons.

The six targets and 10-window-per-trial cap were declared as a compute screen.
They must not be described as an 18-subject full-data result.

## Stage 1: seed-42 representation screen

| Representation | Parameters | Active macro-F1 | Difference vs R0 | Subjects improved vs R0 | Subject-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| R0 | 27,331 | 0.6765 | -- | -- | -- |
| R1-core | 27,651 | 0.7187 | +0.0423 | 6/6 | [+0.0095, +0.0796] |
| R2 | 48,451 | 0.6662 | -0.0102 | 3/6 | [-0.0643, +0.0384] |
| R3 | 75,812 | 0.7262 | +0.0497 | 3/6 | [-0.0112, +0.1113] |
| R2-wide | 78,787 | 0.6842 | +0.0077 | 5/6 | [-0.1201, +0.0930] |

The seed-42 evidence supports carrying R1-core and R3 forward. It does not
support the claim that a raw learned branch is automatically better than the
handcrafted representation: R2 is below R0, and adding comparable parameter
capacity in R2-wide does not reproduce the R3 mean. This makes a pure parameter
count explanation for R3 less plausible, while not proving that fusion is the
cause.

## Stage 2: three-seed paired confirmation

The statistical unit is the subject. Each subject is first averaged over the
three seeds, and only then used in the paired bootstrap.

| Representation | Mean active macro-F1 |
|---|---:|
| R1-core | 0.7128 |
| R3 | 0.7387 |
| Paired R3 - R1-core | +0.0260 |

R3 improves 4 of 6 subjects. The paired subject-level median difference is
+0.0340 and the 2,000-resample 95% bootstrap interval is
[-0.0244, +0.0683]. The interval crosses zero. Therefore the current result is a
positive mean trend, not stable evidence that R3 is superior for the Myo
population.

The variation is important. R3 changes subject-level active macro-F1 relative to
R1-core by approximately +0.0964, -0.0795, -0.0024, +0.0211, +0.0469, and
+0.0732 across the six targets. A better average can coexist with a substantial
failure for one wearer.

## Per-class interpretation

| Class | R1-core recall | R3 recall | R3 - R1-core |
|---|---:|---:|---:|
| Neutral | 0.9315 | 0.9343 | +0.0028 |
| Radial deviation | 0.8866 | 0.8940 | +0.0074 |
| Wrist flexion | 0.7130 | 0.7431 | +0.0301 |
| Ulnar deviation | 0.7097 | 0.7315 | +0.0218 |
| Wrist extension | 0.8282 | 0.7921 | -0.0361 |
| Hand close | 0.7458 | 0.7833 | +0.0375 |
| Hand open | 0.5231 | 0.6069 | +0.0838 |

The largest mean benefit appears on hand open, which is also the weakest class
under R1-core. R3 also helps hand close and wrist flexion, but loses wrist
extension recall. This pattern is compatible with the raw temporal branch adding
some information for hand-state dynamics, but the experiment does not isolate
that mechanism. Ulnar deviation, hand close, and hand open also have large
between-subject variability, so class means should not be treated as universal
wearer behavior.

## Decision and limitations

- Keep R1-core as the credible lighter candidate: it has nearly the same scale
  as R0 and its seed-42 paired gain is consistent across all six targets.
- Keep R3 as the second candidate because its three-seed mean is 0.0260 above
  R1-core and it notably improves mean hand-open recall.
- Do not claim that R3 has passed the Round 3 retention gate. The plan requires
  at least a 0.01 gain on at least two datasets; only Myo has been screened, and
  even its six-subject interval crosses zero.
- Do not advance R2 or R2-wide as standalone candidates from this screen.
- Do not begin personal-adapter model selection until the corresponding GRABMyo
  forearm and wrist screens establish whether the pattern transfers across
  hardware views.
- Repeat the decisive comparison with more held-out subjects and without the
  10-window cap before calling it the final Myo specialist result.

Complete model, prediction, metric, confusion, calibration, and risk-coverage
artifacts are stored outside Git under
`EMG-project-hla-data/runs/myo-hla-screen-s42-87a66f6`,
`myo-hla-screen-s43-87a66f6`, and `myo-hla-screen-s44-87a66f6`. The aggregated
paired result is `myo-hla-r1core-vs-r3-3seed.json` in the same external run root.
