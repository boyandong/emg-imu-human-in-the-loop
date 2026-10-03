# F7 exact affine-SPD personal-anchor candidate

The opt-in candidate implements the goal file's affine-invariant
`||log(C1^-1/2 C2 C1^-1/2)||_F` distance. Its class prototype is the
arithmetic mean of **one centered F2a covariance per calibration trial**;
unequal window counts cannot give a trial extra weight. A fixed 1e-10 SPD
ridge makes zero-variance windows valid. It does not replace the frozen
source-fitted tangent-space candidate.

An independent 2x2 oracle checks a diagonal case, a non-orthogonal affine
change of basis, an extreme non-floored eigenvalue, equal trial mass,
held-out state immutability and calibration/evaluation disjointness.

The [saved native EPN612 screen](F7_AFFINE_EPN/results.json) uses the public
200 Hz, eight-channel training-sample trials of users 16–18 for validation
and previously inspected users 19–21 for descriptive final. Per user and
class, a fixed seed selects nested 1/2/5 complete trial budgets; all other
trials are evaluated. Softmax temperature is the median distance between
class prototypes from calibration **only**. No population training or deep
model is run, and no zero-shot personal prototype exists.

| Phase | 1-shot F1 / loss | 2-shot F1 / loss | 5-shot F1 / loss |
|---|---:|---:|---:|
| Validation, three users | 0.3687 / 1.6728 | 0.4388 / 1.6119 | 0.5142 / 1.5714 |
| Descriptive final, three users | 0.3007 / 1.7080 | 0.3371 / 1.6618 | 0.4078 / 1.6430 |

This proves the formula is executable on eligible native trials, not that
it improves the feature bank. The selected trials differ from the saved
tangent-anchor study; direct F1/loss subtraction would be invalid. A matched
Core-versus-Core+F7 incremental comparison would need frozen common trial
identities, source-only probability calibration and a selection rule fixed
before an independent holdout. There is no own-device claim or default change.
