# New-v2 observed-axis robustness envelope

The [frozen protocol](ENVELOPE_PROTOCOL.json) defines a descriptive
seven-axis vector for each of the four arms. Force, wearing and day have
matched new-v2 public comparisons; independent user, posture, speed and
real quality remain `N/A`. Song's same-day sessions are a one-person
supplement, not a fourth independent axis. The eight-row
[table](ROBUSTNESS_ENVELOPE.csv) gives each observed axis's pooled native
trial macro-F1, the equal-axis unweighted `mean_R_available`, and `R_min`,
the smallest of those three axis scores. It separately reports the minimum
observed subject/condition macro-F1 cell; this is not the same quantity as
`R_min`. The [audit](ENVELOPE_AUDIT.json) binds the parent family-screen
table, and `envelope_analysis.py --verify` regenerates the output byte for
byte.

| Phase | Arm | Mean of 3 available axes | R_min | R_min axis |
| --- | --- | ---: | ---: | --- |
| Validation | F0v2 | 0.7207 | **0.5775** | Wearing |
| Validation | F0v2+F2a | 0.7307 | 0.5651 | Force |
| Validation | F0v2+F3c | 0.7229 | 0.5649 | Force |
| Validation | F0v2+F2a+F3c | **0.7424** | 0.5671 | Force |
| Final, descriptive | F0v2 | 0.6706 | **0.4939** | Force |
| Final, descriptive | F0v2+F2a | 0.6630 | 0.5357 | Force |
| Final, descriptive | F0v2+F3c | **0.7068** | 0.4703 | Force |
| Final, descriptive | F0v2+F2a+F3c | 0.6641 | 0.4797 | Force |

An apparent mean gain can conceal a weaker worst axis. For example,
the full arm has the highest validation mean but its validation force
score, 0.5671, is below F0v2's worst-axis score, 0.5775. On descriptive
final data, F3c has the highest mean yet lowers the worst axis to 0.4703.
F0v2+F2a has the highest descriptive final minimum, 0.5357, but is a
final-only finding and must not be promoted using these inspected users.
None of these arms is shown to lift the complete seven-axis envelope.

The later [MANUS reduced-bank study](MANUS_SPATIAL_REPORT.md) gives a
speed-stratified public eight-channel cross-session check of F2a/F3c, but
MANUS has no Rest and cannot fit F0v2. It therefore does not replace the
`N/A` speed cell of this exact four-arm envelope.

A later [versioned four-axis envelope](ENVELOPE_USER_REPORT.md) separately
adds a same-day, subject-disjoint GRABMyo user-axis result for the exact
F0v2/F2a/F3c candidates. This three-axis frozen table is retained for
comparison.

This index averages three *different classification tasks* with different
gestures, devices, users and sample rates. It is an explicit, transparent
summary requested by the protocol, not a pooled accuracy estimate or a
statistical comparison of interchangeable populations. Force also changes
users, so the user axis is not separately identified. The within-axis
subject/condition minimum can be based on much smaller groups and is
reported separately in the CSV without treating it as a confidence bound.
