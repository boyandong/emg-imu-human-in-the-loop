# Four observed-axis new-v2 robustness envelope

The [versioned protocol](ENVELOPE_USER_PROTOCOL.json) binds the frozen
[original three-axis screen](FAMILY_SCREEN.csv) and the new
[same-day subject-disjoint user screen](GRAB_USER_SCREEN.csv). The
[reproducer](envelope_user_analysis.py) writes eight phase–arm rows to
[ROBUSTNESS_ENVELOPE_4AXIS.csv](ROBUSTNESS_ENVELOPE_4AXIS.csv); its
[audit](ENVELOPE_USER_AUDIT.json) records both source hashes and can be
verified byte for byte. The original [three-axis envelope](ENVELOPE_REPORT.md)
remains unchanged.

| Phase | Arm | User F1 | Equal four-axis mean | Minimum axis F1 | Minimum axis |
| --- | --- | ---: | ---: | ---: | --- |
| Validation | F0v2 | **0.8054** | **0.7419** | **0.5775** | Wearing |
| Validation | F0v2+F2a | 0.7485 | 0.7352 | 0.5651 | Force |
| Validation | F0v2+F3c | 0.7350 | 0.7259 | 0.5649 | Force |
| Validation | F0v2+F2a+F3c | 0.7256 | 0.7382 | 0.5671 | Force |
| Final, descriptive | F0v2 | 0.9458 | 0.7394 | 0.4939 | Force |
| Final, descriptive | F0v2+F2a | 0.8899 | 0.7197 | **0.5357** | Force |
| Final, descriptive | F0v2+F3c | **0.9462** | **0.7667** | 0.4703 | Force |
| Final, descriptive | F0v2+F2a+F3c | 0.8899 | 0.7206 | 0.4797 | Force |

Across the four available axes, F0v2 has the strongest validation
equal-axis mean and minimum. Its descriptive final minimum is not the
largest, but a final-only ranking cannot choose a feature arm. The
observed minimum subject/condition cell is separately preserved in the
table and must not be confused with the minimum *axis* score.

Day and user use different splits of the **same GRABMyo subset**, so they
are correlated pieces of evidence rather than independent populations.
The force axis also changes subjects as well as intensity. The four axes
have different devices, sample rates, users and gesture labels, so their
equal-weight mean is an explicit descriptive index, not pooled accuracy.
Posture, speed and real signal quality remain `N/A` for this exact
F0v2/F2a/F3c bank; the separate MANUS speed-stratified study uses a
reduced bank because MANUS has no Rest class. No complete seven-axis
robustness claim or own-device deployment decision follows.
