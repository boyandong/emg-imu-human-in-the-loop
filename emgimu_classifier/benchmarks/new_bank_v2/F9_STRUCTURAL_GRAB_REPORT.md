# Exploratory structural F9 rule on unseen GRABMyo users

The [frozen rule](F9_STRUCTURAL_GRAB_PROTOCOL.json) separates severe signal
structure failures from the parent mask's soft amplitude/correlation shift
scores. A native trial is marked invalid only if any channel in any of its
twenty windows is zero for at least half the window, flat for at least half,
or clipped for at least 10% when measured ADC rails are known. GRAB's
physical-mV export has no attested ADC rail, so clipping is unavailable.
These fixed severe cutoffs and the source-fitted F9 state were established
before this replay; no classifier or target threshold was fitted.

| Phase | Structural coverage | Correct trials rejected | Errors rejected | Parent 0.5-mask coverage |
| --- | ---: | ---: | ---: | ---: |
| Validation users 5–6 | 56/56 | 0 | 0 | 52/56 |
| Descriptive final users 7–8 | 56/56 | 0 | 0 | 40/56 |

The structural rule avoided the parent mask's false rejections, but it caught
none of the 14 existing F0v2 errors. A controlled constant-channel signal
and known-ADC rail saturation are detected by the analytical tests. No
natural GRAB trial met the severe structural rule, and GRAB supplies no
physical-fault truth, so sensitivity to actual equipment failures is unknown.
This is a post-parent exploratory analysis: both target groups had already
been inspected. It does **not** justify deploying a live Unknown gate or
claiming improved recognition.

The [result](F9_STRUCTURAL_GRAB_RESULTS.json) binds the protocol, original
source-fitted state and checksum-verified GRAB files. Its
[trial rows](F9_STRUCTURAL_GRAB_TRIALS.csv) are independently read back
against the frozen F0v2 predictions and parent F9 trial IDs.
