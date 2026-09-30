# Source-frozen F9 quality gate on unseen GRABMyo users

The [protocol](F9_GRAB_GATE_PROTOCOL.json) fixed a 0.5 minimum-channel-quality
trial gate before fitting F9 or examining target outcomes. Its window-count
correction was committed before either step. Source subjects 1–4 supplied
112 native Day1 trials and 2,240 windows. Validation subjects 5–6 and
descriptive final subjects 7–8 supplied 56 trials each, with 20 fixed
512-sample windows per trial. All 224 records passed the frozen official
checksum manifest. The F0v2 probabilities were read unchanged from the
[parent predictions](GRAB_USER_PREDICTIONS.csv), with exact trial-ID and
label matching; no classifier was trained in this study.

ADC rail, mains frequency and pre-highpass status are not attested for this
physical-mV export. All three corresponding F9 observations were explicitly
marked unavailable. The remaining zero/flatline, amplitude and correlation
thresholds were fixed from source windows only.

| Phase | Coverage | Correct trials rejected | Errors rejected | Baseline errors |
| --- | ---: | ---: | ---: | ---: |
| Validation | 52/56 (92.9%) | 4 | 0 | 11 |
| Descriptive final | 40/56 (71.4%) | 14 | 2 | 3 |

The final result is especially uneven by user: subject 7 retains 26/28
trials; subject 8 retains only 14/28. Thus the fixed gate rejects many correct
trials while catching few mistakes. It is **not promoted** to the live
Unknown/fusion path. The [result](F9_GRAB_GATE_RESULTS.json) records
source-state and file hashes, component thresholds, pooled/subject counts;
the [trial rows](F9_GRAB_GATE_TRIALS.csv) allow independent readback.

This is a public 8-channel, 2048 Hz unseen-user diagnostic without measured
hardware-fault labels. The previously inspected final users are descriptive.
It cannot establish safe 250 Hz device behavior or a physical fault detector.
