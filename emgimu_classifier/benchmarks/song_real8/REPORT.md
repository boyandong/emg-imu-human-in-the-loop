# Song real 8-channel exploratory study

The user supplied one participant's four HDF5 v3 sessions, recorded on 2026-09-18 with 8-channel EMG at nominal 250 Hz and 6-axis IMU at nominal 112 Hz. Raw files remain at `E:/qxy/emg_meta/emg_meta/data/Song/` and are not redistributed. The script verifies every HDF5 file against its collection-readiness SHA-256 before use. Full machine-readable counts and confusion matrices are in [RESULTS.json](RESULTS.json).

The collection-assigned split was held fixed: S01/S02 train, S03 validation, S04 final test. We used only valid, completed formal trials and up to three non-overlapping 200 ms windows wholly inside each recorded stable interval. Model fitting and feature standardization used S01/S02 only. We selected the four-hand-state feature configuration by S03 trial-level macro-F1, then evaluated S04 once. This selects F0 local EMG detail alone over F0+scale pattern and F0+IMU. A separately specified 28-state endpoint used F0+IMU without test-set model selection. Prediction probabilities were averaged within each trial before scoring; a trial counts once.

| Endpoint | Validation S03 | Final test S04 |
| --- | ---: | ---: |
| Four hand states, accuracy / macro-F1 | 96.4% / 96.5% (140 trials) | 90.3% / 90.0% (144 trials) |
| 28 hand×arm states, accuracy / macro-F1 | 70.0% / 61.2% (140 trials) | 64.6% / 53.1% (144 trials) |

On S04, four-state recall was 33/36 fist, 27/36 index pinch, 34/36 neutral, and 36/36 open hand. This suggests the earlier “almost always rest, rarely open” behavior is not inherent to every classifier on these recordings; it does not prove the existing live UniBo pipeline has been repaired. The offline study operates on cued stable segments and uses zero-phase filtering; it does not reproduce live continuous decoding, transition periods, online calibration, or end-to-end latency.

The actual pre-formal calibration blocks also support a paired 0/1/2-shot
session-calibration test ([CALIBRATION_RESULTS.json](CALIBRATION_RESULTS.json)).
Each shot uses one neutral, pinch, fist and open block; each contributes three
200 ms windows. The source model uses S01/S02 only. S03 chooses the strength
of neutral feature recentering and personal prototype mixing separately for
each budget, before testing S04. Both budgets selected zero recentering and a
0.5 personal-prototype mixing weight. All 144 S04 formal trials remain test
trials because calibration comes from distinct earlier protocol blocks.

| S04 four-state endpoint | 0 shot | 1 shot | 2 shots |
| --- | ---: | ---: | ---: |
| Trial accuracy | 90.3% | 93.8% | 93.8% |
| Trial macro-F1 | 90.0% | 93.6% | 93.6% |
| Changed decisions vs zero shot | — | 5 corrected, 0 new errors | 5 corrected, 0 new errors |

The paired macro-F1 gain is 3.56 percentage points. A descriptive 4,000-draw
class-stratified trial bootstrap gives a 95% interval of 0.72–7.00 points;
the exact paired accuracy discordance test gives p=0.0625. The gain is
promising but uncertain in this single-session test. A second shot provided
no additional benefit. The selected cue durations sum to 14/28 seconds, while
the actual S04 spans from the first to last selected block were 26.0/63.99
seconds because other protocol blocks occurred between them. Neither measure
includes device preparation or operator time, so these are not product
onboarding times.

We repeated the full source fit, S03 selection and S04 evaluation with causal
continuous 40 Hz high-pass and 50/100 Hz notch filters. A prefix-invariance
test confirms future samples cannot alter earlier filtered outputs. This is
still **offline scoring of cued stable intervals**, not live continuous
recognition. The causal [classification results](CAUSAL_RESULTS.json) and
[calibration results](CAUSAL_CALIBRATION_RESULTS.json) are:

| Causal S04 endpoint | 0 shot | 1 shot | 2 shots |
| --- | ---: | ---: | ---: |
| Four-state trial accuracy | 91.0% | 91.7% | 91.0% |
| Four-state trial macro-F1 | 90.7% | 91.3% | 90.6% |
| Corrected / new errors vs zero | — | 2 / 1 | 2 / 2 |

Causal 28-state trial accuracy/macro-F1 are 68.1%/54.9% with the fixed
F0+IMU design. The one-shot causal macro-F1 difference versus zero is only
+0.57 points, with a descriptive paired 95% bootstrap interval of -1.70 to
+3.28 points. Two-shot is slightly worse. Thus the larger zero-phase
calibration improvement is not robust to this necessary real-time-compatible
signal-processing change. The difference itself is one-session evidence and
does not isolate a physiological cause.

S01–S03 each failed collection readiness. The study deliberately uses their individually valid trials for exploratory analysis under the user's instruction; it does not satisfy the formal data-collection acceptance gate. S04 passed readiness. S01/S02/S03/S04 yielded 143/142/140/144 usable formal trials respectively. All data come from one participant on one date. Cue labels have `raw_cues_only` semantics, rather than verified physiological movement onsets. There is no cross-participant or cross-day generalization estimate, clinical claim, or deployment approval.

Reproduce locally from the classifier directory with `PYTHONPATH=src python benchmarks/song_real8_study.py --source E:/qxy/emg_meta/emg_meta/data/Song --output benchmarks/song_real8/RESULTS.json`. On PowerShell, set `$env:PYTHONPATH='src'` before calling Python. The script prints per-session and feature-family progress. Unit checks: `PYTHONPATH=src;. python -m unittest tests.test_song_real8_study -v` (PowerShell syntax for the environment variable).

For calibration reproduction, use `PYTHONPATH=src;. python benchmarks/song_calibration_study.py --source E:/qxy/emg_meta/emg_meta/data/Song --output benchmarks/song_real8/CALIBRATION_RESULTS.json` on PowerShell. The scripts verify the original HDF5 hashes and reject incomplete calibration blocks or overlap with formal data.
Pass `--filter-mode causal` to either script and use the matching `CAUSAL_*.json` output filename to reproduce the causal comparison.

The S01/S02 causal F0 model can now be exported as a small hash-checked local
JSON bundle for the collection application's realtime page. The bundle is
generated by `benchmarks/export_song_live_model.py` and remains outside Git
under the application's ignored `models/song_real8_f0/` directory. The
[runtime replay audit](LIVE_EXPORT_REPLAY_AUDIT.json) rebuilt the source
classifier and checked all 416 S03 windows (maximum probability discrepancy
`1.41e-7`), then fed the first 100 seconds of S04 in 37-sample chunks; all
999 emitted 200 ms windows matched the offline causal filter exactly. A Qt
offscreen test loaded a schema-conforming synthetic bundle and displayed a probability
frame. This verifies the preprocessing/model transfer, not the USB device,
continuous event scoring or latency.
In the local CPU replay, 37-sample chunks took 0.58 ms at the 95th percentile
(0.97 ms maximum), compared with 148 ms of nominal sample time per chunk.
These are processing times only; acquisition, UI queueing and output data age
remain unmeasured.

The [continuous S04 cue-timeline audit](CONTINUOUS_REPLAY_AUDIT.json) replayed
all 297,942 EMG samples through the exported runtime in 37-sample chunks.
Across 11,916 sliding windows, 7,865 (66.0%) were predicted as neutral;
this includes calibration, rest, transitions and uncued time, so it is not an
accuracy score. Within the recorded stable cue intervals, 911 fully contained
windows reached 80.1% frame accuracy and 144 trial means reached 89.6%.
Open Hand was correct in 36/36 stable-cue trials, Fist in 35/36, Neutral in
33/36 and Index Pinch in 25/36. In 1,175 windows wholly inside pre-prompt
rest intervals, 87.1% were predicted neutral. These labels are task cues, not
measured EMG onsets; no continuous event-detection accuracy follows from them.
The live UI now keeps a debounced current Song label visible while its frames
continue, instead of clearing a sustained action after two seconds.

The same S04 replay now also applies the live worker's fixed 0.5 probability
threshold and three-consecutive-frame rule to every 100 ms prediction frame.
On the 911 frames wholly inside stable cue intervals, the raw class winner
matches the cue in 80.1%; the resulting persistent decoder state matches in
73.1% (macro-F1 72.3%). Decoder-state recall is 74.6% Fist, 40.2% Index
Pinch, 90.6% Neutral and 81.8% Open Hand; 24 stable frames remain unresolved.
This shows that the display rule can hide short or inconsistent active
predictions, especially Pinch. The Qt page emits its latest state per processed
input batch, so this is a decoder-state replay, not a measured screen refresh
rate or a live accuracy test. These S04 results were inspected after the rule
was set and must not be used to tune its threshold or duration on the final
session.

A follow-up [decoder rule study](DECODER_RULE_STUDY.json) replayed the same
source-trained model on S03 first, comparing 12 causal rules: probability EMA
alpha 0.25/0.5/0.75/1.0 and one/two/three consecutive 100 ms frames, all at
the existing 0.5 threshold. The current unsmoothed three-frame rule reached
82.16% stable-cue frame macro-F1 on S03, 67.78% neutral state in pre-prompt
rest and 38.2 state changes per minute across its full stream. Removing the
debounce raised S03 macro-F1 to 85.98% but caused 131.9 changes/minute. The
study required a candidate to preserve rest neutrality within two percentage
points and to produce no more full-stream state changes than the current rule;
none of the eleven alternatives met both conditions. The rule therefore stays
unchanged. Selection was fixed before loading S04, although S04's earlier
current-rule results had already motivated this exploratory study. This is a
negative usability tradeoff result, not a confirmatory test or a reason to
claim live recognition is solved.

The separately frozen [cue-relative event audits](CUE_EVENT_REPORT.md) read
the F0 and current F0+SPD continuous decoder states one trial at a time.
Among 108 active stable-cue intervals, F0 matches 92 and F0+SPD matches 100.
Index Pinch increases from 21/36 to 30/36, but late pre-prompt rest
intervals with an active decoded state increase from 16/144 to 28/144.
The current model's successful active-event first-correct median is 0.288 s
relative to the recorded stable cue start; actual physiological onset and
screen latency were not captured.

The [causal SPD increment study](SPD_INCREMENT_RESULTS.json) tests a missing
native eight-channel feature family with exactly the same Song split and 200 ms
stable formal-trial scoring. A log-tangent SPD reference and two logistic models
were fit from S01/S02 only: source F0 and source F0+F2c SPD. Neither uses S03
or S04 to fit its source feature state or classifier. S03/S04 trial macro-F1 is
93.59%/90.68% for F0 and 97.14%/95.08% for F0+SPD. On S04, the latter
corrects six F0 errors with no new errors; log loss falls from 0.4272 to
0.2376 and Brier from 0.0533 to 0.0298. The exact paired accuracy
discordance p-value is 0.03125, and a descriptive class-stratified paired
bootstrap gives a macro-F1 difference interval of +1.45 to +7.81 points.
These values concern cued stable trials from one person/day. S04 was already
examined in earlier project work, so the interval and p-value are exploratory
descriptions, not independent confirmation or a live-recognition claim.

The same study fits F7 SPD personal prototypes from the pre-formal Song
calibration blocks, one or two blocks per class, with S01/S02's SPD reference
frozen. S03 selects fixed F0/SPD probability mixing weights of 0.75 and 0.25
for the respective budgets before S04 is loaded. S04 mixture macro-F1 is
92.25%/92.10%, above source F0 but below source-only F0+SPD at 95.08%; the
anchor alone reaches 79.01%/88.83%. This does not justify adding a personal
SPD calibration step to the product. F2c/F7 here are tangent-space candidates,
not recovered historical RLCS or exact affine-invariant geodesics.

The F0+SPD source model is exported as a separate hash-checked local bundle at
`collection/emg_meta/emg_meta/models/song_real8_f0_spd/`; the original F0
bundle remains available. The collection application's Song model selector
discovers both. The [export replay](SPD_LIVE_EXPORT_REPLAY_AUDIT.json) checks
all 416 S03 stable windows against a fresh source fit (maximum probability
error `1.51e-7`) and 999 chunked S04 windows against offline causal filtering
(zero probability error). In the local CPU replay, 37-sample chunks took
0.41 ms at the 95th percentile and 0.87 ms maximum; this does not include USB
or UI age.
From `emgimu_classifier`, regenerate the local bundle with
`PYTHONPATH=src;. python benchmarks/export_song_spd_live_model.py --source E:/qxy/emg_meta/emg_meta/data/Song --output ../collection/emg_meta/emg_meta/models/song_real8_f0_spd`
(set `PYTHONPATH` as a PowerShell environment variable on Windows), then run
`benchmarks/verify_song_live_export.py` against that bundle. The small learned
JSON bundle is intentionally local and Git-ignored; the exporter and digest
evidence are versioned.

The [full S04 F0+SPD replay](SPD_CONTINUOUS_REPLAY_AUDIT.json) gives 88.0%
raw stable-cue frame accuracy and 79.9% accuracy after the unchanged online
three-frame decision rule, versus 80.1% and 73.1% for F0. Mean-probability
stable-trial accuracy is 92.4% versus 89.6%, with Index Pinch correct in
29/36 rather than 25/36 trials. The cost is more active predictions during
pre-prompt rest: the raw neutral-prediction fraction falls from 87.1% to
78.9% there. The bundle is
therefore offered as a selectable experimental model, not promoted as a
universally better default. These are cue-timeline replay values; neither
physical USB recognition nor event-onset latency has been validated.

The decoded display state confirms a smaller but real rest tradeoff: over all
pre-prompt rest windows, the active-state fraction is 21.1% for F0 and 27.7%
for F0+SPD. In the final 400 ms before each prompt, where residual motion from
the preceding trial should matter less, it is 9.3% versus 17.2% (291 windows
for either model). These are cue-defined intervals rather than verified
physiological rest; the newly added counters are in the corresponding
continuous replay JSONs.

A separate [neutral-bias study](NEUTRAL_BIAS_STUDY.json) applied fixed neutral
logit offsets 0 to 1.5 to F0+SPD probabilities, keeping the online threshold
and three-frame rule. S03 alone selected +1.0 under constraints on late-rest
active state, stable-cue macro-F1 and state-change rate. On S04, this reduces
F0+SPD late-rest active display from 17.2% to 12.0% and changes stable-cue
decoded macro-F1 from 79.8% to 80.9%; F0 remains lower at 9.3% late-rest
active display and 72.3% stable-cue decoded macro-F1. The S04 baseline had
already motivated this exploration, and the same-person/day result does not
establish a calibrated deployment threshold. The +1.0 bias is not installed
in the selectable bundle or made the application default.

The [28-state SPD increment study](SPD_28_STATE_RESULTS.json) separately adds
source-fitted F2c SPD to the fixed F0+real-IMU classifier for hand×arm labels.
It rejects execution unless its F0+IMU baseline exactly reproduces the saved
causal S03/S04 accuracy and macro-F1. The reproducing environment was Python
3.13.9, NumPy 2.4.3, SciPy 1.17.1 and scikit-learn 1.8.0; a second local
environment with older NumPy/SciPy gave a one-trial S04 baseline difference
and was rejected. On S03, joint macro-F1 rises from 59.58% to 60.81%. On
S04, accuracy rises from 68.06% to 70.83% and log loss falls from 1.2966 to
1.2150, but joint macro-F1 falls from 54.87% to 51.70%. There are 18
corrected trials and 14 new errors (paired accuracy discordance p=0.597).
S04 has 72 still-arm trials and just 12 for each other arm: still joint
accuracy rises from 81.9% to 97.2%, while backward falls from 41.7% to
8.3% and down from 83.3% to 58.3%. This distribution explains how aggregate
accuracy can improve while condition-balanced macro-F1 worsens. The new SPD
arm is not promoted for 28-state recognition. These are same-person/day
cue-labelled trials, with sparse per-condition counts and no live validation.

The actual project collection page also passed an [offscreen real-bundle UI
replay](SPD_UI_OFFSCREEN_AUDIT.json). It discovered and loaded the local
F0+SPD bundle from the same `models/` directory used by `main.py`, then
accepted 1,000 contiguous raw S04 EMG samples and emitted a prediction whose
probabilities matched direct runtime execution exactly. The desktop project
shortcuts `EMG 数据采集` and `EMG-IMU 项目版（Song 8通道）` now point to this
repository's collection `main.py`. The former external shortcut has been
preserved as `EMG 数据采集（旧版，无 Song 模型）`; its
`E:/qxy/emg_meta/emg_meta/` models directory has no Song bundle.
This check does not use the physical USB device or measure actual screen
latency or recognition accuracy while a person performs gestures.
The Song-specific desktop shortcut now passes `--song-realtime`. It opens
the realtime tab and hash-checks/loads the existing F0+SPD bundle at startup;
the generic data-collection shortcut keeps its normal start page. The
versioned [shortcut installer](../../../collection/emg_meta/emg_meta/install_song_shortcut.ps1)
recreates this local Windows link. A current offscreen full-window launch
confirmed that the selected and loaded model was F0+SPD even though UniBo
models are also discoverable. This fixes the launch-path selection, not the
unmeasured physical USB or new-wearing recognition accuracy.

The [S03](CUE_RESPONSE_S03_AUDIT.json) and [S04](CUE_RESPONSE_S04_AUDIT.json)
cue-response audits replay the two exported bundles on full continuous streams
without refitting. They measure the first matching **decoded frame end** after
each recorded prompt starts, within that prompt; this is not movement-onset or
measured screen latency. On S04, F0 shows the prompted hand state at least once
in 139/144 formal cues and F0+SPD in 142/144. The paired difference is three
SPD-only hits, no F0-only hits, 139 both and two neither. For the 139 both-hit
cues, SPD is earlier on 51, equal on 74 and later on 14; the paired latency
median is 0 ms. In particular, S04 Index Pinch cue hits rise from 31/36 to
34/36 while Open Hand is 36/36 for both. S03 has 139/141 hits with either
model and no discordant hits. These counts include cues where the state was
already correct before prompting; a separate transition-needed subset is in
the audit. The decoded state persists across trials, as in the application.
One-person cue timing, not physiological onset or physical USB timing, limits
the conclusion. No threshold or model default changed based on this replay.

The [protocol-to-UI replay](PROTOCOL_TO_UI_AUDIT.json) additionally re-encodes
the first 1,000 saved S04 8-channel samples as signed 24-bit device-protocol
frames. The project's `FrameParser`, `AcquisitionController`, `MainWindow`
signal wiring and Song worker then decode and classify them. All 1,000 rows
retain their saved channel order; the final UI prediction at sample 999 matches
direct runtime probabilities exactly. S04's recorded mean EMG rate was 249.54
Hz against a nominal 250 Hz. The original physical packet bytes were not
saved, so this verifies reconstructed protocol software behavior, not the
actual USB link, live timing, or electrode placement on another wearing.

An [F4d native context diagnostic](F4D_SESSION_SHIFT_AUDIT.json) implements
separate long-term and current-session spectral references without evaluation
reuse. The long reference uses the two initial-rest calibration trials in
S01/S02; S03 and S04 each use their own initial-rest trial for session
calibration and 18 separate still-neutral formal trials for held-out
description. All four source HDF5 digests match the saved readiness records.
The mean absolute 32-coordinate `session−long` log-band shift is 1.048 for
S03 and 1.022 for S04. These shifts are descriptive sensor/context evidence
from one person on one day, not fatigue, cross-day drift, or a demonstrated
recognition benefit. S01–S03 whole sessions failed collection readiness.

The [trial-level probability audit](PROBABILITY_CALIBRATION_AUDIT.json) reloads
the exported source-only F0 and F0+SPD bundles, reproduces every saved S03/S04
accuracy, macro-F1, LogLoss and Brier value within 1e-5, then computes a fixed
10-bin top-label ECE without refitting. S03 ECE is 0.1600 for F0 and 0.0913
for F0+SPD; S04 is 0.1851 and 0.1029. The 40 [bin rows](PROBABILITY_CALIBRATION_BINS.csv)
give support, mean confidence and observed accuracy, including empty bins.
This is reliability of cued stable-trial mean probabilities for one person/day,
not a claim that the live interface is calibrated. S04 was previously inspected.
The four zero-shot family scores, two conditional increments, two paired error
rows and ten calibration-curve rows are now included in the versioned Feature
Bank delivery tables. Six curve rows are a **separate** 0/1/2-shot
F0+personal-SPD-anchor method: 0-shot is source F0, while 1/2-shot mixtures
use only distinct pre-formal calibration blocks. S03 selected the two mixture
weights before S04. On S04, macro-F1 is 0.9068/0.9225/0.9210 at 0/1/2 shots,
but LogLoss worsens from 0.4272 to 0.8647/0.5357; source-only F0+SPD remains
better at 0.9508 F1 and 0.2376 LogLoss. Only two blocks per class exist, so
5-shot is unsupported. No Song result is labelled a full F0–F9 bank or a
cross-person/day robustness estimate.

The [temperature tradeoff audit](TEMPERATURE_STUDY.json) applies seven fixed
probability temperatures to the exported source-only F0+SPD bundle without
fitting another classifier. S03 alone selects a temperature subject to
late-rest false-active, stable-cue macro-F1 and state-change constraints. Its
best eligible candidate, 0.67, reduces stable-trial LogLoss only from 0.1866
to 0.1800, below the prespecified 0.01 minimum improvement; the selected
temperature is therefore 1.0. S04 retains 0.2376 LogLoss, 0.1029 ECE,
0.7980 decoded stable-cue macro-F1 and 17.2% late-rest active display. This
negative result does not justify a UI probability or threshold change. It is
exploratory because S03 failed collection readiness, S04 had already been
inspected, and all sessions are from one person on one day.

The [fixed-gain sensitivity audit](GAIN_SENSITIVITY.json) tests the existing
source-only F0 and F0+SPD bundles on the same S03/S04 causal stable windows,
with all eight filtered channels synthetically multiplied by 0.25–4.0. The
gain-1.0 scores reproduce the frozen probability audit within 1e-5. On S04,
F0 macro-F1 falls from 0.9068 at gain 1.0 to 0.3629 at 0.25, and its
active-to-neutral trial error fraction rises from 4.6% to 57.4%; F0+SPD
falls from 0.9508 to 0.9157, with 4.6% to 5.6% active-to-neutral errors.
At gain 0.5, F0/F0+SPD macro-F1 is 0.7483/0.9293. This identifies
common-amplitude sensitivity as one *possible* mechanism for a rest-heavy
display, and the SPD candidate is less sensitive in this fixed simulation.
It does not identify the cause of the user's later live experience. The
observed S04 per-channel median stable-window RMS is 0.98–1.28 times the
combined S01/S02 source reference, not a measured 0.25× collapse. A new
wearing, electrode change, ADC saturation, class transitions and USB behavior
are not represented by this perturbation; no automatic gain correction was
introduced.

An [F9 raw-ADC observability audit](QUALITY_OBSERVABILITY.json) extracts the
same formal 200 ms windows before high-pass filtering, fixes the signed-24-bit
ADC clipping range from the device protocol, and fits F9 references on S01/S02
only. Its descriptive S03/S04 scores show why the existing composite F9
`min_quality < 0.5` must not become a hard recognition gate: it would mark
35.3%/68.0% of recorded stable windows, including 62.0%/90.7% of Open Hand
windows. Every marked window is explained by the source-relative amplitude
`|z| > 3` term; zero, flatline and ADC-clipping fractions do not explain any
of them. A separate **candidate**, restricted to obvious long flatlines or
ADC clipping, marks 0% of these clean windows and 100% of windows with an
injected constant channel. That synthetic success does not establish real
fault detection or another-wearing specificity, so neither gate was deployed.
Line noise and low-frequency ratios are observable in raw data here, but the
existing F9 quality aggregate does not incorporate them. S01–S03 readiness
failures and same-person/day sampling still limit the diagnostic.

A [pre-formal signal-calibration audit](SIGNAL_CALIBRATION.json) evaluates
another 0/1/2-block-per-class method on the frozen source F0+SPD bundle.
S01/S02 pre-formal blocks provide a source per-channel RMS reference; S03/S04
pre-formal blocks alone estimate a bounded target channel scale. Formal
labels and samples never estimate that scale. On unmodified S04, 0/1/2-shot
macro-F1 is 0.9508/0.9513/0.9438 and LogLoss is
0.2376/0.2554/0.2601. S03 also worsens from 0.9714 F1 and 0.1866 LogLoss
at zero shot to 0.9493/0.2050 (one block) or 0.9565/0.1946 (two).
The synthetic 0.25× S04 gain case recovers F1 from 0.9157 to
0.9513/0.9438 after calibration, but this is an algebraic gain-control
diagnostic, not new-wearing performance. A post-hoc, non-fitted amplitude
check reveals why the real correction is risky: the source calibration/formal
median RMS ratio spans 0.30–0.53 by channel, while S04 one-block
calibration/formal spans 0.72–0.91. The pre-formal and formal contexts are
not interchangeable across these sessions. This correction is not deployed.

The [recent-study reproduction recheck](REPRODUCTION_RECHECK.json) reruns the
temperature, fixed-gain, raw-ADC F9 and signal-calibration scripts into fresh
temporary outputs with the recorded Song HDF5s and ignored local model
bundles. All four JSON results match their versioned counterparts byte for
byte. The audit records source, model, manifest and script hashes and the
command templates. It does not rerun the older training, export or full
Feature Bank studies, and the external raw files/model bundles remain required
for reproduction on another computer.

A subsequent [F4 conditional-increment study](F4_INCREMENT_RESULTS.json)
tests the current reference spectral family on the same causal Song split.
S01/S02 alone fit F0, SPD, F4, scalers and four fixed logistic models; S03
contains 140 evaluation trials and S04 contains 144. Before testing F4, both
source F0 and F0+SPD scores exactly replay the saved SPD study on S03 and S04.

| Arm | S03 macro-F1 / LogLoss | S04 macro-F1 / LogLoss |
|---|---:|---:|
| F0 | 0.9359 / 0.3109 | 0.9068 / 0.4272 |
| F0 + SPD | 0.9714 / 0.1866 | 0.9508 / 0.2376 |
| F0 + F4 | 0.9220 / 0.2997 | 0.8813 / 0.4145 |
| F0 + SPD + F4 | 0.9640 / 0.1742 | 0.9508 / 0.2153 |

Adding F4 to F0+SPD reduces LogLoss by 0.0124 on S03 and 0.0222 on S04,
and Brier by 0.00046/0.00214, but S03 macro-F1 falls by 0.0073. The added
family corrects one S03 trial while introducing two new errors; on S04 it
corrects two and introduces two. This probability-versus-hard-label tradeoff
does not justify replacing the selectable live F0+SPD bundle. S04 had already
been inspected, S01–S03 failed whole-session readiness, and all recordings
remain one person/day. The [1,136 trial probability rows](F4_INCREMENT_TRIAL_PREDICTIONS.csv),
[conditional increments](F4_INCREMENT_CONDITIONAL.csv),
[paired errors](F4_INCREMENT_PAIRED_ERRORS.csv) and
[independent read-back](F4_INCREMENT_VERIFICATION.json) preserve the bounded evidence.

The [leave-one-session-out F0+SPD study](SESSION_HELD_OUT_RESULTS.json) uses all
four supplied Song recordings without fitting any feature or classifier on the
held-out session. Each fold trains on valid stable formal trials from the
other three sessions with the fixed
causal filter, F0+SPD features and logistic classifier. The 569 saved
[trial probabilities](SESSION_HELD_OUT_TRIAL_PREDICTIONS.csv) were read back
and rescored after serialization.

| Held-out session | Trials | Accuracy | Macro-F1 | LogLoss |
|---|---:|---:|---:|---:|
| S01 | 143 | 0.9231 | 0.9222 | 0.2297 |
| S02 | 142 | 0.9718 | 0.9713 | 0.1650 |
| S03 | 140 | 0.9714 | 0.9722 | 0.1705 |
| S04 | 144 | 0.9444 | 0.9437 | 0.2506 |

The pooled trial macro-F1 is 0.9522. This measures transfer between four
same-person, same-day recordings under cued stable-interval scoring. It cannot
measure cross-day or cross-person generalization; S01–S03 failed collection
readiness, and these sessions have already informed other analyses. The S04
three-session-fold F1 is below the existing S01/S02-trained F0+SPD result
(0.9437 versus 0.9508), so these exploratory folds provide no reason to
replace the selectable live bundle with an all-session refit.

A fixed [14-arm family screen](FAMILY_HELD_OUT_SCREEN.json) extends those same
four held-out folds to reference F1, F2a, supervised F2b, F2c, F4, F5 and
real-IMU F6. Every feature reference, CSP filter, scaler and classifier is
fitted only on the other three sessions. The F0+F2c core reproduces all 569
prior trial probability vectors exactly (maximum error zero). The 7,966
[saved arm/trial rows](FAMILY_HELD_OUT_PREDICTIONS.csv) were read back and
rescored. This is a fixed screen, not a search for a selected live model.

| Arm | Pooled macro-F1 | Pooled LogLoss | Pooled Brier |
|---|---:|---:|---:|
| F0 | 0.9222 | 0.3639 | 0.0431 |
| F0+F1 | 0.9133 | 0.3516 | 0.0419 |
| F0+F2a | 0.9612 | 0.2176 | 0.0250 |
| F0+F2b | 0.9491 | 0.2578 | 0.0293 |
| F0+F2c (core) | 0.9522 | 0.2043 | 0.0241 |
| F0+F4 | 0.9306 | 0.3241 | 0.0404 |
| F0+F5 | 0.9295 | 0.3436 | 0.0412 |
| F0+F6 | 0.9273 | 0.3419 | 0.0403 |

| Added to F0+F2c core | ΔLogLoss (positive helps) | ΔBrier | ΔMacro-F1 | Corrected / new-error trials |
|---|---:|---:|---:|---:|
| F1 | +0.0090 | +0.0009 | +0.0054 | 5 / 2 |
| F2a | +0.0154 | +0.0013 | +0.0018 | 4 / 3 |
| F2b | +0.0002 | +0.0001 | ~0 | 1 / 1 |
| F4 | +0.0104 | ~0 | +0.0053 | 12 / 9 |
| F5 | -0.0065 | -0.0007 | +0.0037 | 8 / 6 |
| F6 | +0.0091 | +0.0009 | +0.0091 | 10 / 5 |

The F2a conditional LogLoss increment is positive in each of four folds;
F4's is positive in two and negative in two. The pooled F0+F2c core has
per-class F1 of 0.975 fist, 0.942 pinch, 0.922 neutral and 0.969 open;
adding F6 raises neutral to 0.944 but lowers open slightly to 0.966. These
correlated trials and reused same-day sessions cannot support a statistical
claim about new wearings, people or days. F1/F4/F5/F6 are current reference
implementations, not proven exact historical families; F5 summarizes only a
200 ms stable window rather than a complete gesture bout. No arm is promoted
to live use from this exploratory screen.

The [native cue-arm domain audit](CUE_ARM_DOMAIN_AUDIT.json) joins every
saved trial prediction back to its hash-checked HDF5 formal-trial label,
without refitting. Of 569 trials, 283 are cued `still` and only 47–48 occur
in each of the six moving-arm conditions. The F0+F2c core macro-F1 is 0.993
for `still`, but only 0.828 for `up`, its worst condition. Adding F2a raises
the `up` value to 0.867 and improves LogLoss in six of seven cue-arm groups;
`backward` LogLoss worsens by 0.011. F4 helps five of seven groups by
LogLoss, while F6 helps `up` and `down` by 0.056 each but worsens `right`
and `forward` by 0.025/0.023. Thus the pooled family increments do not imply
uniform condition recovery. These are instructed arm labels, not measured
posture or movement onset; small moving-arm cells, one day/person and
previously inspected S04 prohibit a broader robustness claim.

The [fixed-split F2a increment](F2a_INCREMENT_RESULTS.json) checks the
within-day screening signal against the original S01/S02 source fit, S03
validation and S04 final replay. Four fixed causal logistic arms cover the
same 140/144 formal trials; F0 and F0+F2c reproduce the previously saved
baseline before F2a is evaluated. The [1,136 trial probability rows](F2a_INCREMENT_TRIAL_PREDICTIONS.csv),
[conditional deltas](F2a_INCREMENT_CONDITIONAL.csv),
[paired errors](F2a_INCREMENT_PAIRED_ERRORS.csv) and
[independent read-back](F2a_INCREMENT_VERIFICATION.json) preserve the result.

| Arm | S03 macro-F1 / LogLoss | S04 macro-F1 / LogLoss |
|---|---:|---:|
| F0 | 0.9359 / 0.3109 | 0.9068 / 0.4272 |
| F0+F2a | 0.9425 / 0.2061 | 0.9231 / 0.2533 |
| F0+F2c | 0.9714 / 0.1866 | 0.9508 / 0.2376 |
| F0+F2c+F2a | 0.9642 / 0.1649 | 0.9358 / 0.2149 |

F2a adds conditional probability information to F0+F2c: LogLoss improves
by 0.0217/0.0227 on S03/S04, and Brier also improves. Yet macro-F1 drops
by 0.0071/0.0149. On S04, the combined arm corrects one prior error and
creates three; Index Pinch recall falls from 30/36 to 28/36, while Fist
and Open Hand remain 36/36. This explains why positive LogLoss increments
must not be described as better hard-label recognition. The experiment uses
one person/day, previously inspected S04 and three sessions that failed
whole-session readiness. F2a is a current covariance candidate, not a
recovered historical spatial algorithm, and the live F0+F2c bundle stays
unchanged.
