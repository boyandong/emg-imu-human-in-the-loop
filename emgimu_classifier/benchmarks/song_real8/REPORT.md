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
shortcut is now named `EMG-IMU 项目版（Song 8通道）`; it points to this repository's
collection `main.py`. A separate older `EMG 数据采集` shortcut still points to
`E:/qxy/emg_meta/emg_meta/main.py`, whose models directory has no Song bundle.
This check does not use the physical USB device or measure actual screen
latency or recognition accuracy while a person performs gestures.
