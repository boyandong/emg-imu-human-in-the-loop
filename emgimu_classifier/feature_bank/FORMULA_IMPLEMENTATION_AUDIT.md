# Formula implementation boundaries

This selected formula audit does not prove completion. Exact document headings and source symbol spans are in the CSV.
Synthetic dimensions check interface shape only; they do not establish native topology, probability calibration or accuracy.

| Item | Reviewed status | Dimension | Boundary |
|---|---|---|---|
| F0_noise_candidate | candidate_formula | 6C | Six metrics with thresholds frozen exclusively from native Rest adjacent-difference noise; active contraction magnitude cannot set thresholds. Historical extra R0 features still unavailable. |
| F2b_document_candidate | candidate_formula | 2 H min(2,floor(C/2)) | Uncentered XX transpose/(trace+epsilon), source-only one-vs-rest generalized eigenproblem, source-fixed gamma and top2/bottom2, log variance normalized plus epsilon. Native held-out wearing Core experiment available. |
| F0 | partial | 6C | Six requested metrics present; thresholds fit pooled source differences rather than explicit calibration noise; old R0 retention cannot be proven without old artifacts. |
| F1 | reference_only | C | Matches conceptual RMS/global RMS pattern; mandatory validated X1-H reuse missing. Family identifier is historical-looking but does not establish equivalence. |
| F2a | candidate_formula | C(C+1)/2 | Centered sample covariance, fixed .05 shrinkage, epsilon diagonal, trace normalization and sqrt(2) off-diagonal vectorization; source-only fit. |
| F2b | partial | 2 H min(tails,floor(C/2)) | Source-only one-vs-rest generalized eigensystem; uses centered/shrunk covariance instead of stated uncentered XX transpose. Default one tail component versus suggested two. |
| F2c | candidate_formula | C(C+1)/2 | Explicit permitted log-Euclidean training reference; whitened matrix log and sqrt(2) vech; not geometric-mean claim. |
| F3a | reference_only | 2 floor(C/2) block | 25ms smoothed rectification, lag mean/std correlations; validated old envelope/aggregation missing; circular topology assumed by legacy class. |
| F3b | reference_only | C block | Normalized sorted correlation eigenvalues are candidate CES block; validated old reuse missing; bundled with lag/ringcov blocks. |
| F3c | candidate_formula | 6 floor(C/2) | New independent raw F2a covariance block with verified ring contract; legacy RingGeometryFamily instead uses envelope covariance and must remain a distinct proxy. |
| F4a | reference_only | BC block | Hann periodogram, four bands strictly below Nyquist and channel L2 band vectors; exact historical Frequency implementation unavailable. |
| F4b | candidate_formula | 4C block | Periodogram total power/centroid/MDF and optional entropy normalized by log(number of bins); FFT power units not calibrated physical PSD. |
| F4c | candidate_formula | 2K block | Non-DC low-order unnormalized DCT-II mean/std across channels, K=4 prespecified; cepstral summary not reproduction of paper CCA. |
| F4d | partial | BC | Calibration-only mean log-band subtraction implemented; session-minus-long mean requires explicit separate references; context not fatigue. |
| F5a | validated_reuse_narrow | 5C native G5 | Calls unchanged native UniBo G5 at 4ch/processed200Hz: early-minus-late and raw waveform slope; different from new TemporalFormFamily late-minus-early/envelope slope. |
| F5_reference | reference_only | 7C+1 | New seven-channel metrics and map velocity; log early/late, normalized entropy/time differ from optional conceptual examples; cannot be called original G5. |
| F5b | partial | H | DTW requires CompleteSequenceBatch, explicit full coverage and finite native durations >=1s; compressed bin rate cannot prove completeness. Legacy sparse MANUS runner refuses new execution; full UniBo bout replay preserved. Native boundary validity and stream segmentation remain separate. |
| F5c | optional_candidate | C+C squared | Per-time L2 rectified path; centered start, levels1/2, no absolute time; scale-normalized raw rectification rather than smoothed envelope. |
| F6a | partial | 13 IMU block | Real accel/gyro magnitude summaries and mean gravity direction present; calibration body-frame transform, gravity lowpass and linear acceleration RMS absent. No stable absolute yaw claimed. |
| F6b | candidate_formula | P posture block | Training-fixed posture one-hot, unseen category rejection; explicit oracle context, never fabricated IMU. |
| F7 | partial | 2H+2 | Mean/median prototypes, Euclidean/source-or-cal standardized/cosine distances, fixed-cal similarity and margins; optional shrinkage Mahalanobis and native SPD distance not present. |
| F8 | partial | 2H+H(H-1)/2 | Residual norms, cosines and pair geometry exact generic block; family-specific scale/RLCS/spectral/SPD/quality summaries are separate partial study evidence, not complete in this API. |
| F9_legacy | reference_only | 6C+5 | Legacy flatline is fraction of flat edges rather than longest run; unknown ADC yields zero without mask; optional low-frequency observation unavailable. Preserve legacy measurements. |
| F9v2 | partial | 9C+8 | Adds longest consecutive flat edges/T, source thresholds, correlation anomaly, availability masks and conditional pre-highpass ratio. Legacy quality mask remains unchanged: new observation not automatically a validated fusion gate. |
| CAL_A | candidate_formula | C centers+C scales; output EMG unchanged shape | Explicit rest median and active absolute Q95; raw/cal branches retained in integrated runs, source/current normalization comparison can decline. |
| CAL_C | candidate_formula | 3 quantiles plus C pattern mean and spread per native gesture | Raw global RMS q10/q50/q90 equal-trial empirical CDF plus within-gesture pattern spread; exclude Rest, archive signal units, not measured force. |
| CAL_D_E | partial | K weights | Between/within separation log-softmax and n0/(n0+N) shrinkage implemented; tau/n0 source CV exists only selected protocols, no global source-CV coverage claim. |
| SESSION | partial | two-family branch-specific state | Current rest/scale/quality/signature/local prototypes preserve long profile; controlled native wearing0/1 only, 2/5 unsupported per domain, current device/calendar-day validation missing. |
| FUSION | partial | H output probabilities | Nonnegative weighted probabilities and clipped quality renormalization; reject-all fallback retains valid distribution but explicit Unknown output absent here. Empirical source probability calibration audited for specified runs only. |
