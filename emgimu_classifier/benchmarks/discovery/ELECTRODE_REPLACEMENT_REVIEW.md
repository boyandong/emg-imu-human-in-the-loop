# Secondary electrode re-placement acquisition and native sanity

Official source: https://zenodo.org/records/4039550 (CC BY 4.0), publication DOI 10.1016/j.bspc.2020.102292. This is an optional secondary dataset, not the historical DS2 and not one of the six core benchmark archives.

The complete 315,604,904-byte archive matched official MD5 c577384488e7fd0d7f7916485c87d72b. SHA-256: d30a1096ed2522b32bfafcfcb2d9cdf5ccf28e1e21a41dda77824317a8c1b1ff. The 773-byte README matched official MD5 d22fe9bdc536cee8f4695d4bfe54301c. Raw files and optional decoder dependencies remain in the workspace, outside Git.

Archive inventory contains 267 native recordings: ten users, nine movement codes and three positions, with all three ID7 EX recordings absent. With seed 20260915, subjects 5, 7 and 10 were selected from 1–10; PS recordings at P1 and P3 were extracted. Each nominal recording has eight EMG columns at 1000 Hz. No IMU, verified physical units, common four-class mapping, interval labels or repetition boundaries have been established.

Actual inspection found missing numeric fields in five rows of ID5_PS_P1 and one row of ID5_PS_P3. The strict native reader rejects both complete recordings. No rows were deleted and no values were imputed. Those two sanity plots inspect only intact leading five-second segments; full-recording quality checks apply to the other four files. Observed-extrema fractions are descriptive and do not establish ADC clipping. The machine-readable audit preserves exact bad-row numbers, file hashes, plot hashes and QC scope.

This evidence supports acquisition and sampled data-quality assessment. It does not prove supervised recognition, repetition-based few-shot calibration, online performance or complete-population signal quality. Repeated movements inside a file cannot be counted as independently labeled trials without authentic annotations. No new classification scores were added to the six-core consolidated results.

Reproduce from the classifier directory with workspace-local py7zr on PYTHONPATH:

    python benchmarks/discovery/scripts/sanity_check_electrode_replacement.py --root ../../work/secondary_raw/electrode_replacement --audit benchmarks/discovery/electrode_replacement_native_sanity.json

Reader contract tests: tests/test_electrode_replacement.py. The acquisition script validates official checksums and constrains writes to the workspace.
