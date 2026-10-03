# Song real-eight-channel F9 V3 observation replay

The [frozen protocol](F9_DOCUMENT_V3_PROTOCOL.json) reuses the previously audited 250 Hz, eight-channel, 50-sample raw-ADC formal windows. S01/S02 supply 849 source windows for fitting. S03/S04 supply 416/431 read-only evaluation windows; their session hashes and trial identities are checked before extraction. The [result](F9_DOCUMENT_V3_RESULTS.json) stores only aggregate observations, not raw signal or per-trial data.

| Evaluation session | F9v3 dimensions | Median zero fraction | Median longest flatline / channel | Median absolute robust amplitude z | Synthetic constant ch1 detected above 0.9 | Natural ch1 above 0.9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S03 | 69 | 0 | 0.020 | 0.709 | 416/416 | 0/416 |
| S04 | 69 | 0 | 0.020 | 1.375 | 431/431 | 0/431 |

The ADC count range and raw/pre-highpass availability are explicit. Mains frequency and circular electrode order have no verified metadata here, so both corresponding availability flags are zero; no 50/60 Hz or ring-neighbor observation is inferred from the target windows. The synthetic constant-channel check exercises the formula, not a measured fault distribution or a selected Unknown threshold.

This is one participant on one date. It confirms native shape, finite output, source-state immutability and unavailable-metadata handling for the versioned F9 observation API. It does not support a default quality gate, cross-person/day accuracy or live device fault prevalence. Replay with the user's existing local Song archive: `PYTHONPATH=src;. python benchmarks/song_real8/f9_document_v3_run.py E:/qxy/emg_meta/emg_meta/data/Song` on Windows.
