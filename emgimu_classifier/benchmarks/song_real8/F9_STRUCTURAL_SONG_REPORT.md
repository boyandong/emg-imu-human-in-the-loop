# Exploratory structural F9 replay on Song raw ADC

The [frozen protocol](F9_STRUCTURAL_SONG_PROTOCOL.json) applied the severe
zero/flatline/known-rail saturation rule from the GRAB exploratory diagnostic
to one person's raw 8-channel, 250 Hz Song recordings. Source S01/S02 alone
established the F9 references. The current class has a new availability
attribute, so its object bytes differ from the earlier Song result; the
replay instead requires exact matches to the parent source window count,
availability, output names and every per-channel threshold before target
analysis. Frozen F0 probabilities are joined by native trial identity.

| Session | Native trial coverage | Correct rejected | Errors rejected | Existing F0 errors | Synthetic constant-channel trials detected |
| --- | ---: | ---: | ---: | ---: | ---: |
| S03 validation | 140/140 | 0 | 0 | 9 | 140/140 |
| S04 descriptive final | 144/144 | 0 | 0 | 13 | 144/144 |

No natural formal trial triggered the severe structural rule; copied target
windows with channel 1 held constant did trigger it in every trial. The
earlier soft 0.5 quality gate rejected 4 S03 and 29 S04 trials, mostly
correct F0 predictions. The narrower rule avoids those false rejections but
offers no evidence that it catches actual hardware failures or improves
gesture accuracy. S03 failed readiness, S04 and the earlier mask had been
inspected, and both sessions came from one person on one day. This rule
remains diagnostic and is **not deployed**.

The [result](F9_STRUCTURAL_SONG_RESULTS.json) and
[trial rows](F9_STRUCTURAL_SONG_TRIALS.csv) bind the source/target recording
checksums and frozen F0 predictions, with an independent readback test.
