# F3c public wearing channel-order sensitivity

The [versioned protocol](F3C_CHANNEL_ORDER_PROTOCOL.json) reuses the frozen six-user wearing split and source-only F0v2 + document-F3c classifier recipe. F0v2 stays in native column order. For the F3c block alone, source and target receive the same fixed ordering: the saved native order or one of 12 seeded, distinct rotation/reflection classes. Each candidate is fitted only on source trials; no target windows tune a model or select an order. The [prediction file](F3C_CHANNEL_ORDER_PREDICTIONS.csv) contains 3,120 native-trial rows, and the [result](F3C_CHANNEL_ORDER_RESULTS.json) binds parent, protocol and prediction hashes. Native-order predictions replay the frozen parent exactly.

| Previously inspected phase | Native-order pooled F1 / loss | Native F1 rank | Native loss rank | Shuffled F1 range | Shuffled loss range |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation, users 15–17 | .6829 / .9477 | 2/13 | 1/13 | .6123–.6857 | 1.0398–1.3130 |
| Descriptive final, users 18–20 | .7113 / .8020 | 1/13 | 1/13 | .4959–.6652 | 1.0401–1.2962 |

The native column order is among the stronger orders under this fixed comparison; one nonphysical order has slightly higher validation F1, while none has lower validation loss. This supports **order sensitivity of the algorithmic cyclic-index statistic**, not physical adjacency of saved columns. The comparison is post-hoc, includes only 12 alternatives and previously examined people/domains, and is not a permutation significance test or a new model-selection basis. Without acquisition metadata linking CSV columns to physical Myo pods, F3c remains an exploratory index-order candidate, not verified anatomical ring evidence or a default model.
