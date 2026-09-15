# Gesture ontology

The strict common task uses a label only when its motor intent is close enough to compare.
Original labels remain in processed manifests even when excluded.

| Canonical concept | Dataset labels | Mapping status | Notes |
|---|---|---|---|
| Neutral | Rest; No Motion; No Movement; Relaxed | EXACT_MATCH | Rest is excluded from force-level mechanism analyses unless it has a real force condition |
| Fist or power close | Fist; Hand Close; Close; Power Grip | FUNCTIONALLY_SIMILAR | Power grasp with an object/load remains condition-specific |
| Open | Open; Hand Open; Extension of all fingers | FUNCTIONALLY_SIMILAR | Wrist extension is not Open |
| Pinch | Pinch; two-finger pinch; thumb-index opposition | FUNCTIONALLY_SIMILAR | Key, tripod, chuck and three-finger pinch stay dataset-specific |
| Wrist flexion | Wave In; Wrist Flexion | FUNCTIONALLY_SIMILAR | Myo Wave In is retained under its original label and mapped only in an explicit common task |
| Wrist extension | Wave Out; Wrist Extension | FUNCTIONALLY_SIMILAR | Not merged with Open |
| Other grasps and movements | dataset-native labels | DATASET_SPECIFIC | Used by dataset-specific heads, never forced into four classes |
| Unlabelled or corrupted intervals | source-native markers | NO_MAPPING | Excluded with an auditable reason |

Shared feature encoders may use all valid source labels with dataset-specific classification
heads. A strict four-class evaluation is only created when all four concepts have defensible
source labels and trial boundaries.
