"""Account for every protocol reference and detection, including missed actions."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABELS = {2: 2, 3: 1, 6: 3}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    paths = [HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json',
             HERE/'DETECTED_G5_UNIBO_V1_RESULTS.json']
    boundary, model = [json.loads(p.read_text()) for p in paths]
    lookup = {(e['member'], e['event_index']): e for e in model['events']}
    if len(lookup) != len(model['events']):
        raise ValueError('Duplicate event identity')
    references = []; extra = []; seen = set()
    for recording in boundary['recordings']:
        member = recording['member']
        matches = {m['reference_index']: m for m in recording['matches']}
        if len(matches) != len(recording['matches']):
            raise ValueError('Duplicate matched reference')
        used = set()
        for j, (native, start, end) in enumerate(recording['reference_bounds']):
            label = LABELS.get(native)
            match = matches.get(j)
            event = lookup[(member, match['event_index'])] if match else None
            if event:
                key = (member, match['event_index'])
                if key in seen or [event['start'], event['end']] != recording['detected_bounds'][match['event_index']]:
                    raise ValueError('Detection identity or geometry mismatch')
                seen.add(key); used.add(match['event_index'])
                if event['reference_label'] != label:
                    raise ValueError('Native label mapping mismatch')
            references.append({'member': member, 'reference_index': j, 'native_label': native,
                'label': label, 'start': start, 'end': end,
                'event_index': match['event_index'] if match else None,
                'prediction': event['g5_prediction'] if event else None,
                'oracle_prediction': event['g5_oracle_prediction'] if event and label is not None else None,
                'outcome': ('unsupported_matched' if event else 'unsupported_missed') if label is None
                    else 'missed' if event is None else 'correct' if event['g5_prediction'] == label else 'wrong'})
        for i, bounds in enumerate(recording['detected_bounds']):
            if i not in used:
                key = (member, i); event = lookup[key]
                if key in seen or [event['start'], event['end']] != bounds:
                    raise ValueError('Unmatched event identity mismatch')
                seen.add(key)
                extra.append({'member': member, 'event_index': i, 'start': bounds[0], 'end': bounds[1],
                              'prediction': event['g5_prediction'], 'probability': event['g5_probability']})
    if seen != set(lookup):
        raise ValueError('Unaccounted model events')
    per_class = {}
    for label in (1, 2, 3):
        group = [r for r in references if r['label'] == label]
        counts = {key: sum(r['outcome'] == key for r in group) for key in ('correct', 'wrong', 'missed')}
        counts.update(references=len(group), end_to_end_recall=counts['correct']/len(group),
                      prediction_counts={str(c): sum(r['prediction'] == c for r in group) for c in range(4)})
        per_class[str(label)] = counts
    supported = [r for r in references if r['label'] is not None]
    matched = [r for r in supported if r['event_index'] is not None]
    paired = {'oracle_correct_detected_wrong': sum(r['oracle_prediction'] == r['label'] and r['prediction'] != r['label'] for r in matched),
              'oracle_wrong_detected_correct': sum(r['oracle_prediction'] != r['label'] and r['prediction'] == r['label'] for r in matched)}
    totals = {key: sum(r['outcome'] == key for r in references)
              for key in ('correct', 'wrong', 'missed', 'unsupported_matched', 'unsupported_missed')}
    totals.update(references=len(references), detections=len(lookup), unmatched_detections=len(extra),
                  supported_references=len(supported))
    output = {'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              'generator_sha256': sha(Path(__file__)), 'references': references, 'unmatched_detections': extra,
              'totals': totals, 'per_class': per_class, 'boundary_paired': paired,
              'scope': 'Fixed public UniBo four-channel Day6 diagnostic. Protocol-label reference accounting; unmatched events have unknown truth, and unsupported gestures are not neutral. Oracle comparisons diagnose sensitivity, not causal attribution or biological boundaries.',
              'default_promoted': False, 'completion_proven': False}
    (HERE/'DETECTED_G5_OUTCOME_V1_RESULTS.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'totals': totals, 'per_class': per_class, 'boundary_paired': paired}), flush=True)


if __name__ == '__main__':
    run()
