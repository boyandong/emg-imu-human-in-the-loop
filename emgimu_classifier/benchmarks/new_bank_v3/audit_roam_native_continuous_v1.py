"""Post-run interval geometry diagnosis, without inference or threshold search.

Categories describe saved interval overlap. They do not establish physiological
truth or a causal reason for the detector's errors. No signal processing runs.
"""
import json
from pathlib import Path
from benchmarks.new_bank_v3.roam_native_continuous_v1 import ROOT,HERE,OUT,RESULT,ARMS
from benchmarks.new_bank_v3.roam_native_joint_v1 import sha,write

DIAGNOSTIC=HERE/'ROAM_NATIVE_CONTINUOUS_V1_DIAGNOSTIC.json'


def audit(records):
    rows=[];merged=[]
    for record in records:
        refs=record['references'];events=record['events']
        paired={m['reference_index']:m['event_index'] for m in record['matches']}
        for e in events:
            covered=[r['trial_id'] for r in refs if max(0,min(e['end'],r['end'])-max(e['start'],r['start']))>=(r['end']-r['start'])/2]
            if len(covered)>1:
                merged.append(dict(user=record['user'],shots=record['shots'],member=record['member'],
                    event_id=e['trial_id'],half_covered_reference_ids=covered))
        for i,r in enumerate(refs):
            overlaps=[max(0,min(e['end'],r['end'])-max(e['start'],r['start'])) for e in events]
            ious=[n/(max(e['end'],r['end'])-min(e['start'],r['start'])) for n,e in zip(overlaps,events)]
            best=max(ious,default=0.);intersection=max(overlaps,default=0)
            if i in paired:category='matched'
            elif intersection==0:category='miss_no_detected_interval_overlap'
            elif best<.5:category='miss_overlap_below_iou_threshold'
            else:category='miss_one_to_one_assignment_conflict'
            e=None if i not in paired else events[paired[i]]
            touches=any(s['trial_id']!=r['trial_id'] and (s['end']==r['start'] or s['start']==r['end']) for s in refs)
            rows.append(dict(user=record['user'],shots=record['shots'],member=record['member'],
                reference_id=r['trial_id'],class_name=r['class_name'],category=category,
                best_detected_iou=best,best_reference_coverage=intersection/(r['end']-r['start']),
                touches_another_active_cue=touches,
                matched_event_id=None if e is None else e['trial_id'],
                correct={a:False if e is None else e['labels'][a]==r['class_name'] for a in ARMS}))
    summaries=[]
    for phase,users in (('validation',range(19,24)),('descriptive_final',range(24,29)),('all',range(19,29))):
        for shots in (0,1,2):
            part=[r for r in rows if r['user'] in users and r['shots']==shots]
            categories={c:sum(r['category']==c for r in part) for c in
                ('matched','miss_no_detected_interval_overlap','miss_overlap_below_iou_threshold','miss_one_to_one_assignment_conflict')}
            summaries.append(dict(phase=phase,shots=shots,references=len(part),categories=categories,
                active_cues_touching_another_active_cue=sum(r['touches_another_active_cue'] for r in part),
                detected_intervals_half_covering_multiple_active_cues=sum(r['user'] in users and r['shots']==shots for r in merged),
                correct={a:sum(r['correct'][a] for r in part) for a in ARMS}))
    return rows,merged,summaries


def run():
    records=json.loads((OUT/'recordings.json').read_text(encoding='utf8'))
    result=json.loads(RESULT.read_text(encoding='utf8'))
    assert sha(OUT/'recordings.json')==result['artifacts_sha256'][(OUT/'recordings.json').relative_to(ROOT).as_posix()]
    rows,merged,summaries=audit(records)
    assert len(rows)==300 and len({r['reference_id'] for r in rows})==100
    for s in summaries:
        assert sum(s['categories'].values())==s['references']
        for a in ARMS:
            m=next(c['metrics'] for c in result['aggregates'] if (c['phase'],c['shots'],c['arm'])==(s['phase'],s['shots'],a))
            assert s['correct'][a]==m['correct'] and s['categories']['matched']==m['matched']
    write(DIAGNOSTIC,dict(schema='roam_native_continuous_v1_interval_diagnostic',
        source_sha256=sha(Path(__file__)),result_sha256=sha(RESULT),recordings_sha256=sha(OUT/'recordings.json'),
        reference_budget_rows=300,independent_references=100,rows=rows,multi_cue_intervals=merged,summaries=summaries,
        posthoc_geometric_diagnostic=True,native_inference_rerun=False,thresholds_changed=False,
        interpretation='Overlap-only categories partition every saved reference. Continuous active-to-active cue changes and multi-cue detection intervals are visible; this does not prove a causal mechanism or physiological boundary truth. No target-based detector fix or selection is performed.'))
    print(json.dumps([s for s in summaries if s['phase']=='all' and s['shots']==2]))


if __name__=='__main__':run()
