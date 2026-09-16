"""Audit named Stage3/4 pairs without promoting proxy families to historical ones."""
import argparse,csv,hashlib,json
from pathlib import Path

def load(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))

def pair(rows,a,b):
    return [r for r in rows if {r.get('family_a'),r.get('family_b')}=={a,b} or {r.get('A'),r.get('B')}=={a,b}]

def run(results,output):
    ep=results/'error_complementarity.csv';errors=load(ep);ip=results/'interaction_results.csv';interactions=load(ip)
    stage3=[]
    for req,a,b in (('X1 vs Frequency','F1_X1H','F4_Spectral'),('X1 vs F2 Spatial Coordination','F1_X1H','F2b_CSP')):
        found=pair(errors,a,b);stage3.append({'requirement':req,'family_a':a,'family_b':b,'rows':len(found),
            'run_ids':sorted({r['run_id'] for r in found}),'status':'verified_reference_pair' if found else 'missing',
            'boundary':'Native held-out/source-OOF predictions, but current reference families are not missing historical validated implementations.'})
    g5=[r for r in errors if r.get('family_a')=='validated_G5_bout_mean' and r.get('family_b')=='validated_G5_plus_full_bout_DTW']
    stage3.append({'requirement':'G5 vs Temporal/DTW','rows':len(g5),'run_ids':sorted({r['run_id'] for r in g5}),
        'status':'verified_conditional_increment_not_standalone_DTW' if g5 else 'missing','boundary':'Exact validated G5 baseline versus G5+complete-bout DTW increment; not a standalone DTW model and uses oracle bout boundaries.'})
    ring=[r for r in errors if r.get('family_a')=='F3_Ring_population' and r.get('family_b')=='F3_Ring_local_anchor']
    stage3.append({'requirement':'RLCS vs Personal Anchor','rows':0,'run_ids':[],'status':'missing_exact_historical_RLCS',
        'reference_proxy_rows':len(ring),'boundary':'Reference envelope-ring population/local-anchor rows cannot substitute for missing historical validated RLCS.'})
    raw=[r for r in errors if '_raw' in r.get('family_a','') or '_raw' in r.get('family_b','')]
    stage3.append({'requirement':'Raw vs robust features','rows':len(raw),'run_ids':sorted({r['run_id'] for r in raw}),
        'status':'verified_reference_pairs','boundary':'Current reference raw/robust families; historical equivalence remains unproven.'})
    stage4=[]
    for req,a,b in (('X1 x Frequency','F1_X1H','F4_Spectral'),('X1 x SpatialCoordination','F1_X1H','F2b_CSP')):
        found=pair(interactions,a,b);stage4.append({'requirement':req,'rows':len(found),'run_ids':sorted({r['run_id'] for r in found}),
            'status':'verified_reference_pair' if found else 'missing','boundary':'Source-prespecified current reference implementations; exact historical equivalence unavailable.'})
    audit={'completion_proven':False,'stage3_document_lines':[658,678],'error_table_sha256':hashlib.sha256(ep.read_bytes()).hexdigest(),
        'stage3':stage3,'stage4_document_lines':[685,712],'interaction_table_sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),
        'stage4_reviewed_priority_pairs':stage4,'remaining_stage4_exact_pairs':['RLCS x PersonalAnchor','RLCS x SessionSignature','G5 x TemporalShape','PersonalAnchor x SpatialCoordination','Quality x robust families'],
        'boundary':'Recorded pair metrics and source hashes verified; full named-pair and historical-family coverage remains incomplete.'}
    output.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps({'stage3_rows':sum(x['rows'] for x in stage3),'stage4_rows':sum(x['rows'] for x in stage4),'completion_proven':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.results,a.output)
