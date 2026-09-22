"""Build required delivery schemas while preserving hashes of unchanged source records."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict

SCHEMAS={
    'feature_family_results.csv':('dataset','subject','session/domain','feature_family','calibration_budget','condition','macro_f1','accuracy','log_loss','brier','ece'),
    'conditional_incremental.csv':('dataset','subject','session/domain','core_bank','added_family','condition','delta_logloss','delta_macro_f1','delta_brier'),
    'error_complementarity.csv':('dataset','subject','session/domain','family_a','family_b','error_correlation','disagreement_rate','a_correct_b_wrong','a_wrong_b_correct'),
    'calibration_curve.csv':('dataset','subject','session/domain','condition','shots_per_class','feature_bank','method','macro_f1','log_loss'),
    'ablation_full_bank.csv':('dataset','subject','session/domain','condition','shots_per_class','feature_bank','method','macro_f1','log_loss'),
}
ALIASES={'disagreement_rate':('disagreement','disagreement_fraction'),'calibration_budget':('shots_per_class',),
         'feature_family':('family',),'delta_logloss':('delta_log_loss',),'core_bank':('core',),
         'added_family':('added',),'family_a':('A',),'family_b':('B',)}


def record_hash(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def load_exclusions(results):
    path=results/'scientific_exclusions.json'
    if not path.exists():return []
    return json.loads(path.read_text(encoding='utf-8'))['rules']


def exclusion_notes(rules, name, row):
    return [rule['reason'] for rule in rules
            if rule['run_id']==row['run_id'] and rule['source_artifact']==name
            and row.get(rule['field']) in rule['excluded_values']]


def build(results,output):
    output.mkdir(parents=True,exist_ok=True);sources={};datasets=defaultdict(set);audits={}
    exclusion_rules=load_exclusions(results)
    for name in SCHEMAS:
        with (results/name).open(encoding='utf-8-sig',newline='') as h:sources[name]=list(csv.DictReader(h))
        for row in sources[name]:
            if row.get('dataset'):datasets[row['run_id']].add(row['dataset'])
    def manifest(run,seen=None):
        seen=set() if seen is None else seen
        if run in seen:return {}
        seen.add(run)
        path=results/'manifests'/f'{run}__run_manifest.json'
        meta=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        parent_run=meta.get('source_run') or meta.get('reused_source_run')
        if isinstance(parent_run,str) and parent_run.startswith('feature_bank_'):
            parent=manifest(parent_run,seen)
            # Reused training states may come from another phase: never inherit its target domain.
            permitted=('families',) if parent.get('phase')!=meta.get('phase') else (
                'families','target_session','target_days','target_domains','target_conditions','evaluation_force')
            for key in permitted:
                if key not in meta and key in parent:meta[key]=parent[key]
        return meta
    for name,required in SCHEMAS.items():
        canonical=[];missing=Counter();by_run=defaultdict(Counter)
        for index,row in enumerate(sources[name],1):
            run=row['run_id'];meta=manifest(run);notes={};values={}
            exclusions=exclusion_notes(exclusion_rules,name,row)
            if exclusions:notes['scientific_acceptance_excluded']=exclusions
            for field in required:
                value=row.get(field,'')
                if not value:
                    for alias in ALIASES.get(field,()):
                        if row.get(alias):value=row[alias];notes[field]=f'source alias: {alias}';break
                if not value and field=='dataset' and len(datasets[run])==1:
                    value=next(iter(datasets[run]));notes[field]='same-run recorded dataset'
                if not value and field in ('a_correct_b_wrong','a_wrong_b_correct') and row.get('evaluation_unit')=='whole_native_trial_mean':
                    numerator_key=('core_correct_increment_wrong' if field=='a_correct_b_wrong'
                                   else 'core_wrong_increment_correct')
                    if row.get(numerator_key) and row.get('evaluation_trials'):
                        numerator=int(row[numerator_key]);denominator=int(row['evaluation_trials'])
                        if denominator<=0 or numerator<0 or numerator>denominator:
                            raise ValueError(f'Invalid native trial complementarity counts: {run}')
                        value=str(numerator/denominator)
                        notes[field]=f'exact native trial count ratio: {numerator_key}/evaluation_trials'
                if not value and field=='session/domain':
                    if row.get('condition') and row['condition']!='ALL':
                        value=row['condition'];notes[field]='recorded condition/domain; not an inferred session ID'
                    else:
                        for key in ('target_session','target_days','target_domains','target_conditions','evaluation_force','source_condition'):
                            if meta.get(key) is not None:
                                value=json.dumps({key:meta[key]},ensure_ascii=False,sort_keys=True);notes[field]='run manifest evaluation domain';break
                if not value and field in ('calibration_budget','shots_per_class') and run in (
                        'feature_bank_manus_nested_oof_20260915','feature_bank_epn_nested_oof_20260915','feature_bank_force_nested_oof_20260915'):
                    value='0';notes[field]='source-user OOF; no target personal calibration, distinct from source probability calibration'
                if not value and field=='feature_bank':
                    families=meta.get('families')
                    if isinstance(families,(list,tuple)) and families:
                        value='|'.join(families);notes[field]='manifest family bank; method/removal controls still apply'
                if not value:
                    value='N/A';notes[field]='not recoverable from inspected source fields/manifest'
                    missing[field]+=1;by_run[run][field]+=1
                values[field]=value
            canonical.append({'run_id':run,'source_artifact':name,'source_row_1based':index,
                              'source_record_sha256':record_hash(row),**values,
                              'metadata_notes_json':json.dumps(notes,ensure_ascii=False,sort_keys=True)})
        with (output/name).open('w',encoding='utf-8',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(canonical[0]));w.writeheader();w.writerows(canonical)
        audits[name]={'rows':len(canonical),'required_fields':required,'missing_field_counts':dict(missing),
                      'missing_by_run':{run:dict(counts) for run,counts in by_run.items()},
                      'source_csv_sha256':hashlib.sha256((results/name).read_bytes()).hexdigest(),
                      'canonical_csv_sha256':hashlib.sha256((output/name).read_bytes()).hexdigest()}
    audit={'status':'schema_complete_evidence_partial' if any(a['missing_field_counts'] for a in audits.values()) else 'ok',
           'artifacts':audits,'scope':'five required schemas; exact source record identities; no invented metrics/session IDs; N/A requires source repair or capability boundary',
           'warning':'presence of required columns does not establish scientific requirement completion'}
    (output/'SCHEMA_AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    return verify(results,output)


def verify(results,output):
    audit=json.loads((output/'SCHEMA_AUDIT.json').read_text(encoding='utf-8'));count=0
    exclusion_rules=load_exclusions(results)
    for name,meta in audit['artifacts'].items():
        if hashlib.sha256((results/name).read_bytes()).hexdigest()!=meta['source_csv_sha256']:raise ValueError('Changed source table')
        if hashlib.sha256((output/name).read_bytes()).hexdigest()!=meta['canonical_csv_sha256']:raise ValueError('Changed canonical table')
        with (results/name).open(encoding='utf-8-sig',newline='') as h:source=list(csv.DictReader(h))
        with (output/name).open(encoding='utf-8',newline='') as h:canonical=list(csv.DictReader(h))
        if len(source)!=len(canonical):raise ValueError('Canonical row coverage changed')
        for index,(a,b) in enumerate(zip(source,canonical),1):
            if json.loads(b['metadata_notes_json']).get('scientific_acceptance_excluded',[])!=exclusion_notes(exclusion_rules,name,a):
                raise ValueError('Scientific exclusion annotation mismatch')
            if b['source_record_sha256']!=record_hash(a) or int(b['source_row_1based'])!=index or b['run_id']!=a['run_id']:
                raise ValueError('Canonical provenance mismatch')
            for field in SCHEMAS[name]:
                if a.get(field) and a[field]!=b[field]:raise ValueError('Canonical changed a recorded value')
            count+=1
    result={'status':'ok','rows_checked':count,'schema_evidence_status':audit['status'],
            'scope':'all canonical record identities and original recorded values; schema missingness audited separately'}
    (output/'PROVENANCE_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('output',type=Path);p.add_argument('--verify',action='store_true')
    a=p.parse_args();verify(a.results,a.output) if a.verify else build(a.results,a.output)
