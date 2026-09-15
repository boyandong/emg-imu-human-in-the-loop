from pathlib import Path
import argparse
import csv
import html
import hashlib
import json
import zipfile

PROTOCOLS=(
 ('EPN personal fine-tune','feature_bank_epn612_trial_finetune_final_20260915',6,5.,False,False,'cross_user'),
 ('MANUS session fine-tune','feature_bank_manus_calibration_final_20260915',6,10.,True,False,'session_3'),
 ('Force ProductMode','feature_bank_force_product_final_20260915',7,3.,False,True,'ALL'),
 ('Force source-only anchors','feature_bank_force_zero_final_20260915',7,3.,False,False,'ALL'),
)
CORE_PROTOCOLS=(
 ('Wearing session local-anchor control','feature_bank_wearing_session_final_20260916',5,1.,True,False,'ALL','population_plus_local_anchor'),
 ('Force Core source-temp anchor','feature_bank_force_core_temperature_final_20260916',7,3.,False,False,'ALL','Core'),
 ('MANUS Core source-temp anchor','feature_bank_manus_core_temperature_final_20260916',6,10.,True,False,'ALL','Core'),
 ('Force concat Core anchor','feature_bank_force_concat_core_calibration_final_20260915',7,3.,False,False,'ALL','Core'),
 ('Force Core + Spectral anchor','feature_bank_force_concat_core_calibration_final_20260915',7,3.,False,False,'ALL','Core+F4_Spectral'),
 ('MANUS concat Core anchor','feature_bank_manus_concat_core_calibration_final_20260916',6,10.,True,False,'ALL','Core'),
 ('MANUS Core + Temporal anchor','feature_bank_manus_concat_core_calibration_final_20260916',6,10.,True,False,'ALL','Core+F5_Temporal'),
 ('MANUS Core + IMU anchor','feature_bank_manus_concat_core_calibration_final_20260916',6,10.,True,False,'ALL','Core+F6_IMU'),
)


def recorded_cost(root,run,shots,raw_root,evidence,cache):
    path=root/run/'split_trial_ids.json'
    splits=json.loads(path.read_text())
    if raw_root is None:return None
    entries=[s for s in splits if s['shots']==shots]
    totals=[]
    for entry in entries:
        total=0.
        for trial in entry['calibration']:
            if trial not in cache:
                if run.startswith('feature_bank_force'):
                    native=raw_root/'libemg_force/official/ContractionIntensity-main'/trial.split('_')[0]/(trial+'.csv')
                    data=native.read_bytes();rate=1000.
                elif run.startswith('feature_bank_wearing'):
                    native=raw_root/'libemg_electrode_shift/CIILData-main.zip'
                    with zipfile.ZipFile(native) as handle:data=handle.read(trial)
                    rate=200.
                else:
                    native=raw_root/'semg_manus/semg-manus-dataset-v1.zip'
                    with zipfile.ZipFile(native) as handle:data=handle.read(trial)
                    rate=200.
                samples=sum(bool(line.strip()) and not line.lstrip().startswith(b'#') for line in data.splitlines())
                cache[trial]=samples/rate
                evidence[trial]={'native_path':str(native),'zip_member':trial if not run.startswith('feature_bank_force') else None,
                    'sha256':hashlib.sha256(data).hexdigest(),'samples':samples,'nominal_rate_hz':rate,'signal_seconds':samples/rate}
            total+=cache[trial]
        totals.append(total)
    if not totals:raise AssertionError('Missing recorded calibration splits')
    return sum(totals)/len(totals),min(totals),max(totals),hashlib.sha256(path.read_bytes()).hexdigest()


def build(root:Path,output:Path,raw_root:Path|None=None,local_root:Path|None=None)->None:
    rows=[];curves=[];evidence={};cache={};sources={}
    protocols=tuple((*p,None) for p in PROTOCOLS)+CORE_PROTOCOLS
    for label,run,classes,duration,session,target_force,condition,model in protocols:
        run_root=root if (root/run).is_dir() or local_root is None else local_root
        sources[run]=hashlib.sha256((run_root/run/'calibration_curve.csv').read_bytes()).hexdigest()
        with (run_root/run/'calibration_curve.csv').open(encoding='utf-8-sig',newline='') as handle:
            source=[r for r in csv.DictReader(handle) if r['subject']=='ALL' and r['condition']==condition
                and (model is None or (run.startswith('feature_bank_wearing') and r['method'] in (model,'unsupported_budget'))
                     or (r.get('model')==model and r['method'] in ('with_anchor','unsupported','unsupported_5_shot')))]
        points=[]
        for shots in (0,1,2,5):
            cell=next((r for r in source if int(r['shots_per_class'])==shots),None)
            supported=cell is not None and bool(cell.get('macro_f1',''))
            cost=recorded_cost(run_root,run,shots,raw_root,evidence,cache) if model is not None and supported else None
            rows.append({'protocol':label,'run_id':run,'shots_per_class':shots,'supported':supported,
                'classes':classes,'calibration_trials':classes*shots if supported else 'N/A',
                'estimated_signal_seconds':cost[0] if cost else classes*shots*duration if supported else 'N/A',
                'estimate_basis':'native sample counts / nominal rate; mean user signal duration; preparation/transitions excluded' if cost else f'{duration:g} seconds/trial approximate; preparation and transitions excluded',
                'each_target_session':session,'all_task_gestures_required':True,'target_force_required':target_force,
                'target_posture_required':'not controlled','macro_f1':cell.get('macro_f1','') if supported else '',
                'model':model or 'historical protocol','minimum_user_signal_seconds':cost[1] if cost else 'N/A',
                'maximum_user_signal_seconds':cost[2] if cost else 'N/A','calibration_split_sha256':cost[3] if cost else 'N/A'})
            if supported:points.append((shots,float(cell['macro_f1'])))
        curves.append((label,points))
    with (output/'calibration_burden.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="590" viewBox="0 0 1200 590">',
         '<rect width="1200" height="590" fill="white"/>',
         '<g font-family="Arial" font-size="15" fill="#222">',
         '<text x="80" y="35" font-size="21">Independent performance vs calibration budget</text>']
    def xy(x,y):return 90+x/5*560,450-(y-.2)/.6*350
    for tick in (.2,.3,.4,.5,.6,.7,.8):
        _,y=xy(0,tick);svg.extend([f'<path d="M90 {y}H650" stroke="#ddd"/>',f'<text x="45" y="{y+5}">{tick:.1f}</text>'])
    for tick in (0,1,2,5):
        x,_=xy(tick,.2);svg.append(f'<text x="{x-5}" y="475">{tick}</text>')
    svg.extend(['<path d="M90 100V450H650" fill="none" stroke="#222"/>',
        '<text x="230" y="510">Labelled trials per task class</text>',
        '<text x="15" y="85">Macro-F1</text>'])
    for i,(label,points) in enumerate(curves):
        color=('#2166ac','#b2182b','#1b7837','#762a83','#e08214','#4d9221','#c51b7d','#008837','#666666','#a6611a','#018571','#984ea3')[i]
        coordinates=' '.join(f'{x:.3f},{y:.3f}' for x,y in (xy(a,b) for a,b in points))
        svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        for a,b in points:
            x,y=xy(a,b);svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
        svg.append(f'<text x="690" y="{140+i*35}" fill="{color}">{html.escape(label)}</text>')
    svg.extend(['<text x="80" y="550">Tasks and aggregation differ. Budget changes also change remaining evaluation trials.</text>',
        '<text x="80" y="575">Unsupported budgets have no plotted point; curves do not establish live-device accuracy.</text>','</g></svg>'])
    (output/'performance_vs_calibration_budget.svg').write_text(''.join(svg),encoding='utf-8')
    (output/'calibration_burden_audit.json').write_text(json.dumps({'status':'ok','cost_rows':len(rows),
        'curve_sources_sha256':sources,'native_calibration_trials':evidence,
        'limits':'sample/rate estimates measure recorded signal, not observed human calibration flow or hardware wall time; legacy protocol durations remain approximate; unlike tasks and budget-specific evaluation trials'},indent=2),encoding='utf-8')
    print(f'created {len(rows)} calibration-cost rows and standalone SVG')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--raw-root',type=Path)
    p.add_argument('--local-root',type=Path)
    a=p.parse_args();build(a.root,a.output,a.raw_root,a.local_root)
