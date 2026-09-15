from pathlib import Path
import argparse
import csv
import html

PROTOCOLS=(
 ('EPN personal fine-tune','feature_bank_epn612_trial_finetune_final_20260915',6,5.,False,False,'cross_user'),
 ('MANUS session fine-tune','feature_bank_manus_calibration_final_20260915',6,10.,True,False,'session_3'),
 ('Force ProductMode','feature_bank_force_product_final_20260915',7,3.,False,True,'ALL'),
 ('Force source-only anchors','feature_bank_force_zero_final_20260915',7,3.,False,False,'ALL'),
)


def build(root:Path,output:Path)->None:
    rows=[];curves=[]
    for label,run,classes,duration,session,target_force,condition in PROTOCOLS:
        with (root/run/'calibration_curve.csv').open(encoding='utf-8-sig',newline='') as handle:
            source=[r for r in csv.DictReader(handle) if r['subject']=='ALL' and r['condition']==condition]
        points=[]
        for shots in (0,1,2,5):
            cell=next((r for r in source if int(r['shots_per_class'])==shots),None)
            supported=cell is not None and bool(cell.get('macro_f1',''))
            rows.append({'protocol':label,'run_id':run,'shots_per_class':shots,'supported':supported,
                'classes':classes,'calibration_trials':classes*shots if supported else 'N/A',
                'estimated_signal_seconds':classes*shots*duration if supported else 'N/A',
                'estimate_basis':f'{duration:g} seconds/trial approximate; preparation and transitions excluded',
                'each_target_session':session,'all_task_gestures_required':True,'target_force_required':target_force,
                'target_posture_required':'not controlled','macro_f1':cell.get('macro_f1','') if supported else ''})
            if supported:points.append((shots,float(cell['macro_f1'])))
        curves.append((label,points))
    with (output/'calibration_burden.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="590" viewBox="0 0 1000 590">',
         '<rect width="1000" height="590" fill="white"/>',
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
        color=('#2166ac','#b2182b','#1b7837','#762a83')[i]
        coordinates=' '.join(f'{x:.3f},{y:.3f}' for x,y in (xy(a,b) for a,b in points))
        svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        for a,b in points:
            x,y=xy(a,b);svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
        svg.append(f'<text x="690" y="{140+i*35}" fill="{color}">{html.escape(label)}</text>')
    svg.extend(['<text x="80" y="550">Tasks and aggregation differ. Budget changes also change remaining evaluation trials.</text>',
        '<text x="80" y="575">Unsupported budgets have no plotted point; curves do not establish live-device accuracy.</text>','</g></svg>'])
    (output/'performance_vs_calibration_budget.svg').write_text(''.join(svg),encoding='utf-8')
    print(f'created {len(rows)} calibration-cost rows and standalone SVG')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();build(a.root,a.output)
