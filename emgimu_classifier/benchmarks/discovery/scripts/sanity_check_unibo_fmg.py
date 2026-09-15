from pathlib import Path
from io import BytesIO
import argparse
import json
import zipfile
import numpy as np
from scipy.io import loadmat
from sanity_check_tier1_archives import qc, plot


def run(unibo: Path, fmg: Path, output: Path) -> None:
    output.mkdir(parents=True,exist_ok=True)
    reports={'unibo_inail':[], 'emg_fmg':[]}
    for subject in (1,4,7):
        for posture in (1,2):
            path=unibo/f'user_{subject}_day_1_posture_{posture}.mat'
            data=loadmat(path);values=data['emg']
            if values.shape[1]!=4 or not np.all(np.isfinite(values)):
                raise ValueError(f'invalid UniBo EMG {path}')
            labels=data['label'].reshape(-1)
            if len(labels)!=len(values):raise ValueError('label alignment')
            svg=output/f'unibo_s{subject}_p{posture}_raw_envelope_psd.svg'
            plot(svg,values[:2500],500)
            reports['unibo_inail'].append({**qc(values,str(path),500),'labels':np.unique(labels).tolist(),'plot':svg.name})
    with zipfile.ZipFile(fmg) as archive:
        for subject in (1,14,27):
            for load in (0,1000):
                member=f'Data/Par {subject}/Power/{load}/Power {load} 1.csv'
                # Case and spacing are preserved in the archive.
                candidates=[name for name in archive.namelist() if name.startswith(f'Data/Par {subject}/Power/{load}/') and name.endswith(f' {load} 1.csv')]
                if len(candidates)!=1:raise ValueError(f'expected unique trial: {member}')
                member=candidates[0]
                values=np.loadtxt(BytesIO(archive.read(member)),delimiter=',',skiprows=1,usecols=range(8,16))
                if values.shape!=(36000,8) or not np.all(np.isfinite(values)):raise ValueError(member)
                svg=output/f'emg_fmg_s{subject}_load{load}_raw_envelope_psd.svg'
                plot(svg,values[:10000],2000)
                reports['emg_fmg'].append({**qc(values,member,2000),'load_g':load,'position':1,'plot':svg.name})
    for dataset,samples in reports.items():
        (output/f'{dataset}_sanity.json').write_text(json.dumps({'dataset':dataset,'status':'ok',
            'samples':samples,'plot_unit':'first five seconds, channel 1; full trial QC',
            'saturation_definition':'fraction near observed extrema; not hardware clipping diagnosis'},indent=2),encoding='utf-8')
    print(json.dumps({'status':'ok','trials_checked':12,'plots':12}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--unibo',type=Path,required=True)
    parser.add_argument('--fmg',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.unibo,args.fmg,args.output)
