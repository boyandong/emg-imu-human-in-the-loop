"""Transparent retrospective metadata scores; never change frozen decisions."""
import csv
from pathlib import Path

CRITERIA=('A_failure_relevance','B_product_similarity','C_factor_isolation','D_interaction_value',
          'E_calibration_value','F_raw_quality','G_accessibility','H_processing_cost')
# Current metadata judgement, not a reconstruction of the undocumented old totals.
# Unknown repetition/quality/size lowers confidence and is stated explicitly.
CARDS={
 'Historical DS2 force':((5,2,4,2,5,2,0,2),'Publication protocol only; historical identity and raw archive missing. Current access unresolved; no native QC or reliable storage estimate.'),
 'LibEMG Contraction Intensity':((5,5,4,3,4,4,3,4),'Verified native sample/QC; clear intensity conditions, eight-channel cuff; day labels not reliably encoded. Ramp source has four trials/class; repository license undeclared.'),
 'LibEMG Electrode Shift':((5,5,4,3,2,4,3,5),'Verified native sample/QC; five Before and two trials/class/After-domain. Only0/1-shot per wearing domain. Calendar days not encoded; repository license undeclared.'),
 'UniBo-INAIL':((5,2,4,5,3,4,5,4),'Verified native sample/QC; day by posture factorial, four targeted muscles not eight-ring. Calibration capability follows native independent trials, not window count.'),
 'EMG-EPN612':((5,5,4,1,5,4,5,3),'Verified native sample/QC; cross-user and many labelled trials; no multi-session nuisance protocol. Five-GB JSON archive is manageable by subset streaming.'),
 'sEMG-MANUS':((5,5,3,5,3,3,5,4),'Verified sample/QC with known cohort/trial-count anomalies. Session/speed partly confounded; three speed trials/class/session cannot support5-shot.'),
 'EMG-FMG load and limb position':((5,3,5,5,2,4,5,2),'Verified sample/QC and orthogonal load/position labels. External load is not voluntary force; per-cell calibration repetition support needs explicit inventory. Large raw expansion avoided by streaming.'),
 'GREAT':((5,3,4,4,3,2,4,3),'Metadata-only independent day/posture design;16 channels. No native archive QC or independent repetition inventory yet.'),
 'NinaPro DB6':((5,3,4,5,5,2,2,2),'Metadata-only repeated-day and morning/afternoon design,14 electrodes. Account/terms and actual download size remain unresolved; no native QC.'),
 'GRABMyo':((5,3,4,4,5,2,5,2),'Metadata-only multi-session cohort/seven trials per gesture.28 usable of32 channels, higher storage cost; no native QC.'),
 'NinaPro DB5':((4,4,3,2,3,2,2,3),'Metadata-only two-Myo sensing, no strong multi-day design. Repetition inventory, account/terms and actual size remain to verify.'),
 'FORS-EMG':((4,2,3,2,3,2,2,2),'Metadata-only orientation/coarse placement regions, not controlled re-donning. Native repetition layout, license, access and archive size unverified.'),
 'Three-position electrode replacement':((5,5,4,3,3,2,5,5),'Metadata-only controlled array positions and315MB official archive; native repetitions and QC not yet checked.'),
 'Hyser':((5,1,4,5,4,2,5,0),'Metadata-only real-force/HD observability resource.256 channels and roughly143GB full collection; calibrated subsets would require separate design. No native QC.'),
}


def review(root):
    path=root/'DATASET_CANDIDATES.csv'
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
    if set(r['dataset'] for r in rows)!=set(CARDS):raise ValueError('Candidate inventory changed')
    original=[(r['dataset'],r['score'],r['decision'],r['reason']) for r in rows]
    for row in rows:
        values,basis=CARDS[row['dataset']]
        assert len(values)==8 and all(isinstance(v,int) and 0<=v<=5 for v in values)
        row.update({f'score_review_{key}':value for key,value in zip(CRITERIA,values)})
        row['score_review_total']=sum(values)
        row['score_review_basis']=basis
        row['score_review_status']='retrospective_metadata_judgement; native verification confidence stated in basis'
        if row['dataset']=='Historical DS2 force':
            row['license']='CC-BY-4.0 (publication; archive identity unverified)'
    assert original==[(r['dataset'],r['score'],r['decision'],r['reason']) for r in rows]
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Candidate score review','',
      'Scores are retrospective engineering judgements over recorded metadata. They do not reconstruct original selection scores or prove that grading preceded training.',
      'Original score, decision, reason and frozen experimental protocols remain unchanged. Reviewed totals are unweighted sums; no model selection uses them.',
      'Metadata-only or uncertain native repetition/quality/access information lowers confidence. Numeric scores are not measurements of accuracy.','',
      '| Criterion | 0 | 3 | 5 |','|---|---|---|---|',
      '| A failure relevance | no relevant labelled factor | indirect/context evidence | direct intended nuisance manipulation |',
      '| B product similarity | unrelated sensing | sparse forearm with material channel/topology differences | native eight-channel forearm ring/cuff |',
      '| C factor isolation | unlabelled/confounded | partial factor separation | clear orthogonal labelled factors |',
      '| D interaction value | no nuisance interaction design | limited multiple-factor evidence | labelled multi-factor repeated design |',
      '| E calibration value | no independent calibration samples | limited/uncertain within-cell repetitions | many independent repetitions for budgets |',
      '| F raw quality | unusable/no raw evidence | metadata or documented native anomalies | broad verified native integrity (not established here) |',
      '| G accessibility | release unavailable/unresolved | public repository but license/access constraints | stable official open archive/terms |',
      '| H processing cost | disproportionate full collection | manageable archive/subset cost | small/easy native files |','',
      'Intermediate integers describe graded proximity to anchors. F=4 denotes verified sampled native QC, not full-population quality.',
      'Primary source URLs and exact reviewed vectors are in DATASET_CANDIDATES.csv. Missing native validation remains explicit.',
      'DS2 publication license source: [original descriptor](https://www.mdpi.com/2306-5729/10/12/194); this does not identify the historical archive.']
    (root/'SCORE_REVIEW.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(f'{len(rows)} retrospective eight-criterion scorecards; frozen decisions unchanged')


if __name__=='__main__':review(Path('benchmarks/discovery'))
