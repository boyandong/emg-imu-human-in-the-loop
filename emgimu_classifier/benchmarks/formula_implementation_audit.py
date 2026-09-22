"""Reviewed formula/source boundaries; runtime dimensions do not prove research completion."""
import argparse
import ast
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.families import (
    LocalDetailFamily, ScalePatternFamily, TraceCovarianceFamily, CspSpatialFamily,
    SpdTangentFamily, RingGeometryFamily, SpectralStateFamily, TemporalFormFamily,
    BodyContextFamily, QualityFamily,
)
from emgimu.feature_bank.ring_covariance import RawRingCovarianceFamily
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily
from emgimu.feature_bank.temporal import PathSignatureFamily
from emgimu.feature_bank.relative_spectrum import LogBandEnergyFamily
from emgimu.feature_bank.validated_unibo import ValidatedUniBoFamily
from emgimu.feature_bank.document_signal import RestNoiseLocalDetailFamily,DocumentCspFamily
from emgimu.feature_bank.body_frame import CalibratedBodyContextFamily
from emgimu.feature_bank.spd_anchor import SpdTangentPersonalAnchor

# Decisions are human-readable reviewed boundaries, never inferred from dimensions/tests.
# Exact historical mandatory reuse cannot be replaced by a conceptual candidate.
REVIEWS = (
 ('F6_calibrated_candidate','F6a. IMU body-frame context','body_frame.py','CalibratedBodyContextFamily','candidate_api_native_unavailable',
  'Explicit neutral gravity and guided/measured forward axis establish fixed calibration-relative frame; real IMU rate/units and trial provenance required. Causal gravity EMA and linear acceleration RMS present. Calibration trials rejected in held-out evaluation. Public native calibrated frame evaluation unavailable; no absolute yaw.','15'),
 ('F0_noise_candidate','F0. Local / Traditional Signal Detail','document_signal.py','RestNoiseLocalDetailFamily','candidate_formula',
  'Six metrics with thresholds frozen exclusively from native Rest adjacent-difference noise; active contraction magnitude cannot set thresholds. Historical extra R0 features still unavailable.','6C'),
 ('F2b_document_candidate','F2b. CSP-like spatial feature','document_signal.py','DocumentCspFamily','candidate_formula',
  'Uncentered XX transpose/(trace+epsilon), source-only one-vs-rest generalized eigenproblem, source-fixed gamma and top2/bottom2, log variance normalized plus epsilon. Native held-out wearing Core experiment available.','2 H min(2,floor(C/2))'),
 ('F0','F0. Local / Traditional Signal Detail','families.py','LocalDetailFamily','partial',
  'Six requested metrics present; thresholds fit pooled source differences rather than explicit calibration noise; old R0 retention cannot be proven without old artifacts.','6C'),
 ('F1','F1. Scale–Pattern / X1-H','families.py','ScalePatternFamily','reference_only',
  'Matches conceptual RMS/global RMS pattern; mandatory validated X1-H reuse missing. Family identifier is historical-looking but does not establish equivalence.','C'),
 ('F2a','F2a. Trace-normalized covariance','families.py','TraceCovarianceFamily','candidate_formula',
  'Centered sample covariance, fixed .05 shrinkage, epsilon diagonal, trace normalization and sqrt(2) off-diagonal vectorization; source-only fit.','C(C+1)/2'),
 ('F2b','F2b. CSP-like spatial feature','families.py','CspSpatialFamily','partial',
  'Source-only one-vs-rest generalized eigensystem; uses centered/shrunk covariance instead of stated uncentered XX transpose. Default one tail component versus suggested two.','2 H min(tails,floor(C/2))'),
 ('F2c','F2c. SPD / Riemannian tangent feature','families.py','SpdTangentFamily','candidate_formula',
  'Explicit permitted log-Euclidean training reference; whitened matrix log and sqrt(2) vech; not geometric-mean claim.','C(C+1)/2'),
 ('F3a','F3a. RLCS','families.py','RingGeometryFamily','reference_only',
  '25ms smoothed rectification, lag mean/std correlations; validated old envelope/aggregation missing; circular topology assumed by legacy class.','2 floor(C/2) block'),
 ('F3b','F3b. CES','families.py','RingGeometryFamily','reference_only',
  'Normalized sorted correlation eigenvalues are candidate CES block; validated old reuse missing; bundled with lag/ringcov blocks.','C block'),
 ('F3c','F3c. Ring-relative covariance','ring_covariance.py','RawRingCovarianceFamily','candidate_formula',
  'New independent raw F2a covariance block with verified ring contract; legacy RingGeometryFamily instead uses envelope covariance and must remain a distinct proxy.','6 floor(C/2)'),
 ('F4a','F4a. Frequency coordination','families.py','SpectralStateFamily','reference_only',
  'Hann periodogram, four bands strictly below Nyquist and channel L2 band vectors; exact historical Frequency implementation unavailable.','BC block'),
 ('F4b','F4b. Spectral summary','families.py','SpectralStateFamily','candidate_formula',
  'Periodogram total power/centroid/MDF and optional entropy normalized by log(number of bins); FFT power units not calibrated physical PSD.','4C block'),
 ('F4c','F4c. Cepstral / CCA-like candidate','families.py','SpectralStateFamily','candidate_formula',
  'Non-DC low-order unnormalized DCT-II mean/std across channels, K=4 prespecified; cepstral summary not reproduction of paper CCA.','2K block'),
 ('F4d','F4d. Personal/session-relative spectral shift','relative_spectrum.py','RelativeSpectrumCoordinates','partial',
  'Calibration-only mean log-band subtraction implemented; session-minus-long mean requires explicit separate references; context not fatigue.','BC'),
 ('F5a','F5a. Existing G5','validated_unibo.py','ValidatedUniBoFamily','validated_reuse_narrow',
  'Calls unchanged native UniBo G5 at 4ch/processed200Hz: early-minus-late and raw waveform slope; different from new TemporalFormFamily late-minus-early/envelope slope.','5C native G5'),
 ('F5_reference','F5. Temporal Form','families.py','TemporalFormFamily','reference_only',
  'New seven-channel metrics and map velocity; log early/late, normalized entropy/time differ from optional conceptual examples; cannot be called original G5.','7C+1'),
 ('F5b','F5b. DTW / template distance','temporal.py','TemporalTemplateFamily','partial',
  'DTW requires CompleteSequenceBatch, explicit full coverage and finite native durations >=1s; compressed bin rate cannot prove completeness. Legacy sparse MANUS runner refuses new execution; full UniBo bout replay preserved. Native boundary validity and stream segmentation remain separate.','H'),
 ('F5c','F5c. Low-order path signature（可选）','temporal.py','PathSignatureFamily','optional_candidate',
  'Per-time L2 rectified path; centered start, levels1/2, no absolute time; scale-normalized raw rectification rather than smoothed envelope.','C+C squared'),
 ('F6a','F6a. IMU body-frame context','families.py','BodyContextFamily','partial',
  'Real accel/gyro magnitude summaries and mean gravity direction present; calibration body-frame transform, gravity lowpass and linear acceleration RMS absent. No stable absolute yaw claimed.','13 IMU block'),
 ('F6b','F6b. Public dataset posture context','families.py','BodyContextFamily','candidate_formula',
  'Training-fixed posture one-hot, unseen category rejection; explicit oracle context, never fabricated IMU.','P posture block'),
 ('F7','F7. Personal Anchor Coordinates','calibration.py','PersonalAnchor','partial',
  'Mean/median prototypes, Euclidean/source-or-cal standardized/cosine distances, fixed-cal similarity and margins; optional shrinkage Mahalanobis absent. Separate frozen-source SPD tangent candidate below is not an affine-invariant geodesic.','2H+2'),
 ('F7_SPD_tangent_candidate','F7. Personal Anchor Coordinates','spd_anchor.py','SpdTangentPersonalAnchor','candidate_native_partial',
  'Frozen source-fitted F2c log-tangent reference; one equal-weight mean per native calibration trial, calibration-only class prototypes and Frobenius-equivalent tangent Euclidean distances. EPN held-out candidate study at1/2/5 shots shows mixed validation/final performance; no0-shot personal anchor. Incremental multi-family Core value, affine-invariant geodesic, historical equivalence and own-device validity remain unproven.','2H+2'),
 ('F8','F8. Session Signature','calibration.py','SessionSignature','partial',
  'Residual norms, cosines and pair geometry exact generic block; family-specific scale/RLCS/spectral/SPD/quality summaries are separate partial study evidence, not complete in this API.','2H+H(H-1)/2'),
 ('F9_legacy','F9. Quality / Observability','families.py','QualityFamily','reference_only',
  'Legacy flatline is fraction of flat edges rather than longest run; unknown ADC yields zero without mask; optional low-frequency observation unavailable. Preserve legacy measurements.','6C+5'),
 ('F9v2','F9. Quality / Observability','quality_observability.py','QualityObservabilityFamily','partial',
  'Adds longest consecutive flat edges/T, source thresholds, correlation anomaly, availability masks and conditional pre-highpass ratio. Legacy quality mask remains unchanged: new observation not automatically a validated fusion gate.','9C+8'),
 ('CAL_A','A. Personal normalization','calibration.py','PersonalNormalizer','candidate_formula',
  'Explicit rest median and active absolute Q95; raw/cal branches retained in integrated runs, source/current normalization comparison can decline.','C centers+C scales; output EMG unchanged shape'),
 ('CAL_C','C. Personal natural force envelope','activation_profile.py','PersonalActivationProfile','candidate_formula',
  'Raw global RMS q10/q50/q90 equal-trial empirical CDF plus within-gesture pattern spread; exclude Rest, archive signal units, not measured force.','3 quantiles plus C pattern mean and spread per native gesture'),
 ('CAL_D_E','D. Personal feature reliability','calibration.py','ReliabilityWeights','partial',
  'Between/within separation log-softmax and n0/(n0+N) shrinkage implemented; tau/n0 source CV exists only selected protocols, no global source-CV coverage claim.','K weights'),
 ('SESSION','七、Session Calibration v0','session_pipeline.py','SessionCalibrationPipeline','partial',
  'Current rest/scale/quality/signature/local prototypes preserve long profile; controlled native wearing0/1 only, 2/5 unsupported per domain, current device/calendar-day validation missing.','two-family branch-specific state'),
 ('FUSION','八、Late Fusion：精确定义','calibration.py','late_fusion','partial',
  'Nonnegative weighted probabilities and clipped quality renormalization; reject-all fallback retains valid distribution but explicit Unknown output absent here. Empirical source probability calibration audited for specified runs only.','H output probabilities'),
)


def build(document, output):
    lines = document.read_text(encoding='utf-8').splitlines()
    source_dir = Path('src/emgimu/feature_bank')
    rng = np.random.default_rng(20260916)
    x = rng.normal(size=(16,40,8)); labels = np.arange(16)%4
    batch = FeatureBatch(x,200,imu=rng.normal(size=(16,10,6)),posture=np.array(['up','down']*8))
    measured = {}
    for factory in (LocalDetailFamily,ScalePatternFamily,TraceCovarianceFamily,CspSpatialFamily,
                    SpdTangentFamily,RingGeometryFamily,SpectralStateFamily,TemporalFormFamily,
                    BodyContextFamily,QualityFamily,PathSignatureFamily,LogBandEnergyFamily,
                    lambda:RawRingCovarianceFamily(ring_topology=True),
                    lambda:QualityObservabilityFamily(ring_topology=True),
                    lambda:RestNoiseLocalDetailFamily(rest_label=2),DocumentCspFamily):
        family = factory().fit(batch,labels); before = pickle.dumps(family)
        values = family.transform(batch)
        if values.shape!=(16,len(family.feature_names)) or not np.isfinite(values).all():
            raise AssertionError('dimension/name/finite contract failure')
        if before!=pickle.dumps(family): raise AssertionError('transform changed fitted state')
        measured[type(family).__name__] = {'fixture_dimension':values.shape[1],
            'names':list(family.feature_names),'finite':True,'source_immutable':True}
    native = FeatureBatch(x[:,:,:4],200)
    family = ValidatedUniBoFamily('G5').fit(native,labels)
    values = family.transform(native)
    measured['ValidatedUniBoFamily'] = {'fixture_dimension':values.shape[1], 'names':list(family.feature_names),
                                      'fixture_override':'native four-channel processed200Hz G5'}
    neutral=np.zeros((50,6));neutral[:,2]=9.81
    body=CalibratedBodyContextFamily(imu_sample_rate_hz=50,acceleration_unit='synthetic acceleration units',
        angular_velocity_unit='synthetic angular velocity units').fit(batch,neutral_calibration_imu=neutral,
        forward_axis_device=np.array([1.,0.,0.]),calibration_trial_ids=['synthetic-neutral-source','synthetic-guided-axis-source'])
    before=pickle.dumps(body)
    values=body.transform(batch,trial_ids=[f'synthetic-target-{i}' for i in range(batch.windows)])
    if before!=pickle.dumps(body) or not np.isfinite(values).all():raise AssertionError('Calibrated body fixture failed')
    measured['CalibratedBodyContextFamily']={'fixture_dimension':values.shape[1],'names':list(body.feature_names),
        'fixture_override':'explicit synthetic50Hz IMU with synthetic neutral/guided-axis source; not native anatomical validation',
        'source_immutable':True,'native_evaluation':'N/A'}
    spd_anchor = SpdTangentPersonalAnchor(SpdTangentFamily().fit(batch.take(np.arange(8))))
    spd_anchor.fit(batch.take(np.arange(8,12)),labels[8:12])
    before=pickle.dumps(spd_anchor)
    spd_values=spd_anchor.transform(batch.take(np.arange(12,16)))
    if before!=pickle.dumps(spd_anchor) or spd_values.shape!=(4,len(spd_anchor.feature_names)) or not np.isfinite(spd_values).all():
        raise AssertionError('Frozen-source SPD tangent Anchor fixture failed')
    measured['SpdTangentPersonalAnchor']={'fixture_dimension':spd_values.shape[1],
        'names':list(spd_anchor.feature_names),'source_immutable':True,
        'fixture_override':'synthetic source8, disjoint calibration4 and evaluation4 for API shape only; no native performance claim',
        'native_evaluation':'N/A'}
    rows = []
    for item_id,heading,filename,symbol,status,boundary,dimension in REVIEWS:
        matches = [i+1 for i,text in enumerate(lines) if i+1>=1120 and text.strip()==heading]
        if len(matches)!=1: raise ValueError(f'Exact heading not unique: {heading}: {matches}')
        path = source_dir/filename; text = path.read_text(encoding='utf-8')
        nodes = [node for node in ast.parse(text).body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name==symbol]
        if len(nodes)!=1: raise ValueError(f'Source symbol ambiguous: {path}:{symbol}')
        node = nodes[0]
        rows.append({'item_id':item_id,'document_heading':heading,'document_line':matches[0],
            'document_sha256':hashlib.sha256(document.read_bytes()).hexdigest(),
            'source_path':path.as_posix(),'source_symbol':symbol,'source_line_start':node.lineno,
            'source_line_end':node.end_lineno,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'reviewed_status':status,'dimension_formula':dimension,
            'fixture_whole_class_dimension':measured.get(symbol,{}).get('fixture_dimension','N/A'),
            'boundary':boundary,'scientific_completion':'not_proven'})
    output.mkdir(parents=True,exist_ok=True)
    with (output/'FORMULA_IMPLEMENTATION_AUDIT.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    audit = {'completion_proven':False,'reviewed_rows':len(rows),
        'scope':'reviewed selected formula/API boundaries with exact source hashes and symbol spans; not exhaustive document acceptance',
        'fixture':'synthetic16 windows,40 samples,8channels,200Hz,4classes; IMU10samples and2 oracle postures only for dimension checks',
        'warning':'whole-class fixture dimensions are not subblock dimensions; synthetic dimension/immutability checks do not prove native topology or scientific validity',
        'measured_classes':measured}
    (output/'FORMULA_IMPLEMENTATION_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    report = ['# Formula implementation boundaries','',
        'This selected formula audit does not prove completion. Exact document headings and source symbol spans are in the CSV.',
        'Synthetic dimensions check interface shape only; they do not establish native topology, probability calibration or accuracy.','',
        '| Item | Reviewed status | Dimension | Boundary |','|---|---|---|---|']
    report += [f"| {r['item_id']} | {r['reviewed_status']} | {r['dimension_formula']} | {r['boundary']} |" for r in rows]
    (output/'FORMULA_IMPLEMENTATION_AUDIT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({'reviewed_rows':len(rows),'runtime_classes':len(measured),'completion_proven':False}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('document',type=Path); parser.add_argument('output',type=Path)
    args = parser.parse_args(); build(args.document,args.output)
