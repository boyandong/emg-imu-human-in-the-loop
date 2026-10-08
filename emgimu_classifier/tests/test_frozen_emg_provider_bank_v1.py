from dataclasses import replace
import hashlib
import inspect
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def package():
    evidence=json.loads((ROOT/'feature_bank/FROZEN_EMG_BANK_ACCEPTANCE_V1.json').read_text(encoding='utf8'))
    path=ROOT/evidence['package_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==evidence['package_sha256']
    return pickle.loads(path.read_bytes())


def batch(trials=1):
    x=np.arange(trials*4*40*8,dtype=np.float32).reshape(trials*4,40,8)%7
    ids=np.repeat([f'fixture_{i}' for i in range(trials)],4)
    return FeatureBatch(x,200.),ids


def test_portable_checkpoint_binds_all_sources_and_complete_native_replay():
    d=json.loads((ROOT/'feature_bank/FROZEN_EMG_BANK_ACCEPTANCE_V1.json').read_text(encoding='utf8'))
    for path,digest in d['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert d['source_parameters_exactly_recovered'] and d['source_state_immutable'] and d['source_trials']==2250
    assert (d['channels'],d['sample_rate_hz'],d['window_samples'])==(8,200,40) and not d['requires_IMU']
    assert (d['native_provider_cases'],d['native_fusion_cases'],d['native_omission_cases'])==(10,40,240)
    assert len({(r['user'],r['shots_per_class']) for r in d['records']})==len(d['records'])==40
    assert not d['default_promoted'] and not d['physical_validation_proven'] and not d['completion_proven']
    for row in d['records']:
        assert row['evaluation_trials']==120 and row['provider_max_probability_error']<1e-12
        assert row['calibration_weight_max_error']<1e-12 and row['fusion_max_probability_error']<1e-12
        assert len(row['omission_cases'])==6 and all(c['max_probability_error']<1e-12 for c in row['omission_cases'])


def test_inference_without_labels_or_IMU_preserves_source_state_and_trial_means(package):
    b,ids=batch(2);before=pickle.dumps(package)
    assert 'labels' not in inspect.signature(package.predict).parameters
    result=package.predict(b,ids,window_offsets=np.tile(np.arange(4),len(ids)//4),user_id='fixture_user')
    assert result['trial_ids']==('fixture_0','fixture_1') and result['probabilities'].shape==(2,6)
    np.testing.assert_allclose(result['probabilities'].sum(1),1.,atol=1e-12)
    assert set(result['predicted_labels'])<=set(package.class_names_)
    assert pickle.dumps(package)==before
    order=np.array([7,2,5,0,6,1,4,3])
    permuted=package.predict(b.take(order),ids[order],window_offsets=np.tile(np.arange(4),len(ids)//4)[order],user_id='fixture_user')
    np.testing.assert_allclose(result['probabilities'],permuted['probabilities'],atol=1e-12,rtol=0)
    readout=package.predict_providers(b,ids,window_offsets=np.tile(np.arange(4),len(ids)//4),user_id='fixture_user')
    single=package.predict(b,ids,window_offsets=np.tile(np.arange(4),len(ids)//4),user_id='fixture_user',available=('F0',))
    np.testing.assert_array_equal(single['probabilities'],readout['probabilities']['F0'])
    assert single['provider_names']==('F0',)


def test_calibration_user_and_package_identity_do_not_modify_fitted_source(package):
    b,ids=batch(6);labels={f'fixture_{i}':i for i in range(6)};before=pickle.dumps(package)
    state=package.calibrate_user(b,ids,labels,window_offsets=np.tile(np.arange(4),len(ids)//4),user_id='fixture_user',forbidden_evaluation_trials=('eval_fixture',))
    assert state.fusion_state.n_cal_trials==6 and state.fusion_state.alpha==pytest.approx(2/3)
    e,ei=batch();ei=np.full(len(ei),'eval_fixture')
    result=package.predict(e,ei,window_offsets=np.arange(4),user_id='fixture_user',user_state=state)
    np.testing.assert_allclose(result['weights'],state.fusion_state.weights,rtol=0,atol=1e-12)
    with pytest.raises(ValueError,match='user/bank'):package.predict(e,ei,window_offsets=np.arange(4),user_id='different_user',user_state=state)
    with pytest.raises(ValueError,match='user/bank'):package.predict(e,ei,window_offsets=np.arange(4),user_id='fixture_user',user_state=replace(state,bank_id='different'))
    with pytest.raises(ValueError,match='overlap'):package.predict(b,ids,window_offsets=np.tile(np.arange(4),len(ids)//4),user_id='fixture_user',user_state=state)
    assert pickle.dumps(package)==before


@pytest.mark.parametrize('case',['rate','channels','samples','blank_ids','source_ids','bad_user','unknown_provider','empty_provider','offset_duplicate','offset_gap','offset_shape','offset_float'])
def test_unlabeled_input_contract_rejects_invalid_calls(package,case):
    b,ids=batch();kwargs={'user_id':'fixture_user','window_offsets':np.tile(np.arange(4),len(ids)//4)}
    if case=='rate':b=FeatureBatch(b.emg,250.)
    elif case=='channels':b=FeatureBatch(b.emg[:,:,:7],200.)
    elif case=='samples':b=FeatureBatch(b.emg[:,:39],200.)
    elif case=='blank_ids':ids=np.full(len(ids),'')
    elif case=='source_ids':ids=np.full(len(ids),package.policy_.source_trials[0])
    elif case=='bad_user':kwargs['user_id']=''
    elif case=='unknown_provider':kwargs['available']=('unknown',)
    elif case=='empty_provider':kwargs['available']=()
    elif case=='offset_duplicate':kwargs['window_offsets']=np.array([0,1,1,3])
    elif case=='offset_gap':kwargs['window_offsets']=np.array([0,1,2,4])
    elif case=='offset_shape':kwargs['window_offsets']=np.arange(3)
    else:kwargs['window_offsets']=np.arange(4,dtype=float)
    with pytest.raises(ValueError):package.predict(b,ids,**kwargs)


@pytest.mark.parametrize('case',['label_coverage','unknown_label','evaluation_overlap','available','missing_class'])
def test_calibration_requires_known_labels_and_disjoint_trial_ids(package,case):
    b,ids=batch(6);labels={f'fixture_{i}':i for i in range(6)};kwargs={'user_id':'fixture_user','window_offsets':np.tile(np.arange(4),len(ids)//4)}
    if case=='label_coverage':del labels['fixture_0']
    elif case=='unknown_label':labels['fixture_0']=99
    elif case=='evaluation_overlap':kwargs['forbidden_evaluation_trials']=('fixture_0',)
    elif case=='available':kwargs['available']=('unknown',)
    else:labels['fixture_5']=0
    with pytest.raises(ValueError):package.calibrate_user(b,ids,labels,**kwargs)
