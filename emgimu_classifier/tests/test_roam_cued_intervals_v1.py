"""Native sample identity and explicit boundary/label semantics."""
import io
import zipfile
import numpy as np
import pytest
from emgimu.datasets.roam_cued_intervals_v1 import load_roam_cued_intervals,take_roam_intervals,CHANNELS


def archive_fixture(labels,*,missing_channel=False,bad_clock=False):
    x=np.arange(len(labels)*8,dtype=np.float32).reshape(-1,8)/100
    columns=CHANNELS[:-1] if missing_channel else CHANNELS
    text=','.join((*columns,'gt','time_elapsed'))+'\n'
    text+=''.join(','.join([*[str(x[i,j]) for j in range(len(columns))],str(c),str(i/(2000 if bad_clock else 200))])+'\n'
                  for i,c in enumerate(labels))
    blob=io.BytesIO()
    with zipfile.ZipFile(blob,'w') as z:z.writestr('fixture.csv',text)
    blob.seek(0)
    return zipfile.ZipFile(blob),x


def test_full_native_cue_intervals_preserve_every_sample_and_separate_labels():
    archive,x=archive_fixture(np.repeat([0,1,2,0],250))
    with archive:data=load_roam_cued_intervals(archive,'fixture.csv')
    np.testing.assert_array_equal(np.concatenate(data.batch.sequences),x)
    assert data.batch.starts==(0,250,500,750)
    assert list(data.labels.values())==['relax','open','close','relax']
    assert not hasattr(data.batch,'labels') and all(':cue' in t for t in data.batch.trial_ids)
    assert not data.receipt['resampled'] and not data.receipt['biological_boundaries_proven']
    selected=take_roam_intervals(data,(data.batch.trial_ids[2],data.batch.trial_ids[0]))
    assert selected.batch.starts==(500,0) and list(selected.labels.values())==['close','relax']
    np.testing.assert_array_equal(selected.batch.sequences[0],x[500:750])


def test_short_cue_is_excluded_explicitly_without_interpolating_samples():
    archive,x=archive_fixture(np.r_[np.zeros(250,int),np.ones(100,int),np.full(250,2)])
    with archive:data=load_roam_cued_intervals(archive,'fixture.csv')
    assert data.batch.starts==(0,350) and len(data.receipt['excluded_intervals'])==1
    assert data.receipt['excluded_intervals'][0]['samples']==100
    np.testing.assert_array_equal(data.batch.sequences[1],x[350:])


@pytest.mark.parametrize('failure',['channel','clock','label'])
def test_invalid_native_contract_is_rejected(failure):
    labels=np.repeat([0,1,2],250)
    if failure=='label':labels[0]=3
    archive,_=archive_fixture(labels,missing_channel=failure=='channel',bad_clock=failure=='clock')
    with archive,pytest.raises(ValueError):load_roam_cued_intervals(archive,'fixture.csv')
