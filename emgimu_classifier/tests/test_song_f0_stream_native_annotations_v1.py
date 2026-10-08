"""Independent native HDF5 interval oracle; optional only when data is absent."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/song_real8'


@pytest.mark.parametrize('session',['S03','S04'])
def test_native_full_window_stable_intervals_exclude_unannotated_emissions(session):
    import h5py
    p=json.loads((HERE/'SONG_F0_STREAM_V1_PROTOCOL.json').read_text(encoding='utf8'))
    path=Path(p['source_folder'])/f'2026-09-18_{session}'/'session.h5'
    if not path.is_file():pytest.skip('Native Song archive unavailable; software/readback tests remain separate')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==p['hdf5_sha256'][session]
    with h5py.File(path) as f:
        rows=f['trials'][:];samples=f['streams/emg/raw'].shape[0]
        assert f['streams/emg/raw'].shape[1]==8 and int(f['meta'].attrs['emg_nominal_rate_hz'])==250
    classes=['neutral','index_pinch','fist','open_hand']
    with np.load(HERE/'SONG_F0_STREAM_V1_EMISSIONS.npz',allow_pickle=False) as a:
        ends=a[session+'_ends'];expected=np.full(len(ends),-1,int);trial=np.full(len(ends),-1,int)
        assert np.all(ends<samples)
        for row in rows:
            if row['trial_kind']!=b'formal' or not bool(row['valid']) or row['completion_status']!=b'completed':continue
            start,stop=int(row['stable_start_sample']),int(row['stable_end_sample'])
            if not 0<=start<stop<=samples:continue
            selected=(ends-49>=start)&(ends<stop)
            assert not np.any(trial[selected]>=0)
            label=row['label'].decode('utf8')
            hand=[c for c in classes if label.endswith(c)]
            assert len(hand)==1
            expected[selected]=classes.index(hand[0]);trial[selected]=int(row['trial_id'])
        np.testing.assert_array_equal(a[session+'_truth'],expected)
        np.testing.assert_array_equal(a[session+'_trials'],trial)
        assert np.any(expected<0) and np.any(expected>=0)
