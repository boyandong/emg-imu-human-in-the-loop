"""Frozen GUI lifecycle sources and complete numerical replay provenance."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gui_lifecycle_complete_recording_parity_has_scoped_evidence():
    protocol_path=ROOT/'benchmarks/song_real8/SONG_PERSONAL_GUI_V1_PROTOCOL.json'
    p=json.loads(protocol_path.read_text(encoding='utf8'))
    a=json.loads((ROOT/'feature_bank/SONG_PERSONAL_GUI_V1_ACCEPTANCE.json').read_text(encoding='utf8'))
    assert a['protocol_sha256']==sha(protocol_path)
    assert a['source_sha256']==p['source_sha256']
    for name,digest in a['source_sha256'].items():assert sha(REPO/name)==digest
    assert sha(ROOT/'benchmarks/song_real8/song_personal_session_v1/source_bank.pkl')==p['source_bank_sha256']
    assert sha(ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json')==p['offline_acceptance_sha256']
    for name,digest in p['profile_sha256'].items():
        assert sha(ROOT/'benchmarks/song_real8/song_personal_session_v1'/name)==digest
    previous=json.loads((ROOT/'feature_bank/SONG_GUI_V2_ACCEPTANCE.json').read_text(encoding='utf8'))
    expected=next(r['emissions'] for r in previous['records'] if r['session']=='S04')
    assert len(a['records'])==2 and {r['arm'] for r in a['records']}=={'population','session_5shot'}
    for record in a['records']:
        assert record['windows']==expected and record['maximum_probability_error']<1e-12
        assert record['confirmation_exact'] and record['source_and_profiles_immutable']
        assert record['personal_trials']==(0 if record['arm']=='population' else 20)
        assert record['session_trials']==(0 if record['arm']=='population' else 20)
    assert p['consecutive_frames']==2 and p['window_samples']==50 and p['hop_samples']==10
    assert not a['physical_validation_proven'] and not a['completion_proven'] and not a['default_promoted']
    assert 'not native trial efficacy' in a['scope'] and 'no fitting' in p['scope']
