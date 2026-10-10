"""Freeze software-only full-action GUI/capture/estimated-boundary verification."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    parent=HERE/'SONG_EXTENDED_GUI_V1_PROTOCOL.json';p=json.loads(parent.read_text())
    source=dict(p['source_sha256'])
    paths=[
        'emgimu_classifier/src/emgimu/feature_bank/temporal_bout_live_v1.py',
        'emgimu_classifier/src/emgimu/feature_bank/personal_session_stream_v5.py',
        'emgimu_classifier/src/emgimu/feature_bank/personal_temporal_bouts_v1.py',
        'emgimu_classifier/src/emgimu/feature_bank/complete_bout_window_adapter_v1.py',
        'emgimu_classifier/src/emgimu/feature_bank/document_path_v3.py',
        'emgimu_classifier/src/emgimu/feature_bank/temporal.py',
        'emgimu_classifier/src/emgimu/feature_bank/autonomous_bouts_v1.py',
        'emgimu_classifier/src/emgimu/feature_bank/personal_session_stream_v1.py',
        'collection/emg_meta/emg_meta/emgforce/inference/personal_session_worker.py',
        'collection/emg_meta/emg_meta/emgforce/ui/personal_session_panel.py',
        'emgimu_classifier/tests/test_personal_session_stream_v5.py',
        'collection/emg_meta/emg_meta/emgforce/inference/personal_session_worker_v5.py',
        'collection/emg_meta/emg_meta/emgforce/ui/main_window_v4.py',
        'collection/emg_meta/emg_meta/emgforce/ui/realtime_inference_page_v5.py',
        'collection/emg_meta/emg_meta/main_decision_v3.py',
        'collection/emg_meta/emg_meta/install_decision_shortcut_v3.ps1',
        'collection/emg_meta/emg_meta/tests/test_decision_page_v5.py',
        'collection/emg_meta/emg_meta/tests/test_decision_launcher_v3.py',
        'emgimu_classifier/benchmarks/song_real8/prepare_temporal_live_gui_v1.py']
    source.update({name:sha(ROOT/name) for name in paths})
    for name,digest in source.items():assert sha(ROOT/name)==digest,name
    artifacts={'emgimu_classifier/'+name:digest for name,digest in p['artifact_sha256'].items()}
    native=ROOT/'emgimu_classifier/feature_bank/PERSONAL_TEMPORAL_UNIBO_ACCEPTANCE_V1.json'
    artifacts[native.relative_to(ROOT).as_posix()]=sha(native)
    for name,digest in artifacts.items():assert sha(ROOT/name)==digest,name
    result=dict(schema='temporal_live_gui_v1_protocol',parent_protocol_sha256=sha(parent),
        source_sha256=source,artifact_sha256=artifacts,sample_rate_hz=250,channels=8,
        temporal_bins=32,window_samples=50,hop_samples=10,full_action_seconds=[1,30],
        branches=['off','manual_cued','estimated_auto'],temporal_mixture=.25,
        fixed_detector_policy=dict(onset_s=.08,release_s=.12,preroll_s=.10,min_duration_s=1,max_duration_s=30),
        rest_rule='Fit detector only to separately labelled neutral calibration samples; never fit/update from prediction frames. Prefer current-session Rest detector when a matching current-session profile exists.',
        capture_rule='User-marked complete cued intervals, including onset/offset; no settling or stable-only crop. Raw and causally filtered contiguous samples, starts and separate labels retained. Completeness is a user declaration, not physiological ground truth.',
        verification='Deterministic eight-channel raw fixtures exercise actual frozen source model, parent window parity, independent interval replay, irregular chunks, whole-action registration/persistence, session parents, raw quality Unknown, gap/overflow/censoring, all acquisition signal connections and real Qt subprocess. Entire collection regression plus focused classifier checks. No synthetic gesture accuracy is claimed.',
        output_rule='Window predictions continue separately. Full-action output only after explicit end or confirmed release; automatic boundaries remain estimated. Gap/pause/model switch discards open interval, never manufactures an ending.',
        default_model='six',default_temporal_mode='off',default_promoted=False,physical_validation_proven=False,
        scope='Software-only eight-channel250Hz full-action interface and automatic boundary composition. Existing native UniBo four-channel full-bout guard remains negative; Song stable excerpts are not certified full actions. No new native/physical accuracy, throughput or clinical-fatigue claim. Public-data native automatic fusion outcome remains separate.')
    path=HERE/'TEMPORAL_LIVE_GUI_V1_PROTOCOL.json'
    with path.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
    print('Frozen full-action GUI software protocol; existing native accuracy scope unchanged')


if __name__=='__main__':prepare()
