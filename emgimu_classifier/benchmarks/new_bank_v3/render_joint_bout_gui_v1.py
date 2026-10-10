"""Render the actual versioned Qt entry with Windows fonts and saved profiles.

Run under the collection Python with QT_QPA_PLATFORM=windows. The window need
not be shown; rendering uses real Qt controls and an actual classifier process.
This loads already saved software-fixture profiles, never fits or scores data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[3]
APP=REPO/'collection/emg_meta/emg_meta'
sys.path.insert(0,str(APP))
sys.path.insert(0,str(APP/'third_party/generic-neuromotor-interface'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('personal','session-profile','output','receipt'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont,QFontDatabase
    import main_decision_v4
    app=QApplication([]);app.setFont(QFont('Microsoft YaHei UI',10))
    if app.platformName()!='windows' or 'Microsoft YaHei UI' not in QFontDatabase.families():
        raise RuntimeError('Actual Windows font rendering required')
    target=Path(args.output)
    window=main_decision_v4.create_window(target.parent/'render_project')
    page=window.realtime_inference_page;p=page.personal_session_panel;errors=[]
    try:
        p.model_choice.setCurrentIndex(p.model_choice.findData('seven_joint'))
        p.python.setText('D:/miniconda/python.exe');p.user.setText('fixture');p.session.setText('current')
        p.temporal_personal.setText(args.personal);p.temporal_session.setText(args.session_profile)
        p.load_backend();p.worker.failed.connect(errors.append)
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            app.processEvents()
            if errors:raise RuntimeError(errors[-1])
            if p.state and p.state['temporal']['session_profile_id']:break
            time.sleep(.01)
        if not p.state or not p.state['temporal']['session_profile_id']:raise RuntimeError('Joint profile reload timed out')
        p.ensurePolished();p.resize(1450,p.sizeHint().height());p.layout().activate();app.processEvents()
        if not p.grab().save(str(target)):raise RuntimeError('Panel render failed')
        receipt=dict(schema='joint_bout_gui_v1_windows_render',platform=app.platformName(),font=app.font().family(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            panel_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            schema_loaded=p.state['schema'],personal_trials=p.state['personal_trials'],session_trials=p.state['session_trials'],
            personal_profile_id=p.state['temporal']['personal_profile_id'],session_profile_id=p.state['temporal']['session_profile_id'],
            actual_Qt_controls_and_classifier_process=True,new_fit_or_native_scoring=False,
            physical_validation_proven=False)
        Path(args.receipt).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
        print('Rendered actual Windows-font joint panel with both saved profiles')
    finally:
        if not page.shutdown():raise RuntimeError('Render worker failed to stop')
        window.close();app.processEvents()


if __name__=='__main__':main()
