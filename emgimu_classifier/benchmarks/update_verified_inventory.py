"""Record archives whose hashes were verified during acquisition."""
import json
from pathlib import Path

VERIFIED = {
 'libemg_electrode_shift': ('libemg_electrode_shift/CIILData-main.zip',11563732,'4CFA9A4861193F230179FA87D53EDA7503E66B5CFEDEAE81C81E476F53F6B2B6','711DF74C47DFE811968B8F41A37E2438'),
 'unibo_inail': ('unibo_inail/unibo-inail-main.zip',513385791,'E32F1649B7B9B775A6B03328FB862EBF21A18C7CA6B4D9ABB50B7281CAEFBCE3',None),
 'epn612': ('epn612/EMG-EPN612-Dataset.zip',5483385161,'4EE8DB037385E7BEE1E6AC6F9E9EEA4F0869E25F7825F5EB7E5BE0DFF4F93C21','98BD3C315EFAB607CC54B2ED2F8F3ADA'),
 'semg_manus': ('semg_manus/semg-manus-dataset-v1.zip',594438686,'2D9E1DF613485B9A4AFE9A5A71010CA84E75A521B596F836A195B5EAF3DC1F29','2D3975173CD6B6832D50D0BE0948E1F0'),
 'emg_fmg': ('emg_fmg/Data.zip',9125369365,'3021E8B5FF31FD5AABA5E11F5F1C4D370D07407C0F1451F36C64A3A03B42C677','453A9E17B5A2EE4C5E8DA683D4DF2C29'),
 'libemg_force': ('libemg_force/contraction-intensity-main.zip',207707607,'237212D87F50C9B621E572BC780D99FAC3A9AF60B99EC8B455FB3D3DBB2D15E7','61D4A5AAEA134F3C730517A354E20870'),
}

if __name__ == '__main__':
    path=Path(__file__).parent/'discovery/DATASET_MANIFEST.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    manifest['generated_at']='2026-09-15'
    for dataset in manifest['datasets']:
        if dataset['id'] in VERIFIED:
            rel,size,sha,md5=VERIFIED[dataset['id']]
            dataset.update(status='downloaded_verified',path=f"{manifest['raw_root']}/{rel}",size=size,sha256=sha)
            if md5: dataset['md5']=md5
        if dataset['id']=='ds2_force':
            dataset.update(status='blocked_original_release_404',note='Historical artifacts absent; original Kaggle page and download API returned 404. No substitute is represented as historical DS2.')
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
