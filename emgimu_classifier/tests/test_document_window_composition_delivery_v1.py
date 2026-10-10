"""Frozen study and actual dtype semantics; no training or native rerun."""
import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from benchmarks.song_real8.verify_document_window_composition_v2 import check

ROOT=Path(__file__).resolve().parents[1]


def test_saved_document_composition_read_only_oracles_and_rebuild_equivalence():
    saved=json.loads((ROOT/'feature_bank/DOCUMENT_WINDOW_COMPOSITION_V1_ACCEPTANCE.json').read_text(encoding='utf8'))
    assert check()==saved
    assert saved['matched_existing_probability_cells']==60
    assert saved['existing_source_classifier_parameter_max_error']==0.
    assert saved['added_joint_F2_family_removal_cells']==4
    assert saved['independent_rebuild_is_not_a_new_accuracy_gain'] and not saved['primary_pass']


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_direct_scaler_oracle_preserves_installed_parameter_cast_semantics(dtype):
    source=np.random.default_rng(84).normal(size=(100,9))*np.geomspace(1e-8,1e8,9)
    scaler=StandardScaler().fit(source)
    query=(np.random.default_rng(87).normal(size=(15,9))*np.geomspace(1e-8,1e8,9)).astype(dtype)
    actual=query.copy();actual-=scaler.mean_.astype(actual.dtype);actual/=scaler.scale_.astype(actual.dtype)
    np.testing.assert_array_equal(actual,scaler.transform(query))
