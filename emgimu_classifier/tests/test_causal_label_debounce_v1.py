import pytest
from emgimu.feature_bank.causal_label_debounce_v1 import CausalLabelDebounceV1


def test_fixed_confirmation_independent_sequence_and_prefix():
    labels = [0, 0, 1, 0, 1, 1, 1, 2, 1, 2, 2]
    expected = [-1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
    control = CausalLabelDebounceV1()
    assert [control.update(v) for v in labels] == expected
    changed = CausalLabelDebounceV1()
    assert [changed.update(v) for v in labels[:6] + [2] * 5][:6] == expected[:6]
    immediate = CausalLabelDebounceV1(confirmations=1)
    assert [immediate.update(v) for v in labels] == labels


def test_invalid_input_preserves_pending_confirmation():
    control = CausalLabelDebounceV1()
    assert control.update(2) == -1
    for invalid in (True, 2.0, -1, 3, None):
        with pytest.raises(ValueError): control.update(invalid)
        assert (control.stable, control.candidate, control.count) == (-1, 2, 1)
    assert control.update(2) == 2
    for kwargs in ({'confirmations': 0}, {'confirmations': True}, {'classes': (0, 0)},
                   {'classes': ()}, {'classes': (0, 1.0)}):
        with pytest.raises(ValueError): CausalLabelDebounceV1(**kwargs)
