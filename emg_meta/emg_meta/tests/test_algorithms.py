from emgforce.algorithms import ALGORITHMS, META_CONV_LSTM, PERSONAL_MPF_TDS, get_algorithm


def test_two_switchable_algorithms_have_independent_remote_jobs() -> None:
    assert set(ALGORITHMS) == {META_CONV_LSTM, PERSONAL_MPF_TDS}
    conv = get_algorithm(META_CONV_LSTM); mpf = get_algorithm(PERSONAL_MPF_TDS)
    assert conv.remote_repository != mpf.remote_repository
    assert conv.tmux_session != mpf.tmux_session
    assert conv.input_channels == mpf.input_channels == 8
    assert conv.sample_rate_hz == mpf.sample_rate_hz == 2000
