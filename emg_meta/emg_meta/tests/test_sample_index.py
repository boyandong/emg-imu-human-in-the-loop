from __future__ import annotations

import numpy as np

from emgforce.sample_clock import SampleIndexClock


def test_session_sample_index_starts_at_zero_without_compressing_received_samples() -> None:
    clock = SampleIndexClock()
    global_before, _ = clock.allocate(17)
    clock.begin_session()
    global_index, session_index = clock.allocate(4)
    np.testing.assert_array_equal(global_before, np.arange(17))
    np.testing.assert_array_equal(global_index, np.arange(17, 21))
    np.testing.assert_array_equal(session_index, np.arange(4))
    assert clock.session_index == 4

