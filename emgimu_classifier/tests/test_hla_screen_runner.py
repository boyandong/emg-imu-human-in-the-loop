import importlib.util

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch neural extra is not installed",
)


def test_single_subject_bootstrap_is_valid_json_null() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "tools" / "run_hla_neural_screen.py"
    spec = importlib.util.spec_from_file_location("hla_screen_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module._bootstrap(np.asarray([0.1]), 42) == [None, None]
