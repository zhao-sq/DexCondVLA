from scripts.rescore_bench2dex_task26_without_object_upright import _first_stable_frame
from scripts.rescore_bench2dex_task26_without_object_upright import _max_consecutive_true


def test_first_stable_frame_returns_end_of_first_dwell_window() -> None:
    values = [False, True, True, False, True, True, True]

    assert _first_stable_frame(values, 3) == 6
    assert _first_stable_frame(values, 4) is None


def test_max_consecutive_true() -> None:
    assert _max_consecutive_true([False, True, True, False, True]) == 2
    assert _max_consecutive_true([]) == 0
