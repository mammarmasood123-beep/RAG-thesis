"""Paired bootstrap over questions (N=0 vs N=4 at the same round), 10,000 resamples."""
import numpy as np
import pytest

from eval.aggregate import PAIRED_N_BOOT, paired_bootstrap


def _cells(a, b, t=3):
    return {(4, t): {f"q{i}": (x, float(x)) for i, x in enumerate(a)},
            (0, t): {f"q{i}": (x, float(x)) for i, x in enumerate(b)}}


def test_resample_count_is_10000():
    assert PAIRED_N_BOOT == 10_000


def test_identical_conditions_give_zero_width_interval():
    c = _cells([1, 0, 1, 1, 0], [1, 0, 1, 1, 0])
    r = paired_bootstrap(c, 4, 0, 3, metric=0)
    assert (r["diff"], r["lo"], r["hi"], r["n"]) == (0.0, 0.0, 0.0, 5)


def test_pairing_is_by_question_not_by_position():
    # same marginal accuracy (3/6) but every question flips: paired diffs are +1/-1
    a = [1, 1, 1, 0, 0, 0]
    b = [0, 0, 0, 1, 1, 1]
    r = paired_bootstrap(_cells(a, b), 4, 0, 3, metric=0)
    assert r["diff"] == 0.0 and r["lo"] < -0.3 and r["hi"] > 0.3


def test_only_common_questions_are_paired_and_result_is_deterministic():
    c = _cells([1, 1, 0, 0], [0, 0, 0, 0])
    del c[(0, 3)]["q0"]
    r1 = paired_bootstrap(c, 4, 0, 3, metric=0)
    assert r1["n"] == 3 and r1["diff"] == pytest.approx(1 / 3)
    assert r1 == paired_bootstrap(c, 4, 0, 3, metric=0)


def test_uniform_shift_is_recovered():
    rng = np.random.default_rng(0)
    b = rng.integers(0, 2, 150).tolist()
    a = [0] * 150                      # N=4 wrong everywhere
    r = paired_bootstrap(_cells(a, b), 4, 0, 3, metric=0)
    assert r["diff"] == pytest.approx(-np.mean(b)) and r["lo"] <= r["diff"] <= r["hi"]
