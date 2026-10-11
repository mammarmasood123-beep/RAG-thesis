"""Tests for the two Task-6 additions: per-(N, T) hygiene table and answer-leakage flag."""
import pytest

from eval.aggregate import hygiene_by_cell
from eval.metrics import answer_in_text


# ---------------------------------------------------------------- 6b answer leakage
def test_leak_detects_normalised_answer():
    assert answer_in_text("Gillian Anderson", "...Dana Scully's (Gillian Anderson) separation...") is True


def test_leak_normalisation_articles_punct_case():
    assert answer_in_text("The Beatles", "a band, the BEATLES!, formed") is True


def test_leak_absent():
    assert answer_in_text("Gillian Anderson", "John Dean was White House Counsel.") is False


def test_leak_requires_whole_tokens():
    # "anna" must not match inside "annapolis"
    assert answer_in_text("Anna", "He moved to Annapolis.") is False
    assert answer_in_text("Anna", "He met Anna there.") is True


def test_leak_skips_yes_no_and_empty_answers():
    assert answer_in_text("yes", "yes it is") is None
    assert answer_in_text("No", "no") is None
    assert answer_in_text("the", "the the the") is None   # normalises to empty


# ---------------------------------------------------------------- 6a hygiene table
def _rec(arm, n, rnd, parse_ok=True, unsure=False, fits=True):
    return {"condition": {"arm": arm, "N": n}, "round": rnd,
            "output": {"parse_ok": parse_ok, "said_unsure": unsure} if fits else None}


def test_hygiene_by_cell_rates_and_keys():
    recs = [_rec("closed_book", None, 0, parse_ok=False),
            _rec("closed_book", None, 0),
            _rec("distractor", 4, 3, parse_ok=False, unsure=True),
            _rec("distractor", 4, 3),
            _rec("distractor", 4, 3, unsure=True),
            _rec("distractor", 4, 3),
            _rec("distractor", 0, 2, fits=False)]
    rows = {(r["N"], r["T"]): r for r in hygiene_by_cell(recs)}
    assert rows[("closed_book", 0)] == {"N": "closed_book", "T": 0, "n": 2, "not_fit": 0,
                                         "parse_fail": 1, "parse_fail_rate": 0.5,
                                         "said_unsure": 0, "said_unsure_rate": 0.0}
    r = rows[(4, 3)]
    assert (r["n"], r["parse_fail"], r["said_unsure"]) == (4, 1, 2)
    assert r["parse_fail_rate"] == pytest.approx(0.25) and r["said_unsure_rate"] == pytest.approx(0.5)
    assert rows[(0, 2)]["n"] == 0 and rows[(0, 2)]["not_fit"] == 1
    assert rows[(0, 2)]["parse_fail_rate"] is None
