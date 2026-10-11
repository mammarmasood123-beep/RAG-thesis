"""Scoring uses HotpotQA's official answer EM/F1; parsing mirrors SIM-RAG's regex."""
import pytest

from eval.metrics import em, f1, is_unsure, parse_reasoner_output


@pytest.mark.parametrize("pred,gold,score", [
    ("The Beatles", "beatles", 1),
    ("Beatles.", "the Beatles", 1),
    ("Paul McCartney", "McCartney", 0),
])
def test_em(pred, gold, score):
    assert em(pred, gold) == score


def test_f1_partial_and_yes_no_rule():
    assert f1("Paul McCartney", "McCartney") == pytest.approx(2 / 3)
    assert f1("no", "no") == 1.0
    assert f1("yes", "no") == 0.0
    assert f1("yes it is", "yes") == 0.0  # official rule: yes/no gold needs exact normalised match


def test_parse_standard():
    assert parse_reasoner_output("Answer: Paris\nRationale: capital of France") == \
        ("Paris", "capital of France", True)


def test_parse_blank_lines_between_fields():
    assert parse_reasoner_output("Answer: Paris\n\n\nRationale: x")[:2] == ("Paris", "x")


def test_parse_failure_gives_empty_answer_not_placeholder():
    # SIM-RAG returns "Answer not found", which can score F1 > 0; we return "".
    assert parse_reasoner_output("I think it is Paris.") == ("", "", False)


def test_unsure_detection():
    assert is_unsure("Unsure.") and is_unsure("unsure") and not is_unsure("Paris")
