"""Prompt-token counting and the pre-generation `fits` gate (bug found on the GPU pod, 2026-10-11).

transformers 5.x: apply_chat_template(..., tokenize=True) returns a BatchEncoding, so len() was 2
on every record and the fits gate always passed. The fix renders the template to text and counts
tokenizer.encode(text, add_special_tokens=False). A hard check in run_expA compares that count with
vLLM's len(prompt_token_ids) on every vllm-backend record.

Tests marked `llama3` need the real Llama-3 tokenizer in the local HF cache (HF_HOME) and are
skipped otherwise. Everything else runs anywhere, with no GPU and no vLLM.
"""
import json
import os

import pytest

from harness.context import build_context, messages
from harness.generator import GenResult, count_chat_tokens
from harness.run_expA import Runner, prompt_fits
from prompts.simrag import PROMPTS

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
SYSTEM = PROMPTS["simrag_force_answer_noabstain"]["text"]
PARA = ("Aubrieta (commonly known as Aubretia) is a genus of about 12 species of flowering plants in "
        "the cabbage family Brassicaceae. The genus is named after Claude Aubriet, a French "
        "flower-painter. It originates from southern Europe east to central Asia but is now a common "
        "garden escape throughout Europe.")


def _msgs(n_rounds, per_round=5):
    rounds = [[(f"Title {r}-{i}", PARA) for i in range(per_round)] for r in range(n_rounds)]
    return messages(SYSTEM, build_context("Psychopsis and Aubrieta are both genuses of which type of plants?",
                                          rounds, n_rounds))


# ---------------------------------------------------------------- fake tokenizer, transformers-5 behaviour
class FakeV5Tokenizer:
    """Mimics transformers 5: tokenize=True returns a dict-like BatchEncoding (len == 2)."""
    bos = "<|begin_of_text|>"

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=True, **kw):
        text = self.bos + "".join(f"<|{m['role']}|> {m['content']} <|eot|>" for m in msgs)
        text += " <|assistant|>" if add_generation_prompt else ""
        if tokenize:
            ids = self.encode(text, add_special_tokens=False)
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return text

    def encode(self, text, add_special_tokens=True):
        ids = list(range(len(text.split())))
        return ([999] + ids) if add_special_tokens else ids


def test_old_counting_returned_2_under_transformers5():
    tok = FakeV5Tokenizer()
    assert len(tok.apply_chat_template(_msgs(3), add_generation_prompt=True, tokenize=True)) == 2


def test_count_chat_tokens_counts_rendered_text_without_extra_specials():
    tok = FakeV5Tokenizer()
    m = _msgs(3)
    text = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    n = count_chat_tokens(tok, m)
    assert n == len(tok.encode(text, add_special_tokens=False)) and n > 100


# ---------------------------------------------------------------- the fits gate on its own
def test_prompt_fits_boundary():
    assert prompt_fits(7936, 256, 8192) is True
    assert prompt_fits(7937, 256, 8192) is False
    assert prompt_fits(2, 256, 8192) is True


# ---------------------------------------------------------------- hard check in the runner
class FakeVLLMGen:
    backend, model, params = "vllm", "fake-vllm", 0

    def __init__(self, engine_offset):
        self.engine_offset = engine_offset

    def count_prompt_tokens(self, msgs):
        return 100

    def count_text_tokens(self, text):
        return len(text.split())

    def generate(self, batch, metas):
        return [GenResult("Answer: x\nRationale: y", 100 + self.engine_offset, 5, 1.0, "test") for _ in batch]


ROW = {"qid": "qX", "question": "Q?", "answer": "x", "q_type": "bridge", "level": "hard",
       "gold_titles_canonical": ["A", "B"], "gold_order": ["A", "B"], "flipped": False, "pilot": True}


def _runner(tmp_path, gen):
    cfg = {"generator": {"backend": "stub", "model": "m", "params": 0, "dtype": "float16",
                         "max_model_len": 8192, "max_new_tokens": 256, "temperature": 0.0,
                         "top_p": 1.0, "seed": 0},
           "master_seed": 1, "prompt_id": "simrag_force_answer_noabstain", "log_prompts": False, "batch_size": 8,
           "_run_id": "test", "_config_hash": "h"}
    r = Runner(cfg, tmp_path)
    r.gen = gen
    return r


def test_runner_raises_on_count_mismatch_for_vllm_backend(tmp_path):
    r = _runner(tmp_path, FakeVLLMGen(engine_offset=1))
    with pytest.raises(RuntimeError) as e:
        r.closed_book([ROW])
    msg = str(e.value)
    assert "qX" in msg and "round=0" in msg and "100" in msg and "101" in msg


def test_runner_accepts_matching_counts_for_vllm_backend(tmp_path):
    r = _runner(tmp_path, FakeVLLMGen(engine_offset=0))
    r.closed_book([ROW])
    r.close()
    rec = json.loads((tmp_path / "records.jsonl").read_text().splitlines()[0])
    assert rec["context"]["prompt_tokens"] == rec["modules"][0]["input_tokens"] == 100


def test_stub_backend_is_not_checked(tmp_path):
    class StubLike(FakeVLLMGen):
        backend = "stub"
    r = _runner(tmp_path, StubLike(engine_offset=7))
    r.closed_book([ROW])          # no RuntimeError: stub counts are approximate by design
    r.close()


# ---------------------------------------------------------------- real Llama-3 tokenizer (offline)
def _load_llama3():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers is not installed")
    try:
        return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except Exception as e:  # noqa: BLE001 - any load failure means the files are not available here
        pytest.skip(f"{MODEL} tokenizer not in the local HF cache "
                    f"(HF_HOME={os.environ.get('HF_HOME', '<unset>')}): {type(e).__name__}")


@pytest.fixture(scope="module")
def llama3():
    return _load_llama3()


def test_llama3_count_is_realistic_and_matches_template_tokenisation(llama3):
    m = _msgs(4)                                            # 20 paragraphs, ~1.5K tokens
    n = count_chat_tokens(llama3, m)
    text = llama3.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    ids = llama3(text, add_special_tokens=False)["input_ids"]
    assert n > 500                                          # (a) far greater than 2
    assert n == len(ids)                                    # (b) equals ids of the templated text
    assert ids[0] == llama3.bos_token_id and ids.count(llama3.bos_token_id) == 1   # single BOS
    vllm_style = llama3.apply_chat_template(m, add_generation_prompt=True, tokenize=True, return_dict=False)
    assert n == len(vllm_style)                             # the path vLLM 0.29.0 llm.chat() uses


def test_llama3_overlong_prompt_is_flagged_not_fitting(llama3):
    m = _msgs(20, per_round=8)                              # 160 paragraphs (~12K tokens), well past 8192
    n = count_chat_tokens(llama3, m)
    assert n > 8192
    assert prompt_fits(n, 256, 8192) is False               # (c)
    assert prompt_fits(count_chat_tokens(llama3, _msgs(2)), 256, 8192) is True
