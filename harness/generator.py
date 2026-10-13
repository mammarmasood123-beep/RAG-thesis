"""Generator backends. Both expose the same interface so the runner never branches on backend.

  count_prompt_tokens(messages) -> int    exact for vLLM (chat template applied); approximate for stub
  generate(batch_messages, metas) -> [GenResult]

StubGenerator is for CPU harness checks only. It is an *oracle*: it reads which titles are in
the prompt and is told each question's gold answer and gold titles through `metas`. Its
outputs are deterministic in the prompt bytes, so the N=0 output-identity check is exercised.
Every record it produces is tagged generator.backend = "stub". Its numbers mean nothing.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

_TITLE = re.compile(r"Title: (.*?) Content: ")


@dataclass
class GenResult:
    text: str
    input_tokens: int
    output_tokens: int
    wall_ms: float | None
    wall_source: str          # "vllm_request_metrics" | "batch_wall_div_n" | "stub"


def count_chat_tokens(tokenizer, messages: list[dict]) -> int:
    """Prompt tokens as the engine will see them, independent of the transformers version.
    transformers 5.x returns a BatchEncoding from apply_chat_template(tokenize=True), whose len()
    is 2, so render the template to text and count encode(text, add_special_tokens=False).
    The chat template already contains <|begin_of_text|>, so no special tokens are added."""
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return len(tokenizer.encode(text, add_special_tokens=False))


class StubGenerator:
    backend = "stub"
    model = "oracle-stub"
    params = 0

    def __init__(self, max_model_len: int = 8192, max_new_tokens: int = 256):
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens

    def count_prompt_tokens(self, messages: list[dict]) -> int:
        return sum(len(m["content"].split()) for m in messages) + 10  # +template overhead

    def count_text_tokens(self, text: str) -> int:
        return len(text.split())

    def _h(self, s: str) -> int:
        return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 100

    def generate(self, batch: list[list[dict]], metas: list[dict]) -> list[GenResult]:
        out = []
        for msgs, meta in zip(batch, metas):
            user = msgs[-1]["content"]
            present = set(_TITLE.findall(user))
            gold = set(meta["gold_titles"])
            n_noise = len(present - gold)
            h = self._h(user)
            if not present:                          # closed book: ~20% "memorised"
                correct = h < 20
            elif gold <= present:                    # both hops present: degrades with noise
                correct = h >= min(85, 15 + 3 * n_noise)
            else:                                    # one hop only
                correct = h < 30
            ans = meta["gold_answer"] if correct else f"wrong{h}"
            text = f"Answer: {ans}\nRationale: stub rationale ({n_noise} noise paragraphs)."
            out.append(GenResult(text, self.count_prompt_tokens(msgs), len(text.split()), 0.0, "stub"))
        return out


class VLLMGenerator:
    """Llama-3-8B-Instruct via vLLM, greedy. NOT exercised on CPU; review on the GPU smoke test."""
    backend = "vllm"

    def __init__(self, model: str, dtype: str = "float16", max_model_len: int = 8192,
                 max_new_tokens: int = 256, seed: int = 0, gpu_memory_utilization: float = 0.90):
        from vllm import LLM, SamplingParams  # imported lazily so CPU runs never need vllm
        self.model = model
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len, seed=seed,
                       gpu_memory_utilization=gpu_memory_utilization)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, seed=seed)

    def count_prompt_tokens(self, messages: list[dict]) -> int:
        return count_chat_tokens(self.tokenizer, messages)

    def count_text_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, batch: list[list[dict]], metas: list[dict]) -> list[GenResult]:
        t0 = time.perf_counter()
        outs = self.llm.chat(batch, self.sampling, use_tqdm=False)
        batch_ms = (time.perf_counter() - t0) * 1000
        results = []
        for o in outs:
            m = getattr(o, "metrics", None)
            if m is not None and getattr(m, "finished_time", None) and getattr(m, "arrival_time", None):
                wall, src = (m.finished_time - m.arrival_time) * 1000, "vllm_request_metrics"
            else:
                wall, src = batch_ms / len(outs), "batch_wall_div_n"
            results.append(GenResult(o.outputs[0].text, len(o.prompt_token_ids),
                                     len(o.outputs[0].token_ids), wall, src))
        return results


def make_generator(cfg: dict):
    g = cfg["generator"]
    if g["backend"] == "stub":
        gen = StubGenerator(g["max_model_len"], g["max_new_tokens"])
    elif g["backend"] == "vllm":
        gen = VLLMGenerator(g["model"], g["dtype"], g["max_model_len"], g["max_new_tokens"], g["seed"])
        gen.params = g["params"]   # stated in config (8.03e9 for Llama-3-8B), feeds the FLOPs proxy
    else:
        raise ValueError(g["backend"])
    return gen
