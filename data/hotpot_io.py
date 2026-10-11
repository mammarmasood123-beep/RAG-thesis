"""Load HotpotQA (distractor setting) into one normalised record shape.

Accepts either
  * the original release file  hotpot_dev_distractor_v1.json  (a JSON list;
    context = [[title, [sent, ...]], ...]; supporting_facts = [[title, sent_id], ...]), or
  * a JSON / JSONL export of the HuggingFace `hotpot_qa` "distractor" config
    (context = {"title": [...], "sentences": [[...], ...]};
     supporting_facts = {"title": [...], "sent_id": [...]}).

Paragraph text is "".join(sentences): HotpotQA sentences after the first already
carry their leading space, so this reproduces the article paragraph exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HotpotItem:
    qid: str
    question: str
    answer: str
    q_type: str                    # "bridge" | "comparison"
    level: str                     # "easy" | "medium" | "hard"
    context: list[tuple[str, str]]  # [(title, paragraph_text)] in shipped order
    sf_titles: list[str]           # distinct supporting titles, order of first appearance
    extra: dict = field(default_factory=dict)

    @property
    def context_titles(self) -> list[str]:
        return [t for t, _ in self.context]

    def paragraph(self, title: str) -> str:
        for t, x in self.context:
            if t == title:
                return x
        raise KeyError(title)


def _distinct_in_order(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _normalise(raw: dict) -> HotpotItem:
    ctx = raw["context"]
    if isinstance(ctx, dict):  # HuggingFace layout
        context = [(t, "".join(s)) for t, s in zip(ctx["title"], ctx["sentences"])]
    else:                      # original layout
        context = [(t, "".join(s)) for t, s in ctx]
    sf = raw["supporting_facts"]
    sf_titles = sf["title"] if isinstance(sf, dict) else [t for t, _ in sf]
    return HotpotItem(
        qid=raw.get("_id") or raw.get("id"),
        question=raw["question"],
        answer=raw["answer"],
        q_type=raw["type"],
        level=raw["level"],
        context=context,
        sf_titles=_distinct_in_order(sf_titles),
    )


def load_hotpot(path: str | Path) -> list[HotpotItem]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        raws = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raws = json.loads(text)
    return [_normalise(r) for r in raws]
