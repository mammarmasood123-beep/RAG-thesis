# to_hotpot_json.py  (pip install datasets)
import json
from datasets import load_dataset

ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
out = []
for r in ds:
    out.append({
        "_id": r["id"],
        "question": r["question"],
        "answer": r["answer"],
        "type": r["type"],
        "level": r["level"],
        "supporting_facts": [[t, s] for t, s in
            zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"])],
        "context": [[t, list(s)] for t, s in
            zip(r["context"]["title"], r["context"]["sentences"])],
    })
assert len(out) == 7405, len(out)
with open("hotpot_dev_distractor_v1.json", "w") as f:
    json.dump(out, f)
print(len(out), out[0]["_id"], out[0]["question"][:70])