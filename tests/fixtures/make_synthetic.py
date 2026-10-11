"""Synthetic HotpotQA-format file (original release layout) for CPU harness checks.

NOT data. Paragraph text is random words; it only has the shape of HotpotQA:
2 gold + 8 distractors per question, distractor titles shared across questions (so the
corpus has duplicates to remove), ~80/20 bridge/comparison, some yes/no comparison golds.

Usage: python tests/fixtures/make_synthetic.py out.json [n_questions]
"""
import json
import random
import sys

rng = random.Random(0)
VOCAB = [f"w{i}" for i in range(400)]


def para(topic, n=40):
    words = rng.sample(VOCAB, n - 3) + [topic] * 3
    rng.shuffle(words)
    sents = [" ".join(words[i:i + 10]) + "." for i in range(0, n, 10)]
    return [sents[0]] + [" " + s for s in sents[1:]]


def main(path, n=300):
    shared = [(f"Shared Article {j}", para(f"s{j}")) for j in range(1500)]
    data = []
    for i in range(n):
        q_type = "comparison" if i % 5 == 0 else "bridge"
        ga, gb = (f"Gold {i} A", para(f"ga{i}")), (f"Gold {i} B", para(f"gb{i}"))
        dist = rng.sample(shared, 8)
        ctx = [ga, gb] + dist
        rng.shuffle(ctx)
        answer = ("yes" if i % 10 == 0 else f"Entity {i}") if q_type == "comparison" else f"Answer {i}"
        q = f"Question {i}: what links ga{i} and gb{i} via {' '.join(rng.sample(VOCAB, 6))}?"
        data.append({
            "_id": f"syn{i:04d}", "question": q, "answer": answer, "type": q_type,
            "level": rng.choice(["easy", "medium", "hard"]),
            "supporting_facts": [[ga[0], 0], [gb[0], 1], [ga[0], 2]],
            "context": [[t, s] for t, s in ctx],
        })
    json.dump(data, open(path, "w"))
    print(f"wrote {n} synthetic questions to {path}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
