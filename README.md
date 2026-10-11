# Experiment A — pilot harness (RQ1, noise accumulation)

Design: `claude/Design00_Chapter3_DesignBasis.md` Part III. Pilot: 150 HotpotQA validation
questions, N ∈ {0, 4}, rounds 0..6 in one pass, greedy Llama-3-8B-Instruct.

## Run order

```bash
pip install -r requirements.txt            # CPU deps; add vllm on the GPU box only
./scripts/cpu_smoke.sh                     # tests + full chain on synthetic data, stub generator

# 1. GATE: verify the pool on the real split (writes nothing)
python -m data.build_pool --hotpot hotpot_dev_distractor_v1.json --out artifacts/ --verify_only
#    exit 2 + "STOP" => some question has < 24 usable candidates -> fall back to N in {0,1}

# 2. build pool + sample (500, pilot = first 150, type-proportional, gold order counterbalanced)
python -m data.build_pool   --hotpot hotpot_dev_distractor_v1.json --out artifacts/
python -m data.build_sample --hotpot hotpot_dev_distractor_v1.json --out artifacts/

# 3. real data, stub generator, CPU
python -m harness.run_expA --config configs/pilot.yaml --backend stub --limit 20

# 4. A40: real generator (needs HF token with Llama-3 access)
python -m harness.run_expA --config configs/pilot.yaml --backend vllm
python -m eval.aggregate runs/<run_id>
python -m eval.plot      runs/<run_id>
```

Data: `hotpot_dev_distractor_v1.json` from http://curtis.ml.cmu.edu/datasets/hotpot/ (7,405
questions). A JSON/JSONL export of HuggingFace `hotpot_qa`, config `distractor`, split
`validation` also loads.

## Layout

| path | role |
|---|---|
| `data/build_pool.py` | BM25 top-50 over the split's own paragraph corpus; verification gate |
| `data/build_sample.py` | seeded stratified 500 / pilot 150; counterbalanced gold order |
| `data/bm25.py` | rank_bm25 statistics, inverted-index scoring (cross-checked = rank_bm25) |
| `harness/schedule.py` | pure seeded gold/distractor/position scheduling |
| `harness/context.py` | SIM-RAG Reasoner context template |
| `harness/generator.py` | `StubGenerator` (CPU oracle) and `VLLMGenerator` (untested until GPU) |
| `harness/run_expA.py` | runner; JSONL records; manifest with checks and environment |
| `prompts/simrag.py` | SIM-RAG `force_answer_prompt` verbatim (@3d5f13a) + no-abstain variant |
| `eval/hotpot_official.py` | HotpotQA official EM/F1, vendored verbatim |
| `eval/aggregate.py`, `eval/plot.py` | tables with bootstrap CIs; accuracy-vs-T plot |
| `configs/pilot.yaml` | every parameter and every logged deviation from SIM-RAG |

## Outputs per run (`runs/<run_id>/`)

`records.jsonl` (one per question × round × condition), `prompts.jsonl` (full user
content), `manifest.json` (config, input hashes, checks, environment, token ceiling),
`results/summary.md`, `results/cells.csv`, `results/cells_sensitivity.csv`,
`results/closed_book_correct.txt`, `results/accuracy_vs_T.png`.
