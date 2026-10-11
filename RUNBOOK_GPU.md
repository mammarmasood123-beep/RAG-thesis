# GPU runbook: Experiment A pilot on a RunPod A40 (48 GB)

**Do not start a pod until** (1) the III.5 pool gate passes and `artifacts/pool.jsonl` exists, (2) `python -m pytest tests/ -q` shows 0 failed, 0 skipped on your Mac, and (3) HuggingFace has approved your access to **`meta-llama/Meta-Llama-3-8B-Instruct`** (the repo named in `configs/pilot.yaml`). An idle pod still bills.

## 0. On your Mac: pack and record hashes
```bash
cd "/Users/Mac/Desktop/MS SE/Thesis 3/Implementation/RQ1"
shasum -a 256 expA/hotpot_dev_distractor_v1.json expA/artifacts/{pool,sample,paragraphs}.jsonl expA/configs/pilot.yaml > expA_hashes_mac.txt
tar czf expA_pod.tgz --exclude='expA/runs' --exclude='expA/logs/task*' --exclude='*/__pycache__' --exclude='expA/.pytest_cache' expA
```

## 1. Pod
Template: RunPod PyTorch (any CUDA 12.x image) · GPU: 1× A40 48 GB · container disk ≥ 50 GB · volume at `/workspace` ≥ 50 GB · add your SSH public key in RunPod settings first. The pod's **Connect** panel shows `<IP>` and `<PORT>`.

## 2. Copy onto the pod (from your Mac)
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 expA_pod.tgz expA_hashes_mac.txt root@<IP>:/workspace/
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<IP>
```

## 3. On the pod: install, auth, model download (one time)
```bash
cd /workspace && tar xzf expA_pod.tgz && cd expA
python3 -m venv /workspace/venv && source /workspace/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "vllm==0.29.0" rank_bm25==0.2.2 pyyaml matplotlib pytest
export HF_TOKEN=hf_xxxxxxxx                 # read token; never write it to a file in expA
export HF_HOME=/workspace/hf                # model cache on the persistent volume
python -c "from huggingface_hub import snapshot_download as s; print(s('meta-llama/Meta-Llama-3-8B-Instruct', allow_patterns=['*.json','*.safetensors','tokenizer*']))" | tee model_snapshot_path.txt
export HF_HUB_OFFLINE=1                     # from here on, runs cannot silently fetch a newer revision
sha256sum -c <(sed 's#expA/##' ../expA_hashes_mac.txt)   # must print OK for every line
python -m pytest tests/ -q                  # expect 0 failed, 0 skipped
```
The last path component in `model_snapshot_path.txt` (`…/snapshots/<40-hex>`) is the **model revision**: record it.

## 4. Environment record (before any run)
```bash
mkdir -p env && {
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  nvidia-smi | head -4                       # driver + CUDA version line
  python -c "import sys,torch,vllm,transformers;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,'vllm',vllm.__version__,'transformers',transformers.__version__)"
  echo "model repo: meta-llama/Meta-Llama-3-8B-Instruct  revision: $(basename $(cat model_snapshot_path.txt))"
  sha256sum hotpot_dev_distractor_v1.json artifacts/pool.jsonl artifacts/sample.jsonl artifacts/paragraphs.jsonl configs/pilot.yaml
} | tee env/environment.txt; python -m pip freeze > env/pip_freeze.txt
```

## 5. Smoke test: 20 questions, run twice
```bash
python -m harness.run_expA --config configs/pilot.yaml --backend vllm --limit 20 --run_suffix=_gpu_smoke1
python -m harness.run_expA --config configs/pilot.yaml --backend vllm --limit 20 --run_suffix=_gpu_smoke2
python -m logs.check_scripts.diff_runs runs/*_gpu_smoke1 runs/*_gpu_smoke2
python -m eval.aggregate runs/*_gpu_smoke1
python -c "import json,glob; p=glob.glob('runs/*_gpu_smoke1')[0]; r=[json.loads(l) for l in open(p+'/records.jsonl')]; ran=[x for x in r if x['output']]; print('prompt_tokens == input_tokens on all ran records:', all(x['context']['prompt_tokens']==x['modules'][0]['input_tokens'] for x in ran)); print('max prompt_tokens:', max(x['context']['prompt_tokens'] for x in r))"
```
**Expected:** each run prints `records=260  not_fitting=0  status=OK`, `n0_prompt_identity_violations: 0`, `gold_integrity_violations: 0`. `n0_output_identity_mismatches` is a soft check; any nonzero count means GPU batching is changing outputs. The diff should show `fields that differ besides run_id: NONE` except `modules[].wall_ms`, which is timing and will differ. The token check must print `True`, with a max prompt-token count in the low thousands. **Stop and bring the raw output back** if any of these fail, if `summary.md` shows parse failures above your threshold in any cell, or if the N=0 row is not flat from T=2. Read at least five raw outputs in `runs/*_gpu_smoke1/records.jsonl` before going on.

## 6. Full pilot: 150 questions
```bash
python -m harness.run_expA --config configs/pilot.yaml --backend vllm --run_suffix=_gpu_pilot150
python -m eval.aggregate runs/*_gpu_pilot150 && python -m eval.plot runs/*_gpu_pilot150
```
**Expected:** `records=1950  not_fitting=0  status=OK`, plus `results/{summary.md,cells.csv,paired_ci.csv,hygiene_by_cell.csv,accuracy_vs_T.png}`.

## 7. Download, THEN stop the pod
```bash
# on the pod
cd /workspace/expA && tar czf /workspace/gpu_runs.tgz runs env model_snapshot_path.txt
# on your Mac
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/gpu_runs.tgz "/Users/Mac/Desktop/MS SE/Thesis 3/Implementation/RQ1/"
tar tzf gpu_runs.tgz | head                  # confirm the archive opens before stopping
```
Then stop the pod in the RunPod console. Terminating deletes `/workspace` too.

## What to record for Chapter 3
GPU name and memory; driver and CUDA version; Python, torch, vLLM, transformers versions; model repo **and** the 40-hex revision; sha256 of the dataset, pool, sample, paragraphs and config; run_ids; `manifest.json` of each run (it already embeds config, input hashes, checks and environment).
