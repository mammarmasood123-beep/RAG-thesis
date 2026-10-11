#!/usr/bin/env bash
# Full chain on CPU with the synthetic fixture and the stub generator.
# Checks wiring only: pool gate -> sample -> run -> aggregate -> plot. Numbers are meaningless.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pytest -q
python tests/fixtures/make_synthetic.py artifacts_synth/hotpot_synth.json 300
python -m data.build_pool   --hotpot artifacts_synth/hotpot_synth.json --out artifacts_synth/
python -m data.build_sample --hotpot artifacts_synth/hotpot_synth.json --out artifacts_synth/ \
       --n_total 200 --n_pilot 150 --seed 20261008

sed -e 's#artifacts/#artifacts_synth/#' -e 's#runs_dir: runs#runs_dir: runs_synth#' \
    -e 's#run_name: pilotA#run_name: synthA#' configs/pilot.yaml > artifacts_synth/synthetic.yaml
python -m harness.run_expA --config artifacts_synth/synthetic.yaml --backend stub
RUN=$(ls -td runs_synth/*/ | head -1)
python -m eval.aggregate "$RUN"
python -m eval.plot "$RUN"
echo "OK: $RUN"
