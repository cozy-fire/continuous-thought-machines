#!/usr/bin/env bash
set -euo pipefail

source /environment/miniconda3/etc/profile.d/conda.sh
conda activate /home/featurize/work/envs/ctm

cd /home/featurize/work/continuous-thought-machines
mkdir -p logs
python -m scripts.ctm_forward_smoke \
  > >(tee logs/ctm_forward_smoke.stdout.log) \
  2> >(tee logs/ctm_forward_smoke.stderr.log >&2)
