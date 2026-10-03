#!/usr/bin/env bash
# Experiment C pod setup: the Dream env (Dream and Qwen both run in it) and the two models.
# Models go to /dev/shm/hf unless HF_HOME is set (a pod without a volume loses them on stop).
set -uo pipefail
LOG=/workspace/aamas2_logs; mkdir -p $LOG /root/envs
export PIP_NO_CACHE_DIR=1 HF_HOME=${HF_HOME:-/dev/shm/hf}
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
echo "python: $PY $($PY --version 2>&1)" > $LOG/setup_python.txt
( $PY -m venv /root/envs/hf && /root/envs/hf/bin/pip install -q huggingface_hub hf_xet &&
  for m in Dream-org/Dream-v0-Instruct-7B Qwen/Qwen2.5-7B-Instruct; do
    ( /root/envs/hf/bin/hf download $m > $LOG/dl_${m//\//_}.txt 2>&1; echo "exit=$?" >> $LOG/dl_${m//\//_}.txt ) &
  done; wait; echo DL_DONE > $LOG/dl_done ) &
( $PY -m venv /root/envs/dream && /root/envs/dream/bin/pip install -q --upgrade pip &&
  /root/envs/dream/bin/pip install -q "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124 &&
  /root/envs/dream/bin/pip install -q "transformers==4.46.2" accelerate numpy scipy pytest hf_xet &&
  /root/envs/dream/bin/python -c "import torch, transformers; print('dream', torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.is_available())"
  echo "exit=$?" ) > $LOG/venv_dream.txt 2>&1 &
wait
echo SETUP_DONE > $LOG/setup_done
