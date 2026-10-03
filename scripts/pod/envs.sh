#!/usr/bin/env bash
# Recreate the two venvs on a restarted pod (container disk is wiped; models stay in /workspace).
set -uo pipefail
export PIP_NO_CACHE_DIR=1
LOG=/workspace/aamas2_logs
mkdir -p $LOG /root/envs

venv() {  # $1 name  $2 torch spec  $3 torch index  $4 transformers version
  python3.12 -m venv /root/envs/$1 &&
  /root/envs/$1/bin/pip install -q --upgrade pip &&
  /root/envs/$1/bin/pip install -q "$2" --index-url "$3" &&
  /root/envs/$1/bin/pip install -q "transformers==$4" accelerate numpy scipy pytest hf_xet &&
  /root/envs/$1/bin/python -c "import torch, transformers; print('$1', torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.is_available())"
}

( venv dream  "torch==2.5.1" https://download.pytorch.org/whl/cu124 4.46.2 > $LOG/venv2_dream.txt 2>&1; echo "exit=$?" >> $LOG/venv2_dream.txt ) &
( venv llada2 "torch==2.8.0" https://download.pytorch.org/whl/cu128 4.57.1 > $LOG/venv2_llada2.txt 2>&1; echo "exit=$?" >> $LOG/venv2_llada2.txt ) &
wait
echo ENVS_DONE > $LOG/envs_done
