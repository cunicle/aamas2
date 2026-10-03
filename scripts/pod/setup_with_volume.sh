#!/usr/bin/env bash
# Pod setup for aamas2: two venvs on the container disk, four models into the network volume.
set -uo pipefail
export HF_HOME=/workspace/.cache/huggingface
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

( venv dream  "torch==2.5.1" https://download.pytorch.org/whl/cu124 4.46.2 > $LOG/venv_dream.txt 2>&1; echo "exit=$?" >> $LOG/venv_dream.txt ) &
( venv llada2 "torch==2.8.0" https://download.pytorch.org/whl/cu128 4.57.1 > $LOG/venv_llada2.txt 2>&1; echo "exit=$?" >> $LOG/venv_llada2.txt
  # downloads use the llada2 env's huggingface_hub
  for m in inclusionAI/LLaDA2.0-mini Dream-org/Dream-v0-Instruct-7B inclusionAI/Ling-mini-2.0 Qwen/Qwen2.5-7B-Instruct; do
    ( /root/envs/llada2/bin/hf download $m > $LOG/dl_${m//\//_}.txt 2>&1; echo "exit=$?" >> $LOG/dl_${m//\//_}.txt ) &
  done
  wait ) &
wait
echo SETUP_DONE > $LOG/setup_done
