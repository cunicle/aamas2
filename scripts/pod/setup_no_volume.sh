#!/usr/bin/env bash
# New pod without a volume: models in /dev/shm, envs and repo on the container disk.
set -uo pipefail
LOG=/workspace/aamas2_logs; mkdir -p $LOG
export HF_HOME=/dev/shm/hf PIP_NO_CACHE_DIR=1
( python3.12 -m venv /root/envs/hf && /root/envs/hf/bin/pip install -q huggingface_hub hf_xet &&
  for m in inclusionAI/LLaDA2.0-mini Dream-org/Dream-v0-Instruct-7B Qwen/Qwen2.5-7B-Instruct; do
    ( /root/envs/hf/bin/hf download $m > $LOG/dl_${m//\//_}.txt 2>&1; echo "exit=$?" >> $LOG/dl_${m//\//_}.txt ) &
  done; wait; echo DL_DONE > $LOG/dl_done ) &
bash /workspace/pod_envs.sh &
wait
echo SETUP2_DONE > $LOG/setup2_done
