source /workspace/aamas2_env.sh; cd /workspace/aamas2
until [ -f results/CHAIN_N4_DONE ]; do sleep 60; done
# the 10-01 pod stopped LLaDA2.0's main sweep during k=16 (259 of 400 items); finish it
$PY_LLADA2 scripts/run_dllm.py --model inclusionAI/LLaDA2.0-mini --data bfcl:parallel,parallel_multiple --mode skeleton --block-length 32 --k 16 --order confidence --out results/llada2/bfcl_skel_k.jsonl > results/log_llada2_k16.txt 2>&1
touch results/CHAIN_N5_DONE
