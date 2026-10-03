source /workspace/aamas2_env.sh; cd /workspace/aamas2
until [ -f results/CHAIN_N1_DONE ]; do sleep 60; done
bash scripts/run_minimal.sh choose_llada2 > results/log_choose_llada2.txt 2>&1
touch results/CHAIN_N4_DONE
