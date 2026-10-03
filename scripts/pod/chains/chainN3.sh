source /workspace/aamas2_env.sh; cd /workspace/aamas2
until [ -f results/CHAIN_N2_DONE ]; do sleep 60; done
bash scripts/run_minimal.sh choose_dream > results/log_choose_dream.txt 2>&1
bash scripts/run_minimal.sh choose_ar > results/log_choose_ar.txt 2>&1
touch results/CHAIN_N3_DONE
