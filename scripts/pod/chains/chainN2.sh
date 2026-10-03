source /workspace/aamas2_env.sh; cd /workspace/aamas2
until [ -f results/CHAIN_D_DONE ]; do sleep 60; done
bash scripts/run_minimal.sh estimate_dream > results/log_estimate_dream.txt 2>&1
touch results/CHAIN_N2_DONE
