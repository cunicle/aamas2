source /workspace/aamas2_env.sh; cd /workspace/aamas2
until [ -f results/CHAIN_L_DONE ]; do sleep 60; done
bash scripts/run_minimal.sh swap_dream > results/log_swap_dream.txt 2>&1
bash scripts/run_minimal.sh swap_llada2 > results/log_swap_llada2.txt 2>&1
bash scripts/run_minimal.sh surplus1_llada2 > results/log_surplus1_llada2.txt 2>&1
bash scripts/run_minimal.sh estimate_llada2 > results/log_estimate_llada2.txt 2>&1
bash scripts/run_minimal.sh lensweep_llada2 > results/log_lensweep_llada2.txt 2>&1
touch results/CHAIN_N1_DONE
