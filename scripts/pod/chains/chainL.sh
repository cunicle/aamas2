source /workspace/aamas2_env.sh; cd /workspace/aamas2
bash scripts/run_minimal.sh lenprior_llada2 > results/log_lenprior_llada2.txt 2>&1
bash scripts/run_minimal.sh surplus_llada2 > results/log_surplus_llada2.txt 2>&1
touch results/CHAIN_L_DONE
