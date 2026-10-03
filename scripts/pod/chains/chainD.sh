source /workspace/aamas2_env.sh; cd /workspace/aamas2
bash scripts/run_minimal.sh lenprior_dream > results/log_lenprior_dream.txt 2>&1
bash scripts/run_minimal.sh surplus_dream > results/log_surplus_dream.txt 2>&1
bash scripts/run_minimal.sh surplus_ar > results/log_surplus_ar.txt 2>&1
bash scripts/run_minimal.sh endbias_dream > results/log_endbias_dream.txt 2>&1
touch results/CHAIN_D_DONE
