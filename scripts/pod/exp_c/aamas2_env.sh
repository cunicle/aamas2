export HF_HOME=${HF_HOME:-/dev/shm/hf}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PY_DREAM=/root/envs/dream/bin/python PY_LLADA2=/root/envs/dream/bin/python
cd /workspace/aamas2
