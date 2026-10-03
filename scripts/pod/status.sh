cd /workspace/aamas2
echo "time $(date -u +%F\ %T) UTC | gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
for m in CHAIN_L_DONE CHAIN_D_DONE CHAIN_N1_DONE CHAIN_N2_DONE CHAIN_N3_DONE CHAIN_N4_DONE CHAIN_N5_DONE; do [ -f results/$m ] && echo "$m"; done
echo "-- running:"; pgrep -af "^/root/envs/(dream|llada2)" | sed -E "s/.*scripts\/([a-z_]+\.py).*--model ([^ ]+).*--out ([^ ]+).*/  \1 \2 -> \3/"
echo "-- last progress lines:"; for f in results/log_*.txt; do l=$(tail -c 300 $f | tr "\r" "\n" | grep -E "s/run|runs \(|^[0-9]+/[0-9]+$" | tail -1); [ -n "$l" ] && echo "  $(basename $f): $l"; done
echo "-- new records:"; wc -l results/*/bfcl_surplus.jsonl results/*/bfcl_endbias.jsonl results/*/bfcl_swap.jsonl results/*/bfcl_estimate.jsonl results/qwen/bfcl_s*.jsonl results/*/choose.jsonl results/*/length_prior.jsonl 2>/dev/null | sed "s/^/  /"
echo "-- exceptions:"; /root/envs/dream/bin/python - <<"PY"
import glob, json
for f in sorted(glob.glob("results/*/*.jsonl")):
    n = e = 0
    for l in open(f):
        r = json.loads(l); n += 1; e += ("error" in r)
    if e: print(f"  {f}: {e}/{n} = {e/n:.1%}")
PY
grep -lE "Traceback" results/log_*.txt 2>/dev/null | sed "s/^/  TRACEBACK in /"
