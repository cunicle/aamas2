source /workspace/aamas2_env.sh > /dev/null
echo "time $(date -u +%F\ %T) UTC | gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
for m in CHAIN_C1_DONE CHAIN_C2_DONE CHAIN_C3_DONE; do [ -f results/$m ] && echo "$m"; done
[ -f /workspace/RESULTS_PACKED ] && echo RESULTS_PACKED
echo "-- chain log:"; cat results/CHAIN_C_TIMES 2>/dev/null | sed "s/^/  /"
echo "-- running:"; pgrep -af "^/root/envs/dream/bin/python scripts" | sed -E "s/.*scripts\/([a-z_]+\.py).*--model ([^ ]+).*--out ([^ ]+).*/  \1 \2 -> \3/"
echo "-- last progress lines:"; for f in results/log_*.txt; do l=$(tail -c 400 $f | tr "\r" "\n" | grep -E "s/run|s/agent|teams|runs \(|^[0-9]+/[0-9]+$" | tail -1); [ -n "$l" ] && echo "  $(basename $f): $l"; done
echo "-- records:"; wc -l results/*/agents_c*.jsonl results/*/agents_rule.jsonl results/dream/bfcl_closer*.jsonl results/dream/bfcl_onesided.jsonl results/dream/length_prior_closer*.jsonl results/dream/choose_tau_ltr.jsonl 2>/dev/null | sed "s/^/  /"
echo "-- exceptions:"; $PY_DREAM - <<"PY"
import glob, json
for f in sorted(glob.glob("results/*/agents_c*.jsonl") + glob.glob("results/*/agents_rule.jsonl") + glob.glob("results/dream/bfcl_closer*.jsonl") + glob.glob("results/dream/bfcl_onesided.jsonl") + glob.glob("results/dream/choose_tau_ltr.jsonl")):
    n = e = 0
    for l in open(f):
        r = json.loads(l); n += 1; e += ("error" in r)
    if e: print(f"  {f}: {e}/{n}")
PY
grep -lE "Traceback" results/log_*.txt 2>/dev/null | sed "s/^/  TRACEBACK in /"
