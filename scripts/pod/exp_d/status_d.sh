source /workspace/aamas2_env.sh > /dev/null
echo "time $(date -u +%F\ %T) UTC | gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
for m in results/gate_d/GATE_DONE results/CHAIN_D*_DONE /workspace/RESULTS_PACKED; do [ -f $m ] && echo "$m"; done
echo "-- gate times:"; cat results/gate_d/times.txt 2>/dev/null | sed "s/^/  /"
echo "-- chain log:"; cat results/CHAIN_D_TIMES 2>/dev/null | sed "s/^/  /"
echo "-- running:"; pgrep -af "^/root/envs/dream/bin/python scripts" | sed -E "s/.*scripts\/([a-z_]+\.py)(.*)--out ([^ ]+).*/  \1 \2-> \3/" | cut -c1-200
echo "-- last progress lines:"; for f in results/log_*.txt; do l=$(tail -c 400 $f | tr "\r" "\n" | grep -E "s/run|s/agent|teams|runs \(|^[0-9]+/[0-9]+$" | tail -1); [ -n "$l" ] && echo "  $(basename $f): $l"; done
echo "-- records:"; wc -l results/dream/bfcl_tolerant*.jsonl results/*/agents_d.jsonl results/*/agents_pos.jsonl 2>/dev/null | sed "s/^/  /"
echo "-- exceptions:"; $PY_DREAM - <<"PY"
import glob, json
for f in sorted(glob.glob("results/dream/bfcl_tolerant*.jsonl") + glob.glob("results/*/agents_d.jsonl") + glob.glob("results/*/agents_pos.jsonl")):
    n = e = 0
    for l in open(f):
        r = json.loads(l); n += 1; e += ("error" in r)
    if e: print(f"  {f}: {e}/{n}")
PY
grep -lE "Traceback" results/log_*.txt 2>/dev/null | sed "s/^/  TRACEBACK in /"
