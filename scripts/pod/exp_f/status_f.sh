source /workspace/aamas2_env.sh > /dev/null
echo "time $(date -u +%F\ %T) UTC | gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
for m in /tmp/gate_f/GATE_DONE /tmp/gate_f/GATE_PASS /tmp/gate_f/GATE_FAIL results/CHAIN_F_DONE /workspace/RESULTS_PACKED; do [ -f $m ] && echo "$m"; done
echo "-- gate times:"; cat /tmp/gate_f/times.txt 2>/dev/null | sed "s/^/  /"
echo "-- chain log:"; cat results/log_f_times.txt 2>/dev/null | sed "s/^/  /"
echo "-- running:"; pgrep -af "^/root/envs/dream/bin/python scripts" | sed -E "s/.*scripts\/([a-z_]+\.py).*--model ([^ ]+)(.*)--out ([^ ]+).*/  \1 \2 -> \4/" | cut -c1-220
echo "-- last progress lines:"; for f in results/log_f_*.txt; do l=$(tail -c 400 $f 2>/dev/null | tr "\r" "\n" | grep -E "teams|done:" | tail -1); [ -n "$l" ] && echo "  $(basename $f): $l"; done
echo "-- records:"; wc -l results/{qwen,dream}/agents_f1.jsonl results/{qwen,dream}/agents_f_choose.jsonl 2>/dev/null | sed "s/^/  /"
grep -lE "Traceback" results/log_f_*.txt /tmp/gate_f/log_*.txt 2>/dev/null | sed "s/^/  TRACEBACK in /"
