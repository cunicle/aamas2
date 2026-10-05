source /workspace/aamas2_env.sh > /dev/null
echo "time $(date -u +%F\ %T) UTC | gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
for m in /tmp/gate_e/GATE_DONE /tmp/gate_e/GATE_PASS /tmp/gate_e/GATE_FAIL results/CHAIN_E_*_DONE /workspace/RESULTS_PACKED; do [ -f $m ] && echo "$m"; done
echo "-- gate times:"; cat /tmp/gate_e/times.txt 2>/dev/null | sed "s/^/  /"
echo "-- chain log:"; cat results/log_e_times.txt 2>/dev/null | sed "s/^/  /"
echo "-- running:"; pgrep -af "^/root/envs/(dream|llada2)/bin/python scripts" | sed -E "s/.*scripts\/([a-z_]+\.py).*--model ([^ ]+)(.*)--out ([^ ]+).*/  \1 \2 \3-> \4/" | cut -c1-220
echo "-- last progress lines:"; for f in results/log_e_*.txt; do l=$(tail -c 400 $f | tr "\r" "\n" | grep -E "s/run|runs \(|^[0-9]+/[0-9]+$" | tail -1); [ -n "$l" ] && echo "  $(basename $f): $l"; done
echo "-- records:"; wc -l results/dream/bfcl_closer.jsonl results/qwen/bfcl_surplus.jsonl results/llada2/bfcl_surplus.jsonl results/dream/length_prior_closer_s4.jsonl results/llada2/length_prior_s2.jsonl 2>/dev/null | sed "s/^/  /"
grep -lE "Traceback" results/log_e_*.txt /tmp/gate_e/log_*.txt 2>/dev/null | sed "s/^/  TRACEBACK in /"
