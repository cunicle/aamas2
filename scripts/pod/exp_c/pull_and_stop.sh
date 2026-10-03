#!/usr/bin/env bash
# Wait for /workspace/RESULTS_PACKED, pull the final archive into results/pod_backup_c/, check it
# (tar lists, has the summary tables and the main result files), then stop the pod.
SSH="ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=20 -i $HOME/.ssh/id_ed25519 -p 26284 root@185.216.23.177"
OUT=/c/my-code/research/aamas2/results/pod_backup_c
mkdir -p $OUT
until $SSH 'test -f /workspace/RESULTS_PACKED' < /dev/null; do sleep 30; done
echo "packed seen $(date -u +%T)"
for try in 1 2 3; do
  scp -F /dev/null -o BatchMode=yes -i $HOME/.ssh/id_ed25519 -P 26284 \
      root@185.216.23.177:/workspace/exp_c_results.tar.gz $OUT/exp_c_results_final.tar.gz < /dev/null
  if tar tzf $OUT/exp_c_results_final.tar.gz > $OUT/final_list.txt 2>/dev/null \
      && grep -q "results/summary/closer.md" $OUT/final_list.txt \
      && grep -q "results/summary/agents_bfcl.md" $OUT/final_list.txt \
      && grep -q "results/dream/bfcl_closer.jsonl" $OUT/final_list.txt; then
    echo "archive ok $(stat -c %s $OUT/exp_c_results_final.tar.gz) bytes $(date -u +%T)"
    $SSH 'export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=" | xargs); runpodctl stop pod $RUNPOD_POD_ID' < /dev/null
    echo "STOP_SENT exit=$? $(date -u +%T)"
    exit 0
  fi
  echo "archive check failed (try $try)"; sleep 20
done
echo "PULL_FAILED: pod left running for the 45-minute self-stop"
exit 1
