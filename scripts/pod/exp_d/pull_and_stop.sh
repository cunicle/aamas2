#!/usr/bin/env bash
# Laptop side. Wait for /workspace/RESULTS_PACKED, pull the final archive into results/pod_backup_d/,
# check it (tar lists, has the quick table and every result file), then stop the pod.
# POD_HOST / POD_PORT: the pod's direct ssh address.
SSH="ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST"
OUT=/c/my-code/research/aamas2/results/pod_backup_d
mkdir -p $OUT
check() {  # the archive lists, with the quick table and every result file of experiment D
  tar tzf $OUT/exp_d_results_final.tar.gz > $OUT/final_list.txt 2>/dev/null || return 1
  grep -q "results/summary/exp_d_quick.md" $OUT/final_list.txt || return 1
  for f in dream/bfcl_tolerant dream/bfcl_tolerant_swap dream/bfcl_tolerant_onesided dream/bfcl_tolerant_estimate            qwen/agents_d dream/agents_d qwen/agents_pos dream/agents_pos; do
    grep -q "results/$f.jsonl" $OUT/final_list.txt || return 1
  done
}
until $SSH 'test -f /workspace/RESULTS_PACKED' < /dev/null; do sleep 30; done
echo "packed seen $(date -u +%T)"
for try in 1 2 3; do
  scp -F /dev/null -o BatchMode=yes -i $HOME/.ssh/id_ed25519 -P $POD_PORT \
      root@$POD_HOST:/workspace/exp_d_results.tar.gz $OUT/exp_d_results_final.tar.gz < /dev/null
  if check; then
    echo "archive ok $(stat -c %s $OUT/exp_d_results_final.tar.gz) bytes $(date -u +%T)"
    $SSH 'export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=" | xargs); runpodctl stop pod $RUNPOD_POD_ID' < /dev/null
    echo "STOP_SENT exit=$? $(date -u +%T)"
    exit 0
  fi
  echo "archive check failed (try $try)"; sleep 20
done
echo "PULL_FAILED: pod left running for the 45-minute self-stop"
exit 1
