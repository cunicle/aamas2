#!/usr/bin/env bash
# Laptop side. Wait for /workspace/RESULTS_PACKED, pull both archives into results/pod_backup_e/
# (gitignored), check them (they list; the release archive has every runbook file; gate 4 passed),
# then stop the pod. If gate 4 failed the pod is left for the 45-minute self-stop.
# POD_HOST / POD_PORT: the pod's direct ssh address.
SSHO="-F /dev/null -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/id_ed25519"
SSH="ssh $SSHO -p $POD_PORT root@$POD_HOST"
OUT=/c/my-code/research/aamas2/results/pod_backup_e
mkdir -p $OUT
check() {
  tar tzf $OUT/exp_e_release.tar.gz > $OUT/release_list.txt 2>/dev/null || return 1
  tar tzf $OUT/exp_e_results_full.tar.gz > $OUT/full_list.txt 2>/dev/null || return 1
  for f in dream/bfcl_closer.jsonl dream/length_prior_closer_s4.jsonl dream/length_prior_closer_s4.txt \
           qwen/bfcl_surplus.jsonl llada2/bfcl_surplus.jsonl llada2/length_prior_s2.jsonl llada2/length_prior_s2.txt \
           log_e_dream.txt log_e_llada2.txt gate_e/gate_check.txt gate_e/gate4_check.txt; do
    grep -qx "results/$f" $OUT/release_list.txt || return 1
  done
  grep -qx "results/summary/exp_e_quick.md" $OUT/full_list.txt
}
until $SSH 'test -f /workspace/RESULTS_PACKED' < /dev/null; do sleep 30; done
echo "packed seen $(date -u +%T)"
for try in 1 2 3; do
  scp $SSHO -P $POD_PORT root@$POD_HOST:/workspace/exp_e_release.tar.gz root@$POD_HOST:/workspace/exp_e_results_full.tar.gz \
      $OUT/ < /dev/null
  if check; then
    echo "archives ok $(stat -c %s $OUT/exp_e_release.tar.gz) / $(stat -c %s $OUT/exp_e_results_full.tar.gz) bytes $(date -u +%T)"
    tar xzf $OUT/exp_e_release.tar.gz -O results/gate_e/gate4_check.txt | tail -1
    if tar xzf $OUT/exp_e_release.tar.gz -O results/gate_e/gate4_check.txt | grep -q "^GATE 4 PASS"; then
      $SSH 'export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=" | xargs); runpodctl stop pod $RUNPOD_POD_ID' < /dev/null
      echo "STOP_SENT exit=$? $(date -u +%T)"
    else
      echo "GATE 4 NOT PASSED: pod left running for its 45-minute self-stop"
    fi
    exit 0
  fi
  echo "archive check failed (try $try)"; sleep 20
done
echo "PULL_FAILED: pod left running for the 45-minute self-stop"
exit 1
