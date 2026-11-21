#!/usr/bin/env bash
# Laptop side. Wait for /workspace/RESULTS_PACKED, pull both archives into results/pod_backup_f/
# of this checkout (gitignored), check them (they list; the release archive has every runbook
# file), then stop the pod. The pod is stopped whether or not gate 4 passed: the records are
# pulled either way, and the pod stops itself 45 minutes after RESULTS_PACKED if this fails.
# POD_HOST / POD_PORT: the pod's direct ssh address.
SSHO="-F /dev/null -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/id_ed25519"
SSH="ssh $SSHO -p $POD_PORT root@$POD_HOST"
OUT="$(cd "$(dirname "$0")/../../.." && pwd)/results/pod_backup_f"
mkdir -p "$OUT"
check() {
  tar tzf "$OUT/exp_f_release.tar.gz" > "$OUT/release_list.txt" 2>/dev/null || return 1
  tar tzf "$OUT/exp_f_results_full.tar.gz" > "$OUT/full_list.txt" 2>/dev/null || return 1
  for f in qwen/agents_f1.jsonl qwen/agents_f_choose.jsonl dream/agents_f1.jsonl dream/agents_f_choose.jsonl \
           summary/exp_f_quick.md log_f_ar.txt log_f_dream.txt gate_f/gate_check.txt gate_f/gate4_check.txt; do
    grep -qx "results/$f" "$OUT/release_list.txt" || return 1
  done
}
until $SSH 'test -f /workspace/RESULTS_PACKED' < /dev/null; do sleep 30; done
echo "packed seen $(date -u +%T)"
for try in 1 2 3; do
  scp $SSHO -P $POD_PORT root@$POD_HOST:/workspace/exp_f_release.tar.gz root@$POD_HOST:/workspace/exp_f_results_full.tar.gz \
      "$OUT/" < /dev/null
  if check; then
    echo "archives ok $(stat -c %s "$OUT/exp_f_release.tar.gz") / $(stat -c %s "$OUT/exp_f_results_full.tar.gz") bytes $(date -u +%T)"
    tar xzf "$OUT/exp_f_release.tar.gz" -O results/gate_f/gate4_check.txt | tail -1
    $SSH 'export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=" | xargs); runpodctl stop pod $RUNPOD_POD_ID' < /dev/null
    echo "STOP_SENT exit=$? $(date -u +%T)"
    exit 0
  fi
  echo "archive check failed (try $try)"; sleep 20
done
echo "PULL_FAILED: pod left running for the 45-minute self-stop"
exit 1
