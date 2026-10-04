#!/usr/bin/env bash
# Laptop side. Back up the pod's results/ every 10 minutes into results/pod_backup_d/ (gitignored);
# exit once /workspace/RESULTS_PACKED exists, or after $1 minutes (default 110) as a heartbeat.
# POD_HOST / POD_PORT: the pod's direct ssh address.
SSH="ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/id_ed25519 -p $POD_PORT root@$POD_HOST"
OUT=/c/my-code/research/aamas2/results/pod_backup_d
mkdir -p $OUT
end=$(( $(date +%s) + 60 * ${1:-110} ))
while true; do
  ts=$(date -u +%H%M)
  if $SSH 'test -f /workspace/RESULTS_PACKED' < /dev/null; then echo "PACKED_SEEN $ts"; exit 0; fi
  if $SSH 'cd /workspace/aamas2 && tar czf - --exclude=results/release_2026-10-03 results' < /dev/null > $OUT/partial.tgz.tmp \
      && tar tzf $OUT/partial.tgz.tmp > /dev/null 2>&1; then
    mv -f $OUT/partial.tgz.tmp $OUT/latest_partial.tgz
    echo "backup $ts $(stat -c %s $OUT/latest_partial.tgz) bytes"
  else
    echo "backup failed $ts"
  fi
  [ $(date +%s) -ge $end ] && echo "HEARTBEAT_EXIT $ts" && exit 0
  sleep 600
done
