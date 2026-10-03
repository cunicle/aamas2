#!/usr/bin/env bash
# Back up the pod's results/ every 10 minutes into the repo's (gitignored) results/pod_backup_c/;
# as soon as /workspace/RESULTS_PACKED exists, pull the final archive and exit. Also exits after
# $1 minutes (default 110) so the session gets a heartbeat.
SSH="ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=20 -i $HOME/.ssh/id_ed25519 -p 26284 root@185.216.23.177"
OUT=/c/my-code/research/aamas2/results/pod_backup_c
mkdir -p $OUT
end=$(( $(date +%s) + 60 * ${1:-110} ))
while true; do
  ts=$(date -u +%H%M)
  if $SSH 'test -f /workspace/RESULTS_PACKED' < /dev/null; then
    scp -F /dev/null -o BatchMode=yes -i $HOME/.ssh/id_ed25519 -P 26284 \
        root@185.216.23.177:/workspace/exp_c_results.tar.gz $OUT/exp_c_results_final.tar.gz < /dev/null \
      && tar tzf $OUT/exp_c_results_final.tar.gz > /dev/null && echo "FINAL_PULLED $ts $(stat -c %s $OUT/exp_c_results_final.tar.gz)" && exit 0
    echo "final pull failed $ts"
  fi
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
