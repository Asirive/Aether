#!/usr/bin/env bash
# Worker: pops the first line of $Q (flock-protected) and runs it, until the file is empty.
# Line format:  lab|recall  OUT_PREFIX  SPEC  SEED  [extra args...]
Q=${Q:-ablations/results/jobs.txt}
while true; do
  line=$(flock "$Q.lock" bash -c "head -n1 '$Q'; sed -i '1d' '$Q'")
  [ -z "$line" ] && break
  set -- $line; kind=$1; out=$2; spec=$3; seed=$4; shift 4
  echo "[$(date +%H:%M)] start $kind $spec s$seed" >> ablations/results/jobq.log
  if [ "$kind" = lab ]; then
    OMP_NUM_THREADS=1 python3 -W ignore ablations/lab.py --specs "$spec" --seeds "$seed" --steps 800 --threads 1 --out "$out.json" "$@" >> "$out.log" 2>&1
  else
    OMP_NUM_THREADS=1 python3 -W ignore ablations/recall.py --specs "$spec" --seeds "$seed" --out "$out.json" "$@" >> "$out.log" 2>&1
  fi
  echo "[$(date +%H:%M)] done  $kind $spec s$seed rc=$?" >> ablations/results/jobq.log
done
