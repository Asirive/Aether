#!/usr/bin/env bash
# Usage: run_queue.sh OUTDIR "spec:seed spec:seed ..."   (sequential, 1 thread)
out=$1; shift
for job in $1; do
  spec=${job%%:*}; seed=${job##*:}
  OMP_NUM_THREADS=1 python3 -W ignore ablations/lab.py --specs "$spec" --seeds "$seed" --steps 800 --threads 1 --out "$out.json" >> "$out.log" 2>&1
done
