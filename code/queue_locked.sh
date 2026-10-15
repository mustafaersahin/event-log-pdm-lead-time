#!/bin/bash
# Like queue.sh, but skips jobs another worker has started (its log file exists), so that
# two workers can share one job list.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd $ROOT; [ -n "$VENV" ] && . $VENV/bin/activate; export TORCH_THREADS=1 OMP_NUM_THREADS=1
while read lead model mod seed; do
  log=logs/lead${lead}_${model}_${mod}_s${seed}.log
  [ -f runs/lead$lead/pred_${model}_${mod}_s${seed}.npz ] && continue
  [ -f $log ] && continue
  echo "$(date +%H:%M:%S) start lead$lead $model $mod s$seed"
  python code/run_models.py data/pdm_h24_lead$lead.npz runs/lead$lead $model $mod $seed > $log 2>&1
  tail -1 $log
done < $1
echo "$(date +%H:%M:%S) worker done"
