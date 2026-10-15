#!/bin/bash
# PdM benchmark queue, in order of importance. Each line: lead model modality seed.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd $ROOT; [ -n "$VENV" ] && . $VENV/bin/activate; export TORCH_THREADS=2 OMP_NUM_THREADS=2
run() {
  f=runs/lead$1/pred_$2_$3_s$4.npz
  [ -f $f ] && return
  echo "$(date +%H:%M:%S) start lead$1 $2 $3 s$4"
  python code/run_models.py data/pdm_h24_lead$1.npz runs/lead$1 $2 $3 $4 > logs/lead$1_$2_$3_s$4.log 2>&1
  tail -1 logs/lead$1_$2_$3_s$4.log
}
while read lead model mod seed; do run $lead $model $mod $seed; done < $1
echo "$(date +%H:%M:%S) queue done"
