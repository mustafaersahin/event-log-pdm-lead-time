# Event-log predictive maintenance benchmark

Code and outputs accompanying the manuscript "When Do Error Logs Stop Predicting Failures? Lead Time,
Model Choice and Decision Cost in Event-Log Predictive Maintenance" (M. Erşahin and B. Erşahin).

## Data
Microsoft Azure predictive maintenance dataset (PdM_telemetry, PdM_errors, PdM_maint, PdM_failures,
PdM_machines). The copy used here was downloaded from
https://github.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy (data/); `bash code/get_data.sh` downloads it into `data/` and verifies the MD5 sums
(`code/data.md5`). The data are not redistributed here. MD5 of the files used:

    8300c768c47be96603edb5bbccb665d0  PdM_errors.csv
    589c4ae33c0d073681dbf6691a936134  PdM_failures.csv
    1c77ce096f869181bfb1e8b018de2d79  PdM_machines.csv
    46f5306c2ed343acdf73501d81e9e14e  PdM_maint.csv
    598467429fd85e2441447e5c2f00a773  PdM_telemetry.csv

## Pipeline

    python code/prepare.py data data/pdm_h24_lead0.npz 24 0      # also lead 24 and 48
    python code/run_models.py data/pdm_h24_lead24.npz runs/lead24 gbm both 0
    python code/evaluate.py data/pdm_h24_lead24.npz runs/lead24 results/lead24
    python code/make_pdm_tables.py results <paper_dir>    # tables and figures of the paper
    python code/coupling_table.py data <paper_dir>/tables/coupling.tex

`queue_v2.txt` lists the main runs (model, modality, lead, seed); `queue_seeds.txt` the seed replications;
`queue_tab.txt` the tabular re-runs with prior correction. `queue.sh` / `queue_locked.sh` execute a list.

## Outputs
- `runs/lead*/pred_<model>_<modality>_s<seed>.npz`: validation and test predictions (raw and calibrated) of every run;
  `meta_*.json`: temperature, training time, latency, parameter count.
- `results/lead*/metrics.json`: all metrics, decision-simulation savings and bootstrap intervals.
- `logs/`: training logs.

Environment: Python 3.13, PyTorch 2.x (CPU), scikit-learn 1.9, NumPy 2.x; 4-core CPU shared with other jobs
(training times are therefore conservative). Data split: train < 2015-09-01 <= validation < 2015-11-01 <= test.

Set `VENV` to a virtual environment to be activated by `queue.sh` / `queue_locked.sh`.
