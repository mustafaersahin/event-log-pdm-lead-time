"""Error-failure coupling in the training period (failures before 2015-09-01) -> LaTeX table."""
import sys
import pandas as pd

DATA, OUT = sys.argv[1], sys.argv[2]
f = pd.read_csv(f"{DATA}/PdM_failures.csv", parse_dates=["datetime"])
e = pd.read_csv(f"{DATA}/PdM_errors.csv", parse_dates=["datetime"])
f = f[f.datetime < "2015-09-01"]
rows = []
for comp in sorted(f.failure.unique()):
    ff = f[f.failure == comp]
    cells = []
    for code in [f"error{i}" for i in range(1, 6)]:
        n, lags = 0, []
        for r in ff.itertuples():
            ee = e[(e.machineID == r.machineID) & (e.errorID == code) & (e.datetime <= r.datetime)
                   & (e.datetime > r.datetime - pd.Timedelta(hours=168))]
            if len(ee):
                n += 1
                lags.append((r.datetime - ee.datetime.max()) / pd.Timedelta(hours=1))
        share = n / len(ff)
        cell = f"{100*share:.0f}" + (f" ({pd.Series(lags).median():.0f}/{max(lags):.0f})" if share >= 0.5 else "")
        cells.append(cell)
    rows.append((comp, len(ff), cells))
with open(OUT, "w") as fh:
    fh.write("\\begin{table}[t]\n\\centering\n\\caption{Share of training-period failures (\\%) preceded by each error code within 168 hours. "
             "For shares of at least 50\\%, the median and maximum time between the last such error and the failure are given in hours.}\n"
             "\\label{tab:coupling}\n\\small\n\\begin{tabular}{@{}lrlllll@{}}\n\\toprule\n"
             "Component & Failures & error1 & error2 & error3 & error4 & error5 \\\\\n\\midrule\n")
    for comp, n, cells in rows:
        fh.write(f"{comp} & {n} & " + " & ".join(cells) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print(open(OUT).read())
