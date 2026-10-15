"""Build decision-point samples from the Microsoft Azure Predictive Maintenance dataset.

One sample per machine per day, at 06:00 (decision time t). Only records with timestamp <= t are
used as inputs. Targets refer to the window (t + LEAD, t + LEAD + H]: LEAD = 0 predicts failures in
the next H hours, LEAD = 24 predicts failures on the day after next, and so on.

Outputs (npz):
  seq_events   (N, L, 18)  hourly error indicators (5), maintenance indicators (4),
                           log-days since the last error of each code (5) and since the last
                           maintenance of each component (4)
  seq_telemetry (N, L, 4)  hourly volt, rotate, pressure, vibration (raw; standardised later)
  static       (N, 5)      machine model one-hot (4) and age in years
  tab_*        tabular features for the gradient-boosting and logistic-regression baselines
  y_bin        failure of any component in (t, t + H]
  y_cls        0 = no failure, 1..4 = failed component (comp1..comp4; the lowest if several)
  y_set        (N, 4) multi-hot of failed components
  machine, model, day   identifiers for splitting and cohort analysis
"""
import sys
import numpy as np
import pandas as pd

DATA = sys.argv[1]
OUT = sys.argv[2]
HORIZON_H = int(sys.argv[3]) if len(sys.argv) > 3 else 24
LEAD_H = int(sys.argv[4]) if len(sys.argv) > 4 else 0
L = 168  # hours of history per sample (7 days)

ERR = [f"error{i}" for i in range(1, 6)]
COMP = [f"comp{i}" for i in range(1, 5)]

tel = pd.read_csv(f"{DATA}/PdM_telemetry.csv", parse_dates=["datetime"])
err = pd.read_csv(f"{DATA}/PdM_errors.csv", parse_dates=["datetime"])
mnt = pd.read_csv(f"{DATA}/PdM_maint.csv", parse_dates=["datetime"])
fai = pd.read_csv(f"{DATA}/PdM_failures.csv", parse_dates=["datetime"])
mac = pd.read_csv(f"{DATA}/PdM_machines.csv")

start = pd.Timestamp("2015-01-01 06:00")
hours = pd.date_range(start, "2016-01-01 06:00", freq="h")
H = len(hours)
idx = {h: i for i, h in enumerate(hours)}
n_mach = mac.machineID.max()

# Hourly grids per machine.
telem = np.full((n_mach, H, 4), np.nan, dtype=np.float32)
for row in tel.itertuples(index=False):
    i = idx.get(row.datetime)
    if i is not None:
        telem[row.machineID - 1, i] = (row.volt, row.rotate, row.pressure, row.vibration)
# Forward-fill any gap (the dataset is complete, this is a safeguard).
for m in range(n_mach):
    df = pd.DataFrame(telem[m]).ffill().bfill()
    telem[m] = df.values

err_ind = np.zeros((n_mach, H, 5), dtype=np.float32)
for row in err.itertuples(index=False):
    i = idx.get(row.datetime)
    if i is not None:
        err_ind[row.machineID - 1, i, ERR.index(row.errorID)] = 1

mnt_ind = np.zeros((n_mach, H, 4), dtype=np.float32)
for row in mnt.itertuples(index=False):
    i = idx.get(row.datetime)
    if i is not None:
        mnt_ind[row.machineID - 1, i, COMP.index(row.comp)] = 1


def recency(ind, prior_last_hours):
    """Hours since the last event of each channel, at every hour (cap 365 days)."""
    n, T, C = ind.shape
    out = np.zeros_like(ind)
    last = prior_last_hours.astype(np.float32).copy()  # (n, C), negative = hours before start
    for t in range(T):
        last = np.where(ind[:, t] > 0, t, last)
        out[:, t] = np.minimum(t - last, 365 * 24)
    return out


# Maintenance before 2015 (from 2014-06) gives the initial recency of each component.
prior_m = np.full((n_mach, 4), -365 * 24, dtype=np.float32)
pre = mnt[mnt.datetime < start]
for row in pre.itertuples(index=False):
    h = (row.datetime - start) / pd.Timedelta(hours=1)
    prior_m[row.machineID - 1, COMP.index(row.comp)] = max(prior_m[row.machineID - 1, COMP.index(row.comp)], h)
prior_e = np.full((n_mach, 5), -30 * 24, dtype=np.float32)

rec_e = recency(err_ind, prior_e)
rec_m = recency(mnt_ind, prior_m)
log_days = lambda x: np.log1p(x / 24.0)
events = np.concatenate([err_ind, mnt_ind, log_days(rec_e), log_days(rec_m)], axis=2)

# Failure targets.
fail_set = np.zeros((n_mach, H, 4), dtype=np.float32)
for row in fai.itertuples(index=False):
    i = idx.get(row.datetime)
    if i is not None:
        fail_set[row.machineID - 1, i, COMP.index(row.failure)] = 1

models = sorted(mac.model.unique())
static = np.zeros((n_mach, 5), dtype=np.float32)
for row in mac.itertuples(index=False):
    static[row.machineID - 1, models.index(row.model)] = 1
    static[row.machineID - 1, 4] = row.age

# Decision points: every day at 06:00 with a full week of history and a full horizon.
S_ev, S_te, S_st, T_rows, Yb, Yc, Ys, M, MOD, DAY = [], [], [], [], [], [], [], [], [], []
for t in range(L - 1, H - HORIZON_H - LEAD_H, 24):
    for m in range(n_mach):
        sl = slice(t - L + 1, t + 1)
        S_ev.append(events[m, sl])
        S_te.append(telem[m, sl])
        S_st.append(static[m])
        fs = fail_set[m, t + 1 + LEAD_H: t + 1 + LEAD_H + HORIZON_H].max(axis=0)
        Ys.append(fs)
        Yb.append(float(fs.max() > 0))
        Yc.append(int(np.argmax(fs)) + 1 if fs.max() > 0 else 0)
        M.append(m + 1)
        MOD.append(int(np.argmax(static[m, :4])))
        DAY.append(hours[t])
        # Tabular features.
        f = {}
        for w in (24, 72, 168):
            ws = slice(t - w + 1, t + 1)
            for c in range(5):
                f[f"err{c+1}_cnt_{w}h"] = err_ind[m, ws, c].sum()
            for c in range(4):
                f[f"mnt{c+1}_cnt_{w}h"] = mnt_ind[m, ws, c].sum()
            for k, name in enumerate(["volt", "rotate", "pressure", "vibration"]):
                x = telem[m, ws, k]
                f[f"{name}_mean_{w}h"] = x.mean()
                f[f"{name}_std_{w}h"] = x.std()
                if w == 24:
                    f[f"{name}_min_{w}h"] = x.min()
                    f[f"{name}_max_{w}h"] = x.max()
        for c in range(5):
            f[f"err{c+1}_days_since"] = rec_e[m, t, c] / 24
        for c in range(4):
            f[f"mnt{c+1}_days_since"] = rec_m[m, t, c] / 24
        for k in range(4):
            f[f"model{k+1}"] = static[m, k]
        f["age"] = static[m, 4]
        T_rows.append(f)

tab = pd.DataFrame(T_rows)
ev_cols = [c for c in tab.columns if c.startswith(("err", "mnt"))] + [f"model{k+1}" for k in range(4)] + ["age"]
te_cols = [c for c in tab.columns if c.startswith(("volt", "rotate", "pressure", "vibration"))] + [f"model{k+1}" for k in range(4)] + ["age"]

np.savez_compressed(
    OUT,
    seq_events=np.stack(S_ev).astype(np.float32),
    seq_telemetry=np.stack(S_te).astype(np.float32),
    static=np.stack(S_st).astype(np.float32),
    tab=tab.values.astype(np.float32),
    tab_cols=np.array(tab.columns),
    tab_event_cols=np.array(ev_cols),
    tab_telemetry_cols=np.array(te_cols),
    y_bin=np.array(Yb, dtype=np.float32),
    y_cls=np.array(Yc, dtype=np.int64),
    y_set=np.stack(Ys).astype(np.float32),
    machine=np.array(M),
    model=np.array(MOD),
    day=np.array(DAY, dtype="datetime64[h]"),
)
print("samples", len(Yb), "positives", int(sum(Yb)), "rate %.4f" % np.mean(Yb), "horizon", HORIZON_H, "lead", LEAD_H)
