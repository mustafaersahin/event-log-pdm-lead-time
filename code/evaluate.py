"""Metrics, decision-policy simulation and figures for the PdM benchmark.

Usage: python evaluate.py <npz> <runs_dir> <out_dir>
Reads every pred_<model>_<modality>_s<seed>.npz in runs_dir.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score

NPZ, RUNS, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
d = np.load(NPZ, allow_pickle=True)
day, machine, model_id = d["day"], d["machine"], d["model"]
y_bin, y_cls, y_set = d["y_bin"], d["y_cls"], d["y_set"]
val = (day >= np.datetime64("2015-09-01")) & (day < np.datetime64("2015-11-01"))
test = day >= np.datetime64("2015-11-01")
K_BUDGET = 5        # technician visits per day for a fleet of 100 machines
COST_RATIOS = [5, 10, 20, 50]  # unplanned failure cost / preventive dispatch cost


def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def recall_at_k(risk, y, days, k=K_BUDGET):
    """Share of failures whose machine is among the k highest-risk machines on that day."""
    hit = tot = 0
    for dd in np.unique(days):
        m = days == dd
        if y[m].sum() == 0:
            continue
        top = np.argsort(-risk[m])[:k]
        hit += y[m][top].sum()
        tot += y[m].sum()
    return hit / tot


def simulate(risk, y, days, ratio, policy, k=K_BUDGET):
    """Total cost with dispatch cost 1 and unplanned-failure cost `ratio`.

    A dispatch on a day whose 24 h window contains a failure prevents that failure.
    threshold: dispatch every machine with risk >= 1/ratio (expected-cost rule), at most k per day.
    topk: always dispatch the k highest-risk machines.  reactive: never dispatch.
    oracle: dispatch exactly the failing machines (capacity k).
    """
    cost = 0.0
    for dd in np.unique(days):
        m = np.where(days == dd)[0]
        r, yy = risk[m], y[m]
        if policy == "reactive":
            chosen = np.array([], dtype=int)
        elif policy == "oracle":
            chosen = np.where(yy > 0)[0][:k]
        elif policy == "topk":
            chosen = np.argsort(-r)[:k]
        else:
            cand = np.where(r >= 1.0 / ratio)[0]
            chosen = cand[np.argsort(-r[cand])][:k]
        prevented = yy[chosen].sum()
        cost += len(chosen) + ratio * (yy.sum() - prevented)
    return cost


def cohort_fault_ratio(risk, idx):
    """Weekly fault ratio per machine model: observed vs predicted (WAPE)."""
    weeks = (day[idx] - np.datetime64("2015-11-01")).astype("timedelta64[D]").astype(int) // 7
    obs_all, pred_all = [], []
    for wk in np.unique(weeks):
        for c in range(4):
            sel = (weeks == wk) & (model_id[idx] == c)
            if not sel.any():
                continue
            ms = machine[idx][sel]
            obs, pred = [], []
            for mm in np.unique(ms):
                s2 = ms == mm
                obs.append(y_bin[idx][sel][s2].max())
                pred.append(1 - np.prod(1 - risk[sel][s2]))
            obs_all.append(np.mean(obs))
            pred_all.append(np.mean(pred))
    obs_all, pred_all = np.array(obs_all), np.array(pred_all)
    return float(np.abs(obs_all - pred_all).sum() / obs_all.sum()), float(np.abs(obs_all - pred_all).mean())


runs = defaultdict(list)
for f in sorted(glob.glob(f"{RUNS}/pred_*.npz")):
    mdl, mod, seed = re.match(r".*pred_(\w+?)_(events|telemetry|both)_s(\d+)\.npz", f).groups()
    runs[(mdl, mod)].append(f)

rows, curves, calib = [], {}, {}
for (mdl, mod), files in sorted(runs.items()):
    per_seed = defaultdict(list)
    for f in files:
        z = np.load(f)
        idx = z["index"]
        pos = {i: k for k, i in enumerate(idx)}
        te = np.where(test)[0]
        k = np.array([pos[i] for i in te])
        p_raw, p_cal = z["p_raw"][k], z["p_cal"][k]
        r_raw, r_cal = 1 - p_raw[:, 0], 1 - p_cal[:, 0]
        yb = y_bin[te]
        vk = np.array([pos[i] for i in np.where(val)[0]])
        r_val = 1 - z["p_cal"][vk][:, 0]
        # Threshold maximising F1 on validation.
        pr, rc, th = precision_recall_curve(y_bin[val], r_val)
        f1s = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
        thr = th[np.argmax(f1s[:-1])]
        # Component identification among true failures: top-1 component in the failed set.
        fail = yb > 0
        comp_pred = np.argmax(p_cal[fail][:, 1:], axis=1)
        comp_hit = y_set[te][fail][np.arange(fail.sum()), comp_pred].mean()
        meta = json.load(open(f.replace("pred_", "meta_").replace(".npz", ".json")))
        m = {
            "auprc": average_precision_score(yb, r_cal),
            "auroc": roc_auc_score(yb, r_cal),
            "f1": f1_score(yb, r_cal >= thr),
            "recall_at_k": recall_at_k(r_cal, yb, day[te]),
            "component_acc": comp_hit,
            "ece_raw": ece(r_raw, yb),
            "ece_cal": ece(r_cal, yb),
            "brier_cal": float(np.mean((r_cal - yb) ** 2)),
            "fit_s": meta["fit_seconds"],
            "infer_ms": meta["inference_ms_per_sample"],
            "n_params": meta["n_params"],
        }
        wape, mae = cohort_fault_ratio(r_cal, te)
        m["cohort_wape"], m["cohort_mae"] = wape, mae
        for ratio in COST_RATIOS:
            base = simulate(r_cal, yb, day[te], ratio, "reactive")
            for pol in ("threshold", "topk"):
                m[f"saving_{pol}_{ratio}"] = 1 - simulate(r_cal, yb, day[te], ratio, pol) / base
            m[f"saving_threshold_raw_{ratio}"] = 1 - simulate(r_raw, yb, day[te], ratio, "threshold") / base
            m[f"saving_oracle_{ratio}"] = 1 - simulate(r_cal, yb, day[te], ratio, "oracle") / base
        for kk, v in m.items():
            per_seed[kk].append(v)
        if meta["seed"] == 0:
            curves[(mdl, mod)] = precision_recall_curve(yb, r_cal)[:2]
            calib[(mdl, mod)] = (r_raw, r_cal, yb)
    row = {"model": mdl, "modality": mod, "n_seeds": len(files)}
    for kk, v in per_seed.items():
        v = [x for x in v if x is not None]
        row[kk] = float(np.mean(v)) if v else None
        row[kk + "_sd"] = float(np.std(v)) if len(v) > 1 else 0.0
    rows.append(row)

# Machine-level block bootstrap of AUPRC for seed-0 predictions (95% CI).
boot = {}
te = np.where(test)[0]
mach_te = machine[te]
uniq = np.unique(mach_te)
rng = np.random.default_rng(0)
groups = {mm: np.where(mach_te == mm)[0] for mm in uniq}
draws = [np.concatenate([groups[mm] for mm in rng.choice(uniq, len(uniq))]) for _ in range(1000)]
for key, (r_raw, r_cal, yb) in calib.items():
    vals = [average_precision_score(yb[b], r_cal[b]) for b in draws if yb[b].sum() > 0]
    boot[f"{key[0]}|{key[1]}"] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

json.dump({"rows": rows, "bootstrap_auprc_ci": boot, "test_positives": int(y_bin[test].sum()),
           "test_samples": int(test.sum()), "k_budget": K_BUDGET, "cost_ratios": COST_RATIOS},
          open(f"{OUT}/metrics.json", "w"), indent=2)
np.save(f"{OUT}/curves.npy", {"pr": curves, "calib": calib}, allow_pickle=True)
for r in rows:
    print(f"{r['model']:12s} {r['modality']:9s} n={r['n_seeds']} AUPRC={r['auprc']:.3f}±{r['auprc_sd']:.3f} "
          f"AUROC={r['auroc']:.3f} R@5={r['recall_at_k']:.3f} comp={r['component_acc']:.3f} "
          f"ECE {r['ece_raw']:.4f}->{r['ece_cal']:.4f} save10={r['saving_threshold_10']:.3f}")
