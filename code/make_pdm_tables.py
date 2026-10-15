"""Tables, macros and figures of the PdM paper from evaluate.py outputs.

Usage: python make_pdm_tables.py <results_root> <paper_dir>
Expects <results_root>/lead{0,24,48}/metrics.json and curves.npy.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT, P = sys.argv[1], sys.argv[2]
os.makedirs(f"{P}/tables", exist_ok=True)
os.makedirs(f"{P}/figs", exist_ok=True)
LEADS = [0, 24, 48]
MODELS = [("logreg", "Logistic regression"), ("gbm", "Gradient boosting"), ("cnn", "1D-CNN"), ("tcn", "TCN"),
          ("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")]
MODS = [("events", "Events"), ("telemetry", "Telemetry"), ("both", "Both")]
NAMES = dict(MODELS)

metrics, curves = {}, {}
for lead in LEADS:
    path = f"{ROOT}/lead{lead}/metrics.json"
    if os.path.exists(path):
        m = json.load(open(path))
        metrics[lead] = {(r["model"], r["modality"]): r for r in m["rows"]}
        metrics[lead]["_ci"] = m["bootstrap_auprc_ci"]
        metrics[lead]["_meta"] = m
        curves[lead] = np.load(f"{ROOT}/lead{lead}/curves.npy", allow_pickle=True).item()

macros = {}
DIG = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))


def mname(s):
    return "".join(DIG.get(c, c) for c in s if c.isalnum())


def get(lead, model, mod, key):
    r = metrics.get(lead, {}).get((model, mod))
    return None if r is None else r.get(key)


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------- Table: AUPRC by model x modality x lead
lines = ["\\begin{table*}[t]", "\\centering",
         "\\caption{Test AUPRC by model, input modality and lead time (failure in the 24-hour window starting $\\Delta$ hours after the decision). "
         "Test failure rate 1.9\\%, so a random ranking scores about 0.02. Bold: best model per column.}",
         "\\label{tab:auprc}", "\\small", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{@{}lccccccccc@{}}", "\\toprule",
         " & \\multicolumn{3}{c}{$\\Delta = 0$ h} & \\multicolumn{3}{c}{$\\Delta = 24$ h} & \\multicolumn{3}{c}{$\\Delta = 48$ h} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
         "Model & " + " & ".join(["Events", "Telem.", "Both"] * 3) + " \\\\", "\\midrule"]
best = {}
for lead in LEADS:
    for mod, _ in MODS:
        vals = [(get(lead, m, mod, "auprc") or -1, m) for m, _ in MODELS]
        best[(lead, mod)] = max(vals)[1]
for m, name in MODELS:
    cells = []
    for lead in LEADS:
        for mod, _ in MODS:
            v = get(lead, m, mod, "auprc")
            sd = get(lead, m, mod, "auprc_sd")
            n = get(lead, m, mod, "n_seeds")
            s = fmt(v)
            if v is not None and n and n > 1:
                s += f"\\,$\\pm$\\,{sd:.3f}"
            if v is not None and best[(lead, mod)] == m:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
            if v is not None:
                macros[f"auprc{m}{mod}L{lead}"] = fmt(v)
    lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
open(f"{P}/tables/auprc.tex", "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: detailed metrics for one lead
def detail_table(lead, label):
    lines = ["\\begin{table*}[t]", "\\centering",
             f"\\caption{{Detailed test metrics for $\\Delta = {lead}$~h (both modalities unless stated). AUPRC is the mean over seeds where several were trained (TCN, GRU, LSTM, 1D-CNN and Transformer with both inputs at 24 and 48~h: three seeds), with the 95\\% machine-level bootstrap interval of the first seed; other metrics are seed means; "
             "R@5: share of failures among the five highest-risk machines of the day; Comp.: correct failed component among true failures; "
             "ECE before\\,$\\rightarrow$\\,after temperature scaling; WAPE of weekly cohort fault ratios; Train: wall-clock training time on a shared CPU.}",
             f"\\label{{{label}}}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
             "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{@{}llccccccccr@{}}", "\\toprule",
             "Model & Input & AUPRC [95\\% CI] & AUROC & F1 & R@5 & Comp. & ECE & Brier & WAPE & Train (s) \\\\", "\\midrule"]
    for m, name in MODELS:
        for mod in ("both", "events", "telemetry"):
            r = metrics.get(lead, {}).get((m, mod))
            if r is None or (mod != "both" and m not in ("gbm", "tcn")):
                continue
            ci = metrics[lead]["_ci"].get(f"{m}|{mod}")
            cis = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            lines.append(f"{name} & {mod} & {r['auprc']:.3f}{cis} & {r['auroc']:.3f} & {r['f1']:.3f} & {r['recall_at_k']:.3f} & "
                         f"{r['component_acc']:.3f} & {r['ece_raw']:.4f}\\,$\\rightarrow$\\,{r['ece_cal']:.4f} & {r['brier_cal']:.4f} & "
                         f"{r['cohort_wape']:.3f} & {r['fit_s']:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


for lead in LEADS:
    if lead in metrics:
        open(f"{P}/tables/detail_lead{lead}.tex", "w").write(detail_table(lead, f"tab:detail{lead}"))
        for key, r in metrics[lead].items():
            if not isinstance(key, tuple):
                continue
            m, mod = key
            for k in ("auroc", "f1", "recall_at_k", "component_acc", "ece_raw", "ece_cal", "brier_cal", "cohort_wape"):
                if r.get(k) is not None:
                    macros[f"{k}{m}{mod}L{lead}"] = fmt(r[k], 4 if k.startswith(("ece", "brier")) else 3)
            for ratio in (5, 10, 20, 50):
                for pol in ("threshold", "topk", "threshold_raw", "oracle"):
                    v = r.get(f"saving_{pol}_{ratio}")
                    if v is not None:
                        macros[f"save{pol}{ratio}{m}{mod}L{lead}"] = f"{100 * v:.1f}"
                        macros[f"save{pol}{ratio}{m}{mod}L{lead}abs"] = f"{abs(100 * v):.1f}"
            ci = metrics[lead]["_ci"].get(f"{m}|{mod}")
            if ci:
                macros[f"cilo{m}{mod}L{lead}"] = f"{ci[0]:.3f}"
                macros[f"cihi{m}{mod}L{lead}"] = f"{ci[1]:.3f}"

# ---------------------------------------------------------------- Table: decision costs
lines = ["\\begin{table}[t]", "\\centering",
         "\\caption{Maintenance cost saving relative to the reactive policy (\\%, higher is better) on the test period, with capacity $K_{\\max}=5$ visits per day, "
         "for cost ratios $C_f/C_d$. Thr.: threshold policy with calibrated (cal.) or uncalibrated (raw) probabilities; Top-5: dispatch the five highest-risk machines daily. Best model with both modalities per lead time.}",
         "\\label{tab:cost}", "\\small", "\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
         "Lead & Policy & 5 & 10 & 20 & 50 \\\\", "\\midrule"]
for lead in LEADS:
    if lead not in metrics:
        continue
    bm = best[(lead, "both")]
    r = metrics[lead][(bm, "both")]
    for k, (pol, pname) in enumerate((("threshold", "Thr. (cal.)"), ("threshold_raw", "Thr. (raw)"), ("topk", "Top-5"), ("oracle", "Oracle"))):
        first = f"\\multirow{{4}}{{*}}{{\\makecell[l]{{$\\Delta={lead}$\\\\{NAMES[bm]}}}}}" if k == 0 else ""
        vals = [f"{100 * r[f'saving_{pol}_{ratio}']:.1f}" for ratio in (5, 10, 20, 50)]
        lines.append(f"{first} & {pname} & " + " & ".join(vals) + " \\\\")
    if lead != LEADS[-1]:
        lines.append("\\midrule")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(f"{P}/tables/cost.tex", "w").write("\n".join(lines) + "\n")
for lead in LEADS:
    if lead in metrics:
        macros[f"bestbothL{lead}"] = NAMES[best[(lead, 'both')]]
        macros[f"besteventsL{lead}"] = NAMES[best[(lead, 'events')]]
        macros[f"besttelemetryL{lead}"] = NAMES[best[(lead, 'telemetry')]]

# ---------------------------------------------------------------- Figure: AUPRC vs lead
fig, ax = plt.subplots(figsize=(3.5, 2.4))
styles = {"events": ("#1f5f99", "o"), "telemetry": ("#c7a252", "s"), "both": ("#2e7d4f", "^")}
for mod, name in MODS:
    ys = [get(lead, best.get((lead, mod)), mod, "auprc") if lead in metrics else None for lead in LEADS]
    xs = [lead for lead, y in zip(LEADS, ys) if y is not None]
    ys = [y for y in ys if y is not None]
    if ys:
        ax.plot(xs, ys, marker=styles[mod][1], color=styles[mod][0], label=f"{name} (best model)", lw=1.4, ms=4)
ax.axhline(0.0192, color="#999", lw=0.8, ls="--")
ax.text(47, 0.04, "random", fontsize=6.5, color="#777", ha="right")
ax.set_xticks(LEADS)
ax.set_xlabel("Lead time $\\Delta$ (hours)", fontsize=8)
ax.set_ylabel("Test AUPRC", fontsize=8)
ax.set_ylim(0, 1.02)
ax.tick_params(labelsize=7)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=6.5, frameon=False)
fig.tight_layout()
fig.savefig(f"{P}/figs/fig_lead.pdf")
fig.savefig(f"{P}/figs/fig_lead.png", dpi=200)

# ---------------------------------------------------------------- Figure: reliability diagrams
if 24 in curves:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, (model, mod) in zip(axes, (("logreg", "events"), ("gbm", "both"))):
        key = (model, mod)
        if key not in curves[24]["calib"]:
            continue
        r_raw, r_cal, y = curves[24]["calib"][key]
        for r, lab, col in ((r_raw, "before scaling", "#c7a252"), (r_cal, "after scaling", "#1f5f99")):
            edges = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mk = (r >= lo) & (r <= hi)
                if mk.sum() >= 20:
                    xs.append(r[mk].mean())
                    ys.append(y[mk].mean())
            ax.plot(xs, ys, marker="o", ms=3, lw=1.2, color=col, label=lab)
        ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{NAMES[model]}, {mod} inputs, $\\Delta=24$ h", fontsize=8)
        ax.set_xlabel("Predicted risk (bin means, bins with $\\geq$20 samples)", fontsize=7.5)
        ax.set_ylabel("Observed failure rate", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{P}/figs/fig_reliability.pdf")
    fig.savefig(f"{P}/figs/fig_reliability.png", dpi=200)

# ---------------------------------------------------------------- Figure: PR curves for one lead
for lead in LEADS:
    if lead not in curves:
        continue
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    cols = ["#9e9e9e", "#c7a252", "#8e44ad", "#1f5f99", "#d9534f", "#2e7d4f", "#555555"]
    for (m, name), col in zip(MODELS, cols):
        key = (m, "both")
        if key in curves[lead]["pr"]:
            pr, rc = curves[lead]["pr"][key]
            ax.plot(rc, pr, lw=1.1, color=col, label=name)
    ax.set_xlabel("Recall", fontsize=8)
    ax.set_ylabel("Precision", fontsize=8)
    ax.set_title(f"Both inputs, $\\Delta={lead}$ h", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{P}/figs/fig_pr_lead{lead}.pdf")
    fig.savefig(f"{P}/figs/fig_pr_lead{lead}.png", dpi=200)

# ---------------------------------------------------------------- Figure: cost saving vs ratio
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ratios = [5, 10, 20, 50]
for lead, col in zip(LEADS, ["#2e7d4f", "#1f5f99", "#c7a252"]):
    if lead not in metrics:
        continue
    r = metrics[lead][(best[(lead, "both")], "both")]
    ax.plot(ratios, [100 * r[f"saving_threshold_{x}"] for x in ratios], marker="o", ms=3.5, color=col,
            label=f"$\\Delta={lead}$ h, calibrated")
    ax.plot(ratios, [100 * r[f"saving_threshold_raw_{x}"] for x in ratios], marker="x", ms=3.5, color=col, ls=":",
            label=f"$\\Delta={lead}$ h, uncalibrated")
ax.axhline(0, color="#999", lw=0.8)
ax.set_xscale("log")
ax.set_xticks(ratios)
ax.set_xticklabels([str(x) for x in ratios])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Cost ratio $C_f / C_d$", fontsize=8)
ax.set_ylabel("Cost saving vs. reactive (%)", fontsize=8)
ax.tick_params(labelsize=7)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=5.8, frameon=False, ncol=1)
fig.tight_layout()
fig.savefig(f"{P}/figs/fig_cost.pdf")
fig.savefig(f"{P}/figs/fig_cost.png", dpi=200)

with open(f"{P}/tables/macros.tex", "w") as fh:
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{mname(k)}}}{{{v}}}\n")
print(len(macros), "macros")
