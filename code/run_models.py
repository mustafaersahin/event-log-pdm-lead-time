"""Train and evaluate failure-risk and failure-class models on the prepared PdM samples.

Usage: python run_models.py <npz> <out_dir> <model> <modality> <seed>
  model:    logreg | gbm | cnn | tcn | lstm | gru | transformer
  modality: events | telemetry | both

Temporal split by decision date: train < 2015-09-01 <= validation < 2015-11-01 <= test.
Every model predicts a 5-way distribution (no failure, comp1..comp4); the failure risk is
1 - P(no failure). Probabilities are calibrated by temperature scaling on the validation set.
Saves validation and test predictions (raw and calibrated) and the metrics as JSON.
"""
import json
import os
import sys
import time

import numpy as np

NPZ, OUT, MODEL, MODALITY, SEED = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

d = np.load(NPZ, allow_pickle=True)
day = d["day"]
train = day < np.datetime64("2015-09-01")
val = (day >= np.datetime64("2015-09-01")) & (day < np.datetime64("2015-11-01"))
test = day >= np.datetime64("2015-11-01")
y_cls, y_bin, y_set = d["y_cls"], d["y_bin"], d["y_set"]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, y):
    """Temperature minimising the validation negative log-likelihood (grid + refinement)."""
    def nll(T):
        p = softmax(logits / T)
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 200))
    best = grid[np.argmin([nll(T) for T in grid])]
    fine = np.linspace(best * 0.8, best * 1.25, 100)
    return float(fine[np.argmin([nll(T) for T in fine])])


# ---------------------------------------------------------------------------------------------
# Tabular baselines
# ---------------------------------------------------------------------------------------------
def tabular_logits():
    cols = list(d["tab_cols"])
    keep = {"events": list(d["tab_event_cols"]), "telemetry": list(d["tab_telemetry_cols"]),
            "both": cols}[MODALITY]
    X = d["tab"][:, [cols.index(c) for c in keep]]
    if MODEL == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED))
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, class_weight="balanced", random_state=SEED)
    t0 = time.time()
    clf.fit(X[train], y_cls[train])
    fit_s = time.time() - t0
    proba = clf.predict_proba(X)
    full = np.full((len(X), 5), 1e-9)
    full[:, clf.classes_] = proba
    t0 = time.time()
    clf.predict_proba(X[test])
    infer_ms = 1000 * (time.time() - t0) / test.sum()
    n_params = None
    logits = np.log(full + 1e-12)
    # Balanced class weights act like resampling: class c is weighted by n / (5 n_c), which
    # shifts the predicted odds. The shift is undone analytically before temperature scaling
    # (prior correction), so that "raw" is the model's direct output and "calibrated" is the
    # prior-corrected, temperature-scaled output.
    counts = np.bincount(y_cls[train], minlength=5).astype(np.float64)
    weights = counts.sum() / (5 * np.maximum(counts, 1))
    return logits, fit_s, infer_ms, n_params, -np.log(weights)


# ---------------------------------------------------------------------------------------------
# Sequence models
# ---------------------------------------------------------------------------------------------
def sequence_logits():
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

    parts = []
    if MODALITY in ("events", "both"):
        parts.append(d["seq_events"])
    if MODALITY in ("telemetry", "both"):
        parts.append(d["seq_telemetry"])
    X = np.concatenate(parts, axis=2)
    # Standardise each channel with training statistics.
    mu = X[train].reshape(-1, X.shape[2]).mean(axis=0)
    sd = X[train].reshape(-1, X.shape[2]).std(axis=0) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    S = d["static"].copy()
    S[:, 4] = (S[:, 4] - S[train, 4].mean()) / (S[train, 4].std() + 1e-6)
    S = S.astype(np.float32)
    C, L = X.shape[2], X.shape[1]

    class Head(nn.Module):
        def __init__(self, feat):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(feat + 5, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 5))

        def forward(self, h, s):
            return self.mlp(torch.cat([h, s], dim=1))

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(C, 32, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
                nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64))
            self.head = Head(128)

        def forward(self, x, s):
            h = self.net(x.transpose(1, 2))
            return self.head(torch.cat([h.amax(dim=2), h.mean(dim=2)], dim=1), s)

    class TCNBlock(nn.Module):
        def __init__(self, cin, cout, dil):
            super().__init__()
            self.pad = 2 * dil
            self.c1 = nn.Conv1d(cin, cout, 3, dilation=dil)
            self.c2 = nn.Conv1d(cout, cout, 3, dilation=dil)
            self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
            self.drop = nn.Dropout(0.1)

        def forward(self, x):
            y = torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0))))
            y = self.drop(torch.relu(self.c2(nn.functional.pad(self.drop(y), (self.pad, 0)))))
            return torch.relu(y + self.down(x))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, cin = [], C
            for dil in (1, 2, 4, 8, 16, 32):  # receptive field 253 h > 168 h
                layers.append(TCNBlock(cin, 48, dil))
                cin = 48
            self.net = nn.Sequential(*layers)
            self.head = Head(96)

        def forward(self, x, s):
            h = self.net(x.transpose(1, 2))
            return self.head(torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1), s)

    class RNN(nn.Module):
        def __init__(self, cell):
            super().__init__()
            self.rnn = (nn.LSTM if cell == "lstm" else nn.GRU)(C, 64, batch_first=True)
            self.head = Head(64)

        def forward(self, x, s):
            out, _ = self.rnn(x)
            return self.head(out[:, -1], s)

    class Transformer(nn.Module):
        def __init__(self, patch=6, dim=64):
            super().__init__()
            self.patch = patch
            self.proj = nn.Linear(C * patch, dim)
            self.pos = nn.Parameter(torch.zeros(1, L // patch, dim))
            layer = nn.TransformerEncoderLayer(dim, 4, 128, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
            self.head = Head(dim)

        def forward(self, x, s):
            b = x.shape[0]
            tokens = self.proj(x.reshape(b, L // self.patch, self.patch * C)) + self.pos
            h = self.enc(tokens)
            return self.head(h.mean(dim=1), s)

    net = {"cnn": CNN, "tcn": TCN, "lstm": lambda: RNN("lstm"), "gru": lambda: RNN("gru"),
           "transformer": Transformer}[MODEL]()
    n_params = sum(p.numel() for p in net.parameters())

    # Imbalance: each epoch uses every failure sample and a random NEG_RATE share of the
    # no-failure samples. The resulting prior shift is undone at prediction time by adding
    # log(1 / NEG_RATE) to the no-failure logit, so the outputs remain probabilities.
    NEG_RATE = 0.25
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    Xt, St, Yt = torch.tensor(X), torch.tensor(S), torch.tensor(y_cls)
    tr_pos = np.where(train & (y_cls > 0))[0]
    tr_neg = np.where(train & (y_cls == 0))[0]
    prior_fix = np.zeros(5, dtype=np.float32)
    prior_fix[0] = np.log(1.0 / NEG_RATE)

    def predict(ix, bs=1024):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(ix), bs):
                b = ix[i:i + bs]
                out.append(net(Xt[b], St[b]).numpy() + prior_fix)
        return np.concatenate(out)

    from sklearn.metrics import average_precision_score
    best, best_state, patience = -1, None, 0
    t0 = time.time()
    va_idx = np.where(val)[0]
    for epoch in range(20):
        net.train()
        neg = rng.choice(tr_neg, int(NEG_RATE * len(tr_neg)), replace=False)
        perm = rng.permutation(np.concatenate([tr_pos, neg]))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b], St[b]), Yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        p = softmax(predict(va_idx))
        ap = average_precision_score(y_bin[val], 1 - p[:, 0])
        if ap > best + 1e-4:
            best, patience = ap, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                break
    fit_s = time.time() - t0
    net.load_state_dict(best_state)
    logits = predict(np.arange(len(X)))
    # Single-sample latency, as in online scoring of one machine.
    te = np.where(test)[0]
    t0 = time.time()
    predict(te[:2000], bs=1)
    infer_ms = 1000 * (time.time() - t0) / 2000
    # The undersampling correction is already part of predict(); nothing further to undo.
    return logits, fit_s, infer_ms, n_params, np.zeros(5)


logits, fit_s, infer_ms, n_params, prior_shift = tabular_logits() if MODEL in ("logreg", "gbm") else sequence_logits()
corrected = logits + prior_shift
T = fit_temperature(corrected[val], y_cls[val])
p_raw, p_cal = softmax(logits), softmax(corrected / T)

np.savez_compressed(
    f"{OUT}/pred_{MODEL}_{MODALITY}_s{SEED}.npz",
    p_raw=p_raw[val | test].astype(np.float32), p_cal=p_cal[val | test].astype(np.float32),
    index=np.where(val | test)[0], temperature=T)
json.dump({"model": MODEL, "modality": MODALITY, "seed": SEED, "temperature": T,
           "fit_seconds": fit_s, "inference_ms_per_sample": infer_ms, "n_params": n_params},
          open(f"{OUT}/meta_{MODEL}_{MODALITY}_s{SEED}.json", "w"), indent=2)
print(MODEL, MODALITY, SEED, "T=%.2f fit=%.0fs" % (T, fit_s))
