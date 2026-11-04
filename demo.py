"""Live demo for the CHE629 project. Everything printed here is computed while you watch.

It calls the project's own modules (optimus.py, synth.py, bench.py, novelty/models.py, novelty/train.py)
exactly as the full pipeline does, just on a small dataset so it finishes in a few minutes on CPU.

usage:
  python3 demo.py                      # reproduce + train (synthetic data)
  python3 demo.py reproduce            # shipped weights -> predictions on the 20k real held-out test set
  python3 demo.py train                # generate synthetic data, train 5 models from scratch, compare
  python3 demo.py train --data real    # same, but on a subset of the real data (needs data/train.csv)
  python3 demo.py train --n 3000       # smaller / larger training set (default 5000)

What `train` does, live:
  1. builds a synthetic 5'UTR dataset with known rules (synth.py, ViennaRNA for MFE)
  2. trains k-mer ridge, LightGBM, BiLSTM, the Optimus 5-Prime CNN and the CNN+attention hybrid
  3. evaluates all of them with the shared metrics (bench.evaluate): r2, Spearman, RMSE, accuracy, F1, confusion matrix
  4. saves a predicted-vs-observed figure to figures/demo_scatter.png
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)                                    # so relative paths in the project work from anywhere
sys.path[:0] = [ROOT, os.path.join(ROOT, "novelty")]

import numpy as np, pandas as pd, torch

torch.set_num_threads(2)
from optimus import OptimusCNN, one_hot
from bench import BiLSTM, evaluate, tertile_edges, kmer_features, feature_matrix
from models import build
from train import fit, predict, Scaler

W = "weights"
BAR = "=" * 78


def head(s):
    print(f"\n{BAR}\n{s}\n{BAR}", flush=True)


# ------------------------------------------------------------------ 1. reproduction with shipped weights
def reproduce():
    head("1. REPRODUCTION  (shipped weights, live inference on the 20,000-UTR real held-out test set)")
    for f in ("cnn_full_s0.pt", "hybrid_full_s0.pt", "class_edges.json", "scaler_s0.json"):
        if not os.path.exists(f"{W}/{f}"):
            sys.exit(f"missing {W}/{f}")
    te = pd.read_csv("data/test.csv")
    x, y = one_hot(te.utr), te.rl.values
    edges = np.array(json.load(open(f"{W}/class_edges.json"))["edges"])
    sc = json.load(open(f"{W}/scaler_s0.json"))
    for label, arch, path, paper in (("Optimus 5-Prime CNN (paper)", "cnn", f"{W}/cnn_full_s0.pt", "0.93"),
                                     ("CNN + attention hybrid (ours)", "hybrid", f"{W}/hybrid_full_s0.pt", "--")):
        t0 = time.time()
        m = build(arch, 0.2)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        p = predict(m, x) * sc["sd"] + sc["mu"]
        r = evaluate(y, p, edges)
        print(f"{label:32s} r2={r['r2']:.3f} (paper {paper})  F1={r['f1_macro']:.3f}  "
              f"RMSE={r['rmse']:.3f}  [{time.time() - t0:.1f}s]", flush=True)


# ------------------------------------------------------------------ 2. data
def get_data(kind, n_train, n_test):
    if kind == "real":
        if not os.path.exists("data/train.csv"):
            sys.exit("data/train.csv not found. Create it with:  python3 00_prep.py <GSM3130435_egfp_unmod_1.csv.gz>\n"
                     "(or run without --data real to use the synthetic dataset)")
        tr = pd.read_csv("data/train.csv").sample(n_train, random_state=0).reset_index(drop=True)
        te = pd.read_csv("data/test.csv")
        print(f"real eGFP data: {len(tr):,} training UTRs (random subset of train.csv), {len(te):,} held-out test UTRs")
        return tr, te, None
    try:
        import RNA  # noqa: F401  (ViennaRNA, used by synth.py for MFE)
    except ImportError:
        sys.exit("ViennaRNA is needed to generate synthetic data:  pip install ViennaRNA\n"
                 "(or use --data real if you have data/train.csv)")
    from synth import make_dataset, NOISE_SD
    t0 = time.time()
    seqs, mfe, y_true, y_obs = make_dataset(n_train + n_test, seed=0)
    df = pd.DataFrame({"utr": seqs, "mfe": mfe, "rl_true": y_true, "rl": y_obs})
    te, tr = df.iloc[:n_test].reset_index(drop=True), df.iloc[n_test:].reset_index(drop=True)
    ceiling = np.var(te.rl_true) / (np.var(te.rl_true) + NOISE_SD ** 2)
    print(f"synthetic data generated in {time.time() - t0:.0f}s: {len(tr):,} train / {len(te):,} test UTRs "
          f"(rules known, so the best possible r2 on the noisy labels is {ceiling:.3f})")
    return tr, te, ceiling


# ------------------------------------------------------------------ 3. live training + comparison
def train_all(kind="synth", n_train=5000, n_test=5000):
    head(f"2. LIVE TRAINING  ({n_train:,} training sequences, models trained from scratch right now)")
    tr, te, ceiling = get_data(kind, n_train, n_test)
    ytr, yte = tr.rl.values.astype(np.float32), te.rl.values.astype(np.float32)
    edges = tertile_edges(ytr)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr)); n_val = int(0.15 * len(tr))
    i_val, i_fit = perm[:n_val], perm[n_val:]
    xtr, xte = one_hot(tr.utr.values), one_hot(te.utr.values)
    preds = {}

    # --- benchmark 1: position-specific k-mer ridge (bench.kmer_features), k and alpha chosen on validation
    t0 = time.time(); print("\n[k-mer linear]", flush=True)
    from sklearn.linear_model import Ridge
    best = None
    for k in (3, 4, 5):
        X = kmer_features(tr.utr, k)
        for a in (10, 100, 1000):
            m = Ridge(alpha=a, solver="sparse_cg").fit(X[i_fit], ytr[i_fit])
            v = np.corrcoef(ytr[i_val], m.predict(X[i_val]))[0, 1] ** 2
            print(f"  k={k} alpha={a:<5} val r2={v:.3f}", flush=True)
            if best is None or v > best[2]:
                best = (k, a, v)
    k, a, _ = best
    m = Ridge(alpha=a, solver="sparse_cg").fit(kmer_features(tr.utr, k), ytr)
    preds["k-mer linear"] = m.predict(kmer_features(te.utr, k))
    print(f"  chose k={k}, alpha={a}  [{time.time() - t0:.0f}s]", flush=True)

    # --- benchmark 2: LightGBM on hand-crafted features (bench.feature_matrix)
    try:
        import lightgbm as lgb
        if "mfe" not in tr.columns:
            raise ImportError("no mfe column")
        t0 = time.time(); print("\n[LightGBM]", flush=True)
        Ftr, Fte = feature_matrix(tr.utr, tr.mfe), feature_matrix(te.utr, te.mfe)
        g = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
        g.fit(Ftr.iloc[i_fit], ytr[i_fit], eval_set=[(Ftr.iloc[i_val], ytr[i_val])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        preds["LightGBM"] = g.predict(Fte)
        print(f"  best iteration {g.best_iteration_}  [{time.time() - t0:.0f}s]", flush=True)
    except ImportError as e:
        print(f"\n[LightGBM] skipped ({e})", flush=True)

    # --- neural models: same training loop (novelty/train.fit: Adam + early stopping on validation loss)
    def train_net(name, net, lr, max_epochs=20):
        print(f"\n[{name}]  {sum(p.numel() for p in net.parameters()):,} parameters", flush=True)
        sc = Scaler(ytr[i_fit])
        t0 = time.time()
        net, info = fit(net, xtr[i_fit], sc.fwd(ytr[i_fit]), xtr[i_val], sc.fwd(ytr[i_val]),
                        lr=lr, max_epochs=max_epochs, patience=5, seed=0, log=lambda s: print(s, flush=True))
        preds[name] = sc.inv(predict(net, xte))
        print(f"  best epoch {info['best_epoch']}  [{time.time() - t0:.0f}s]", flush=True)

    train_net("BiLSTM", BiLSTM(), 1e-3)
    train_net("Optimus 5-Prime CNN", OptimusCNN(), 1e-3)
    train_net("CNN + attention hybrid (ours)", build("hybrid", 0.1), 5e-4)

    # --- shared evaluation (bench.evaluate), identical for every model
    head("3. RESULTS  (held-out test set, never used for training or model selection)")
    rows, res = [], {}
    for name, p in preds.items():
        r = evaluate(yte, p, edges); res[name] = r
        row = {"Model": name, "r2": r["r2"], "Spearman": r["spearman"], "RMSE": r["rmse"],
               "Accuracy": r["accuracy"], "Precision": r["precision_macro"], "Recall": r["recall_macro"],
               "F1": r["f1_macro"]}
        if kind == "synth":
            row["r2 vs true MRL"] = float(np.corrcoef(te.rl_true, p)[0, 1] ** 2)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if ceiling is not None:
        print(f"\nnoise ceiling for r2 on this synthetic test set: {ceiling:.3f}")
    best_name = df.iloc[0]["Model"]
    print(f"\nconfusion matrix, best model ({best_name}); rows = true class, cols = predicted "
          f"(low / medium / high MRL):")
    print(np.array(res[best_name]["confusion_matrix"]))
    print("\n(small-data numbers: the full 260k-sequence runs in the report reach r2 ~0.92-0.94 on real data)")

    # --- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    n = len(preds)
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.4), sharex=True, sharey=True)
    for ax, (name, p) in zip(np.atleast_1d(axs), preds.items()):
        ax.scatter(yte, p, s=2, alpha=0.3)
        lo, hi = float(min(yte.min(), p.min())), float(max(yte.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(f"{name}\nr2={res[name]['r2']:.3f}", fontsize=8.5)
        ax.set_xlabel("observed MRL")
    np.atleast_1d(axs)[0].set_ylabel("predicted MRL")
    fig.tight_layout(); fig.savefig("figures/demo_scatter.png", dpi=150)
    print("\nsaved figures/demo_scatter.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="all", choices=["all", "reproduce", "train"])
    ap.add_argument("--data", default="synth", choices=["synth", "real"])
    ap.add_argument("--n", type=int, default=5000, help="training sequences (default 5000)")
    ap.add_argument("--n-test", type=int, default=5000, help="synthetic test sequences (default 5000)")
    a = ap.parse_args()
    if a.what in ("all", "reproduce"):
        reproduce()
    if a.what in ("all", "train"):
        train_all(a.data, a.n, a.n_test)
