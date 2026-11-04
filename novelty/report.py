"""Collect saved results into tables (markdown/CSV) and figures. usage: python3 novelty/report.py {A|B|C|D|all}"""
import glob, json, os, pickle, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = "results/novelty", "figures/novelty"
os.makedirs(F, exist_ok=True)
INK, MUTED, GRID = "#1f1f1e", "#73726c", "#e4e3dd"
COL = {"cnn": "#2a78d6", "transformer": "#eb6834", "hybrid": "#1baf7a",
       "scratch": "#2a78d6", "finetune": "#eb6834", "multi": "#1baf7a"}
CHEMS, CLAB = ["U", "psi", "m1psi"], {"U": "U", "psi": "Ψ", "m1psi": "m1Ψ"}
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11.5,
                     "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.bbox": "tight"})


def ms(v, d=3):
    return f"{np.mean(v):.{d}f} ± {np.std(v):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def r2(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return np.corrcoef(a[m], b[m])[0, 1] ** 2


# ------------------------------------------------------------------------------------------------ Part B
def collect_b():
    rows = []
    for size in ("1k", "5k", "full"):
        for arch in ("cnn", "transformer", "hybrid"):
            cf = f"{R}/partB/{size}/{arch}_choice.json"
            if not os.path.exists(cf):
                continue
            best = json.load(open(cf))["chosen"]
            for p in sorted(glob.glob(f"{R}/partB/{size}/{arch}_{size}_{best}_s?.json")):
                j = json.load(open(p))
                rows.append({"size": size, "n_train_total": {"1k": 1000, "5k": 5000, "full": 260000}[size], "arch": arch,
                             "config": best, "seed": j["seed"], "params": j["params"], "best_epoch": j["best_epoch"],
                             **{k: j["test"][k] for k in ("r2", "pearson", "spearman", "rmse")}})
    return pd.DataFrame(rows)


def part_b():
    df = collect_b()
    df.to_csv(f"{R}/partB/all_runs.csv", index=False)
    g = df.groupby(["size", "arch"], sort=False)
    with open(f"{R}/partB/summary.md", "w") as f:
        f.write("U data, fixed 20k high-read test set. Mean ± sd over seeds (5 at 1k/5k, 3 at full).\n\n"
                "| Size | Architecture | Params | Config (val-chosen) | Seeds | r² | Pearson | Spearman | RMSE | Best epoch |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for (size, arch), d in g:
            f.write(f"| {size} | {arch} | {d.params.iloc[0]:,} | {d.config.iloc[0]} | {len(d)} | {ms(d.r2.values)} | {ms(d.pearson.values)} | "
                    f"{ms(d.spearman.values)} | {ms(d.rmse.values)} | {ms(d.best_epoch.values, 1)} |\n")
        rs = sorted(glob.glob(f"{R}/partB/randsplit/*_s0.json"))
        if rs:
            f.write("\nRandom-split sanity check (full data, random 20k test, seed 0 only)\n\n| Architecture | r² random split | r² high-read split (same seed) |\n|---|---|---|\n")
            for p in rs:
                j = json.load(open(p)); arch = j["name"].split("_")[0]
                hr = df[(df.arch == arch) & (df["size"] == "full") & (df.seed == 0)].r2
                f.write(f"| {arch} | {j['test']['r2']:.3f} | {hr.iloc[0]:.3f} |\n" if len(hr) else f"| {arch} | {j['test']['r2']:.3f} | – |\n")
    sens = []
    for size in ("1k", "5k"):
        for arch in ("cnn", "transformer", "hybrid"):
            ps = sorted(glob.glob(f"{R}/partB/sens_patience30/{size}/{arch}_{size}_*_s?.json"))
            if ps:
                sens.append((size, arch, [json.load(open(p))["test"]["r2"] for p in ps]))
    if sens:
        with open(f"{R}/partB/summary.md", "a") as f:
            f.write("\nEarly-stopping sensitivity (same configs and subsets; patience 30 instead of 10, max 300 epochs; all architectures)\n\n"
                    "| Size | Architecture | r² patience 10 (main) | r² patience 30 | per-seed r² (patience 30) |\n|---|---|---|---|---|\n")
            for size, arch, v in sens:
                main = df[(df["size"] == size) & (df.arch == arch)].r2.values
                f.write(f"| {size} | {arch} | {ms(main)} | {ms(v)} | {', '.join(f'{x:.2f}' for x in v)} |\n")
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for arch in ("cnn", "transformer", "hybrid"):
        d = df[df.arch == arch]
        if d.empty: continue
        s = d.groupby("n_train_total").r2.agg(["mean", "std"]).reset_index()
        ax.errorbar(s.n_train_total * (1 + 0.04 * ["cnn", "transformer", "hybrid"].index(arch)), s["mean"], yerr=s["std"].fillna(0),
                    color=COL[arch], marker="o", ms=6, lw=2, capsize=3, label={"cnn": "CNN (Optimus 5-Prime)", "transformer": "Transformer",
                                                                                 "hybrid": "CNN + attention hybrid"}[arch])
    ax.set_xscale("log"); ax.set_xticks([1000, 5000, 260000], ["1k", "5k", "260k"])
    ax.set(xlabel="Training sequences (log scale)", ylabel="Test r² (20k high-read test set)", ylim=(0, 1))
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title("Learning curves (mean ± sd over seeds)", loc="left"); ax.legend(frameon=False, loc="lower right")
    fig.savefig(f"{F}/B_learning_curves.png"); plt.close(fig)
    print(open(f"{R}/partB/summary.md").read())


if __name__ == "__main__":
    what = sys.argv[1]
    for k, fn in (("B", part_b),):
        if what in (k, "all"):
            fn()
