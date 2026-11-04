"""Part D: attribution and motif discovery.

usage: python3 novelty/partD.py arch
  arch : CNN vs transformer (vs hybrid if trained), Part B full-data seed-0 models: attribution agreement.
"""
import common
import json, os, sys
import numpy as np
import pandas as pd
import torch
from scipy import stats
from attribution import integrated_gradients, ism, ism_contrib, top_kmers, annotate, seq_of
from data import build_chem_table, CHEMS
from models import CNN, build
from optimus import one_hot

OUT = "results/novelty/partD"
os.makedirs(OUT, exist_ok=True)
t = build_chem_table()
Y = t[[f"rl_{c}" for c in CHEMS]].values
test = np.where((t.split == "test") & ~np.isnan(Y).any(1))[0]
rng = np.random.default_rng(0)
S = np.sort(rng.choice(test, 2000, replace=False))
np.save(f"{OUT}/attribution_idx.npy", S)
X = one_hot(t.utr.values[S])
SEQS = t.utr.values[S]
NO_AUG = np.array(["ATG" not in s for s in SEQS])


def load_single(c, s):
    m = CNN(dropout=0.2); m.load_state_dict(torch.load(f"results/novelty/partA/single_{c}_s{s}.pt")); return m.eval()


def attributions(model, tag):
    p = f"{OUT}/attr_{tag}.npz"
    if os.path.exists(p):
        z = np.load(p); return z["ig"], z["hyp"], z["ism"]
    ig, hyp = integrated_gradients(model, X)
    d = np.concatenate([ism(model, X[i:i + 250]) for i in range(0, len(X), 250)])
    np.savez_compressed(p, ig=ig, hyp=hyp, ism=d)
    return ig, hyp, d


def gc_sensitivity(d):
    """Mean predicted change (model units, standardised MRL) for A/T->G/C substitutions in uAUG-free UTRs,
    excluding substitutions that create an ATG."""
    vals = []
    for n in np.where(NO_AUG)[0]:
        s = SEQS[n]
        for p in range(50):
            if s[p] in "AT":
                for b in (1, 2):
                    m = s[:p] + "ACGT"[b] + s[p + 1:]
                    if "ATG" not in m:
                        vals.append(d[n, p, b])
    return float(np.mean(vals))


def uaug_creation(d):
    """Mean predicted change when a single substitution creates an ATG (in uAUG-free UTRs)."""
    vals = []
    for n in np.where(NO_AUG)[0]:
        s = SEQS[n]
        for p in range(50):
            for b in range(4):
                if "ACGT"[b] != s[p] and "ATG" in s[:p] + "ACGT"[b] + s[p + 1:]:
                    vals.append(d[n, p, b])
    return float(np.mean(vals))


def stage_arch():
    B = "results/novelty/partB/full"
    models = {}
    for arch in ("cnn", "transformer", "hybrid"):
        cf = f"{B}/{arch}_choice.json"
        if not os.path.exists(cf):
            continue
        best = json.load(open(cf))["chosen"]
        pt = f"{B}/{arch}_full_{best}_s0.pt"
        if os.path.exists(pt):
            do = float(best.split("_do")[1])
            m = build(arch, do); m.load_state_dict(torch.load(pt)); models[arch] = m.eval()
    res, pp, igp = {}, {}, {}
    for a, m in models.items():
        ig, hyp, d = attributions(m, f"arch_{a}")
        pp[a], igp[a] = ism_contrib(d, X).sum(-1), ig.sum(-1)
        res[f"{a}_uaug_create"], res[f"{a}_gc_sens"] = uaug_creation(d), gc_sensitivity(d)
    names = list(models)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            res[f"{a}_vs_{b}"] = {
                "ism_per_seq_r": float(np.nanmean([stats.pearsonr(pp[a][n], pp[b][n])[0] for n in range(len(S))])),
                "ig_per_seq_r": float(np.nanmean([stats.pearsonr(igp[a][n], igp[b][n])[0] for n in range(len(S))])),
                "top5_position_overlap": float(np.mean([len(set(np.argsort(-np.abs(pp[a][n]))[:5]) & set(np.argsort(-np.abs(pp[b][n]))[:5])) / 5
                                                        for n in range(len(S))]))}
    if "cnn" in models:
        res["within_cnn_seed_agreement_note"] = "see chem stage: CNN seeds share architecture; compare to cross-arch r"
        m1 = load_single("U", 1)
        _, _, d1 = attributions(m1, "U_s1")
        p1 = ism_contrib(d1, X).sum(-1)
        res["cnn_seed0_vs_seed1_ism_per_seq_r"] = float(np.nanmean([stats.pearsonr(pp["cnn"][n], p1[n])[0] for n in range(len(S))]))
    kms = []
    for a in models:
        for sign, lab in ((-1, "repressive"), (1, "activating")):
            _, _, d = attributions(models[a], f"arch_{a}")
            df = top_kmers(ism_contrib(d, X), X, k=6, top_frac=0.05, sign=sign).head(15)
            df["arch"], df["sign"] = a, lab
            df["known"] = [";".join(annotate(k)) for k in df.kmer]
            kms.append(df)
    pd.concat(kms).to_csv(f"{OUT}/kmer_enrichment_by_arch.csv", index=False)
    json.dump(res, open(f"{OUT}/arch_attribution.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"arch": stage_arch}[sys.argv[1]]()
    print(f"PART D {sys.argv[1]} DONE", flush=True)
