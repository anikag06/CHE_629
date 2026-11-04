import os
import numpy as np
import pandas as pd

UP = "/mnt/user-data/uploads"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMS = ["U", "psi", "m1psi"]
REP1 = {"U": f"{UP}/GSM3130435_egfp_unmod_1_csv.gz", "psi": f"{UP}/GSM3130437_egfp_pseudo_1_csv.gz",
        "m1psi": f"{UP}/GSM3130439_egfp_m1pseudo_1_csv.gz"}
REP2 = {"U": f"{UP}/GSM3130436_egfp_unmod_2_csv.gz", "psi": f"{UP}/GSM3130438_egfp_pseudo_2_csv.gz",
        "m1psi": f"{UP}/GSM3130440_egfp_m1pseudo_2_csv.gz"}
CHEM_TABLE = f"{ROOT}/data/chem_table.csv"


def _chem_labels(path):
    d = pd.read_csv(path, usecols=lambda c: c in ("utr", "rl", "total", "total_reads"))
    dep = "total_reads" if "total_reads" in d else "total"
    d["utr"] = d.utr.str[:50]
    d = d.dropna(subset=["rl"]).sort_values(dep, ascending=False).drop_duplicates("utr")
    return d.set_index("utr").rl


def build_chem_table():
    """U split from the reproduction (data/train.csv, data/test.csv) with rep1 MRL for every chemistry
    and rep2 MRL (test rows only, for the replicate ceiling). Missing labels are NaN."""
    if os.path.exists(CHEM_TABLE):
        return pd.read_csv(CHEM_TABLE)
    tr = pd.read_csv(f"{ROOT}/data/train.csv").assign(split="train")
    te = pd.read_csv(f"{ROOT}/data/test.csv").assign(split="test")
    t = pd.concat([tr, te], ignore_index=True)[["utr", "split", "rl", "mfe"]].rename(columns={"rl": "rl_U"})
    for c in ["psi", "m1psi"]:
        t[f"rl_{c}"] = t.utr.map(_chem_labels(REP1[c]))
    for c in CHEMS:
        t[f"rep2_{c}"] = t.utr.map(_chem_labels(REP2[c]))
    t.to_csv(CHEM_TABLE, index=False)
    return t


def u_split():
    tr, te = pd.read_csv(f"{ROOT}/data/train.csv"), pd.read_csv(f"{ROOT}/data/test.csv")
    return tr, te


def nested_subsets(n_train, sizes, seed, out_dir):
    """Nested random subsets of training indices (smaller contained in larger); saved to disk."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    subs = {s: np.sort(perm[:s]) for s in sizes}
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/subsets_seed{seed}.npz", **{str(k): v for k, v in subs.items()})
    return subs


def train_val(idx, seed, val_frac):
    """Split a training subset into fit / validation indices."""
    rng = np.random.default_rng(10_000 + seed)
    p = rng.permutation(idx)
    n_val = max(1, int(round(len(idx) * val_frac)))
    return np.sort(p[n_val:]), np.sort(p[:n_val])


def random_split(n, n_test, seed):
    rng = np.random.default_rng(20_000 + seed)
    p = rng.permutation(n)
    return np.sort(p[n_test:]), np.sort(p[:n_test])
