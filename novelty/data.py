
import os
from pathlib import Path

import numpy as np
import pandas as pd


# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Directory containing the original GEO CSV files.
# Can be configured separately on Windows and Linux.
UP = Path(
    os.environ.get(
        "GSE114002_RAW_DIR",
        ROOT / "GSE114002_RAW"
    )
)

CHEMS = ["U", "psi", "m1psi"]

# Biological replicate 1
REP1 = {
    "U": UP / "GSM3130435_egfp_unmod_1.csv.gz",
    "psi": UP / "GSM3130437_egfp_pseudo_1.csv.gz",
    "m1psi": UP / "GSM3130439_egfp_m1pseudo_1.csv.gz"
}

# Biological replicate 2
REP2 = {
    "U": UP / "GSM3130436_egfp_unmod_2.csv.gz",
    "psi": UP / "GSM3130438_egfp_pseudo_2.csv.gz",
    "m1psi": UP / "GSM3130440_egfp_m1pseudo_2.csv.gz"
}

CHEM_TABLE = ROOT / "data" / "chem_table.csv"


def _chem_labels(path):
    d = pd.read_csv(
        path,
        usecols=lambda c: c in ("utr", "rl", "total", "total_reads")
    )

    dep = "total_reads" if "total_reads" in d else "total"

    d["utr"] = d.utr.str[:50]

    d = (
        d.dropna(subset=["rl"])
         .sort_values(dep, ascending=False)
         .drop_duplicates("utr")
    )

    return d.set_index("utr").rl


def build_chem_table():
    """
    U split from the reproduction (data/train.csv, data/test.csv)
    with rep1 MRL for every chemistry and rep2 MRL
    (test rows only, for the replicate ceiling).

    Missing labels are NaN.
    """

    if CHEM_TABLE.exists():
        return pd.read_csv(CHEM_TABLE)

    # Check that all required files exist
    required_files = list(REP1.values()) + list(REP2.values())

    missing = [str(p) for p in required_files if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "The following required GEO files are missing:\n"
            + "\n".join(missing)
        )

    tr = pd.read_csv(ROOT / "data" / "train.csv").assign(split="train")
    te = pd.read_csv(ROOT / "data" / "test.csv").assign(split="test")

    t = (
        pd.concat([tr, te], ignore_index=True)
        [["utr", "split", "rl", "mfe"]]
        .rename(columns={"rl": "rl_U"})
    )

    for c in ["psi", "m1psi"]:
        t[f"rl_{c}"] = t.utr.map(_chem_labels(REP1[c]))

    for c in CHEMS:
        t[f"rep2_{c}"] = t.utr.map(_chem_labels(REP2[c]))

    CHEM_TABLE.parent.mkdir(parents=True, exist_ok=True)

    t.to_csv(CHEM_TABLE, index=False)

    return t


def u_split():
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    te = pd.read_csv(ROOT / "data" / "test.csv")

    return tr, te


def nested_subsets(n_train, sizes, seed, out_dir):
    """Nested random subsets of training indices
    (smaller contained in larger); saved to disk.
    """

    rng = np.random.default_rng(seed)

    perm = rng.permutation(n_train)

    subs = {
        s: np.sort(perm[:s])
        for s in sizes
    }

    os.makedirs(out_dir, exist_ok=True)

    np.savez(
        Path(out_dir) / f"subsets_seed{seed}.npz",
        **{str(k): v for k, v in subs.items()}
    )

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
