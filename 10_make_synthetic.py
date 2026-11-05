
import numpy as np
import pandas as pd
import time
import os
from multiprocessing import freeze_support

from synth import make_dataset, NOISE_SD, mfe_batch


def main():
    t = time.time()

    # Generate 280,000 synthetic sequences
    seqs, mfe, yt, yo = make_dataset(280_000, seed=0)

    df = pd.DataFrame({
        "utr": seqs,
        "mfe": mfe,
        "rl_true": yt,
        "rl": yo
    })

    # Train-test split
    te, tr = df.iloc[:20_000], df.iloc[20_000:]

    # Create synthetic data directory if it doesn't exist
    os.makedirs("data/synth", exist_ok=True)

    # Save synthetic train and test datasets
    tr.to_csv("data/synth/train.csv", index=False)
    te.to_csv("data/synth/test.csv", index=False)

    # Calculate theoretical R2 ceiling
    ceil = np.var(te.rl_true) / (
        np.var(te.rl_true) + NOISE_SD**2
    )

    print(
        f"synthetic done in {time.time()-t:.0f}s; "
        f"MRL obs mean {df.rl.mean():.2f} "
        f"sd {df.rl.std():.2f}; "
        f"r2(true,obs) on test = "
        f"{np.corrcoef(te.rl_true, te.rl)[0,1]**2:.3f}; "
        f"theoretical ceiling {ceil:.3f}"
    )

    # Calculate MFE for real eGFP data
    # Required for the LightGBM benchmark
    for split in ("train", "test"):
        d = pd.read_csv(f"data/{split}.csv")
        d["mfe"] = mfe_batch(list(d.utr))
        d.to_csv(f"data/{split}.csv", index=False)

    print(f"all done {time.time()-t:.0f}s")


if __name__ == "__main__":
    freeze_support()
    main()
