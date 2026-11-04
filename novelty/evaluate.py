import numpy as np
from scipy import stats


def metrics(y, p):
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    r = stats.pearsonr(y, p)[0]
    return {"r2": float(r ** 2), "pearson": float(r), "spearman": float(stats.spearmanr(y, p)[0]),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "n": int(len(y))}
