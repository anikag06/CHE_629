# Optimus 5-Prime: novelty extensions, results

Builds on the reproduction of Sample et al., *Nat Biotechnol* 2019 (GEO GSE114002) in this folder. All numbers below were recomputed from the saved predictions in `results/novelty/`. Seeds and subset indices are saved to disk.

**Hardware:** 2 CPU cores, no GPU. One full-data epoch takes about 2.3 min for the CNN, 6.4 min for the transformer and 5.6 min for the hybrid.

## 0. Data availability (checked before modelling)

| Question | Finding | Consequence |
|---|---|---|
| U / Ψ / m1Ψ random-library data in GSE114002? | All three, 2 replicates each (GSM3130435–40) | Part A feasible |
| Same sequences across chemistries? | Yes, once the Ψ/m1Ψ rows are trimmed from 59 nt (UTR + 9 nt of CDS) to the first 50 nt. 19,968 of the 20k U test UTRs have labels in all three chemistries | Shared test set across chemistries |
| Chemistry of the human / ClinVar library | **U only.** There is one GEO sample, and its 5,971 control UTRs agree with U (r² 0.86 / 0.80) far better than with Ψ (0.49 / 0.59) or m1Ψ (0.45 / 0.64) | Part C is U-only |
| ClinVar pairs | 867 wild types, 2,135 variants. The authors' filter (variant ≥ 620 reads, non-zero Δ) gives **1,584 pairs**, of which **45** have \|log2 Δ\| > 0.5 | Evaluation set |
| Synthesis-error SNVs | **17,038** single-nt variants of designed sequences. **5,519 pairs** have both members ≥ 620 reads | Extra *training* pairs only (noisier labels) |
| Paper's "76%" for high-effect variants | Recomputed from the authors' released predictions, it's **r² = 0.762** on the high-effect subset. Their direction accuracy there is 93.3% | Both reported |

**Noise ceilings.** The best r² a model can reach against replicate 1 equals the Pearson r between replicates: U 0.971, Ψ 0.916, m1Ψ 0.865.

## Shared protocol

- **Architectures:**
  - CNN: the paper's model, 474,681 parameters.
  - Transformer: pre-LN, d = 96, 4 layers, 4 heads, learned positions, mean pooling; 456,753 parameters.
  - Hybrid: 2 convolution layers (96 filters, kernel 8) feeding a 3-layer transformer; 421,425 parameters.
- **Optimizer:** Adam, batch 128, MSE on standardized MRL. Dropout is applied only in the dense head, for all architectures.
- **Validation split:** carved from each training subset, 20% at 1k/5k and 10% at full.
- **Early stopping:** on validation loss, restoring the best epoch. Patience 10 (max 200 epochs) at 1k/5k; patience 2 (max 15) at full.
- **Tuning:** the same grid for every architecture. At 1k/5k: lr {1e-3, 3e-4} × dropout {0.1, 0.3}. At full: lr {1e-3, 3e-4} with dropout 0.2. Tuned on seed 0's validation split; the chosen config is used for the other seeds.
- **Seeds:** 5 at 1k/5k, 3 at full. Subsets are random and nested (the 1k set is inside the 5k set) and drawn from the training split only. The 20k test set is never used for any choice.

## Part B: architectures × data regimes (U, fixed 20k high-read test set)

| Size | CNN | Transformer | Hybrid |
|---|---|---|---|
| 1k (5 seeds) | **0.317 ± 0.123** | 0.031 ± 0.007 | 0.210 ± 0.251 |
| 5k (5 seeds) | **0.778 ± 0.009** | 0.050 ± 0.011 | **0.778 ± 0.005** |
| 260k (3 seeds) | 0.922 ± 0.002 | 0.728 ± 0.012 † | **0.942 ± 0.001** |

† Every full-data transformer run hit the 15-epoch cap while validation loss was still falling, so this value is budget-limited.

Pearson, Spearman, RMSE, chosen configs and best epochs are in `results/novelty/partB/summary.md`. At full data the hybrid also has the lowest RMSE (0.454 vs 0.522 for the CNN).

- **The hybrid is the best model at full data:** +0.020 r² over the paper's CNN on all 3 seeds, with fewer parameters.
- **The CNN is the most data-efficient.** It wins at 1k and ties the hybrid at 5k.
- **Negative result: the plain transformer fails at small data.** It can't fit even its training set at 1k/5k. Bug checks, done on training data only: changing the learning rate or switching to a sinusoidal positional encoding didn't help. The same encoder behind a convolution stem (the hybrid) learns well. Conclusion: learning motifs like ATG from single-nucleotide tokens through attention needs many more updates than small datasets provide. At full data it sits on a plateau for about 8 epochs, then breaks out.
- **The hybrid is unstable at 1k.** Its outcome is bimodal: either about 0.65 or about 0.04, depending on whether it escapes the initial plateau.

**Early-stopping sensitivity** (same configs and subsets, patience 30, all architectures):

| | CNN | Transformer | Hybrid |
|---|---|---|---|
| 1k | 0.317 → 0.435 ± 0.054 | 0.031 → 0.035 | 0.210 → 0.407 ± 0.300 (still bimodal: 0.04, 0.04, 0.67, 0.63, 0.66) |
| 5k | 0.778 → 0.772 | 0.050 → 0.067 | 0.778 → 0.770 |

Patience 10 under-trains the CNN at 1k. It doesn't fix the transformer or the hybrid's bimodality, and the ranking at 5k is unchanged.

**Random-split sanity check** (full data, random 20k test, seed 0 only): CNN 0.891 vs 0.923 on the high-read split; hybrid 0.896 vs 0.941; transformer 0.208 (that seed never escaped its plateau). Random test sets include noisier low-read UTRs, so a modest drop is expected; the hybrid ≥ CNN ordering holds.

## Part D: attribution and motifs

Single-nucleotide ISM on 2,000 shared test UTRs, comparing the three architectures.

- The hybrid's ISM maps match the CNN's (per-sequence r = 0.96, top-5 position overlap 84%), nearly as closely as two CNN seeds match each other (0.97). It learned the same grammar.
- The transformer agrees less (r 0.79, overlap 64%) and relies less on uAUGs.
- IG agreement is much lower than ISM agreement (CNN vs transformer 0.30), so ISM is the more stable comparison.

## Issues found and fixed during the run

1. **Transformer dropout.** Dropout mask generation took 41% of transformer step time on CPU. Dropout was kept only in the dense head, the same place as in the CNN.
2. **NaN in config selection.** The full-data hybrid at lr 1e-3 collapsed to constant predictions (r² undefined), and `max()` doesn't handle NaN reliably. Selection now treats NaN as the worst score. Every other selection was audited and none was affected.
3. **Noise-ceiling definition.** I first reported replicate r² as the ceiling. The correct ceiling is replicate Pearson r.
4. **Machine restart.** The cloud machine restarted while idle and killed background jobs. Runs resume from saved results, so none were lost.

## Reproduce

From this folder (paths in `novelty/data.py`):

```
python3 novelty/partA.py
python3 novelty/partB.py small
python3 novelty/partB.py full && python3 novelty/partB.py randsplit
python3 novelty/partD.py arch
python3 novelty/partB_sens.py
python3 novelty/report.py B && python3 novelty/unified_metrics.py
```

**Code layout:**
- `data.py`: loaders, chemistry table, nested subsets
- `models.py`: CNN, transformer, hybrid
- `train.py`: training loop with early stopping
- `evaluate.py`: metrics
- `attribution.py`: IG, ISM, k-mers, known-element annotation
- `common.py`: shared run protocol
- `partA.py`: single-chemistry U CNN baselines; `partB*.py`: the extension; `partD.py arch`: attribution agreement
- `report.py`, `unified_metrics.py`: tables and figures

**Outputs:**
- Per-run JSON (metrics, epochs, validation history), predictions and weights in `results/novelty/`
- Subset indices in `results/novelty/partB/subsets/`
- Summaries in `results/novelty/part*/summary.md`
