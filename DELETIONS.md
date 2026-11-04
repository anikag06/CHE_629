# What was removed (vs the original repository)

Only deletions. No code was rewritten; no new logic was added. This repo keeps the
reproduction + the single extension (CNN + attention hybrid).

## Files deleted (17)
- novelty/partA_recal.py        multi-chemistry recalibration
- novelty/partC.py              Siamese variant-effect model
- novelty/partC_common.py       helpers for Part C
- novelty/partC_sat.py          saturation mutagenesis
- novelty/figs_cd.py            figures for Parts C and D
- 05_retrain.py                 retrained model (not in report)
- 11_run_model.py               exact duplicate of runner.py
- build_notebook.py             regenerates an already-committed notebook
- weights/multichem_heads_s0.pt
- figures/novelty/{A_cross_chemistry,B_chemistry_regimes,C_variant_effects,D_motif_and_profile}.png
- paper/figs/{A_cross_chemistry,B_chemistry_regimes,C_variant_effects,D_motif_and_profile}.png

## Parts deleted inside kept files
- novelty/partA.py          removed the multi-chemistry model selection + training (kept single-chemistry CNN baselines)
- novelty/partB.py          removed stage_chem (cross-chemistry transfer); kept small/full/randsplit
- novelty/partD.py          removed stage_chem + run_modisco (motif discovery); kept stage_arch
- novelty/models.py         removed MultiChem, Siamese classes
- novelty/data.py           removed build_pairs() variant-pair builder
- novelty/evaluate.py       removed delta_metrics()
- novelty/report.py         removed part_a, part_c, part_d, chem_regimes; kept part_b
- novelty/unified_metrics.py removed the two multi-chemistry table rows
- demo.py                   removed the variant() section
- scripts/export_demo_weights.py  removed the multichem_heads weight line
- novelty/queue.sh, queue2.sh     removed lines invoking the deleted experiments

## Docs updated to match (prose only)
- README.md, NOVELTY_RESULTS.md     now describe one extension, not four
- paper/main.tex, paper/main.pdf    the final report (Author Contributions already removed)

## Known cosmetic leftovers (harmless, not touched to avoid altering code)
- novelty/partA.py line 1: docstring still mentions the multi-chemistry model
- a few now-unused imports (e.g. json in partA.py) — flagged by linters, do not affect behaviour
- README deps still list modisco-lite (only the removed motif code used it)
- SUBMISSION_CHECKLIST.md still references four extensions
