# Metabolic syndrome, early incident stroke and depressive symptoms in CHARLS

Analysis code and aggregate results for the manuscript:

> **Early incident stroke and depressive symptoms in CHARLS: a small indirect effect robust to mediator misclassification, and a 5.6-fold difference between mediator windows**
> Hao Chen, Jie Li — Department of Pharmacy, Luohe Central Hospital, Henan, China

Submitted to *Journal of Affective Disorders*.

---

## What is in this repository

| Path | Contents |
|---|---|
| `scripts/` | All analysis scripts. `run_all.py` executes the full pipeline in order. |
| `_output/` | 22 aggregate result files (18 JSON + 4 CSV) backing every reported estimate and all figures. |
| `requirements.txt` | Python dependencies and the versions used for the reported results. |

## What is **not** in this repository

**No CHARLS data are redistributed here** — neither the raw waves nor the derived participant-level analysis file. CHARLS is distributed under a data-use agreement that prohibits redistribution. This repository therefore contains only code and aggregate outputs.

To reproduce the analysis you must obtain CHARLS yourself:

1. Register at <http://charls.pku.edu.cn> and accept the data-use agreement.
2. Download the Harmonized CHARLS (Gateway to Global Aging Data, version D), the 2011 blood-assay module, and the 2015/2018 wave files.
3. Place them under `data/` in the layout expected by `scripts/clean_harmonized.py`.

The derived participant-level file (`analysis_harmonized.csv`, n = 11,314) is produced from those inputs by `scripts/clean_harmonized.py` and is **not** committed.

## Environment

Python 3.13 with `pandas`, `numpy`, `statsmodels`, `scipy`, `matplotlib` (see `requirements.txt`; reported results were produced with Python 3.13.14 / pandas 3.0.5 / numpy 2.5.3 / statsmodels 0.15.0).

```
pip install -r requirements.txt
```

## Running the pipeline

```
python scripts/run_all.py
```

Each step runs in order and the pipeline exits non-zero if any step fails. The step list is defined in `STEPS` at the top of `run_all.py`:

1. `clean_harmonized.py` — build the harmonized analysis file
2. `run_flow_missing.py` — participant flow, attrition, missingness predictors
3. `run_mets_v2.py` — primary mediation analysis
4. `run_any_stroke.py` — the mediator-window comparison
5. `run_path_model.py` — path-specific models
6. `run_seed_v2.py` — nine-seed bootstrap stability
7. `run_sensitivity_pack.py` — definition, E-value and reverse-causation analyses
8. `run_review_extras.py` — interaction, imputation and stratified analyses
9. `make_table1_v2.py` — Table 1
10. `make_figs_mets_v3.py` — Figures 1–3

`make_fig_s1.py` produces the supplementary flow figure and is run separately.

The figure step asserts that its input JSON files were regenerated in the current run, so a stale output cannot silently be plotted.

## Two estimation chains

The counterfactual mediation estimate is computed two ways and **the two are two different estimands, not a numerical discrepancy**:

- a `statsmodels` binomial-GLM chain (point estimates quoted in the main text; NIE = 0.0012);
- a hand-rolled iteratively reweighted least squares chain (used for the sensitivity blocks; NIE = 0.0011).

The regression coefficients agree to 3.3 × 10⁻¹⁵; the indirect effects differ by about 2.5% because the statsmodels chain averages the participant-level product of the two counterfactual contrasts, whereas the IRLS chain averages the two factors separately before multiplying. Each reported value is labelled with its chain in the manuscript.

## Key numbers

- Analytical sample: n = 11,314 (from 25,586 baseline respondents)
- Metabolic syndrome (self-reported definition): 11.4% (n = 1,293)
- Early incident stroke (2013/2015): 172 mediator events
- NIE = 0.0012 (95% CI 0.0002–0.0028); TE = 0.0008 (95% CI −0.0241 to 0.0271)
- Mediator-window ratio: 5.60 (0.0065 vs 0.0012)

## Citation

If you use this code, please cite the manuscript above once it is published.

## Contact

Hao Chen — chenhao_zzu@163.com — ORCID 0009-0003-6021-6559
