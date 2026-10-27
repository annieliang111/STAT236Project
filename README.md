# STAT 236 - Network-aware testing on temporal and static networks

## Primary 50-day extension (NEW)

This version implements **a dynamic 50-day Haslemere contact process**, *not* an observed 50-day contact dataset. The source has only three measured daytime sequences (192 five-minute slots each). The simulator keeps those exact sequences on Days 1, 2, and 3, then draws an entire source day at random with replacement for each Day 4 through Day 50. The resulting 9,600 five-minute daytime transmission steps preserve observed within-day ordering. A 96-step (8-hour) overnight gap is included for recovery/diagnosis but has **no modeled transmission**. Random sampled daily block paths are paired across testing policies, held out from decisions until contact observations arrive.

The underlying disease model remains **SHCR** (S susceptible, H hidden infectious, C confirmed/isolated, R recovered). **No exposed E state was added**. Its immediate S-to-H step and fixed five-minute hazard are modeling simplifications. All future contacts, infection states, and test outcomes are simulated; none are claims about observed infections. A permanent missing-dyad mask applies only to the network available for filtering/testing decisions.

The 50-day extension uses: 20 tests/day split into four five-test daytime rounds, 25 initially hidden plus four confirmed individuals, daily baseline diagnosis 0.14 and recovery 0.12, five-minute contact hazard 0.060, test sensitivity 0.90, specificity 1. The 12-hour recency e-folding time is different from the prior three-day sensitivity setting. These are **not empirically estimated disease parameters**. A fresh logistic risk model is trained on 24 separate 50-day synthetic epidemics and held out on another 10 trajectories; 12 matched testing-policy replicates are reported at each of full versus 50% contact visibility. Results and code are in `results/extended_*`, `models/extended_temporal_risk_model.joblib`, and `figures/extended_*`.

```bash
python3 -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 test_extended_haslemere.py
# To reproduce exactly the released extended results from the bundled fitted model:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 extended_temporal_haslemere.py --skip-fit --days 50 --reps 12
# Or fit the extended model again before evaluation:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 extended_temporal_haslemere.py --days 50 --train 24 --heldout 10 --reps 12
python3 create_extended_assets.py
python3 create_extended_webpage.py
pdflatex report.tex && biber report && pdflatex report.tex && pdflatex report.tex
```

**Interpretation:** Day-block resampling permits long-run simulated epidemic outcomes but cannot add new observed contacts, novel ties, seasonal changes, or unobserved overnight edges. The three-day empirical record is a small, nonrandom convenience sample. The original measured-only three-day results remain in the appendix and `temporal_haslemere.py`; static SBM experiments remain unchanged. The compiled manuscript uses `references.bib` (BibLaTeX/Biber).

---

## Retained three-day sensitivity (original temporal analysis)

**Earlier measured-only empirical-network experiment: the measured three-day Haslemere sequence at five-minute resolution.** The true contact network changes for each of 576 observed slots, with 17,363 qualifying observations within four meters. Infections are **simulated**, and testing allocation acts on only past observed contacts. Five tests are selected at each of four daytime rounds (20 tests/day), using approximate SHCR Bayesian filtering and a separately fitted temporal infection-risk estimator. The four standard heuristic/random baselines and two optimizers remain. The temporal experiment compares a recency-weighted optimization score to a cumulative **past-only** contact-frequency score. No policy uses contact observations from future time slots.

**Complementary controlled experiment:** 50-day **static SBM** networks still examine community structure and missing edges (30 paired replications per condition). Previous static 50-day Haslemere-graph results remain in the source package as a historical static-network sensitivity experiment, and are **not** interpreted as a temporal model.

### Fresh files for the temporal experiment

- `temporal_haslemere.py`: reads headerless `Kissler_DataS1.csv`, constructs all 576 time-stamped contact matrices, carries a time-dependent SHCR simulator, mean-field filter, fitted logistic risk policy, and causal history-based allocation.
- `test_temporal_haslemere.py`: fast tests for contact totals, Bayes updates, posterior normalization, random-policy invariance to missing observed contacts, budget compliance, and counting initial confirmed cases.
- `create_temporal_webpage.py`: regenerates the demonstration page from executed CSV results.
- `data/Kissler_DataS1.csv`: the public CSV supplied for the project, included so the package is self-contained (source: https://github.com/skissler/haslemere).
- `models/temporal_risk_model.joblib` and `results/temporal_risk_*.csv`: temporal-specific fitted logistic model trained on 60 separate simulated epidemics, evaluated on 16 held-out episodes. This is independent of the earlier static fitted model.
- `results/temporal_batch_*.csv`, `results/temporal_haslemere_*.csv`: 15 paired, independent evaluation replicates per observation level; **reported 95% Monte Carlo intervals are exploratory** given this replication count. The full benchmark is not claiming 30 temporal replications.
- `figures/temporal_haslemere_*.png`: plots for the measured time-varying contact network.
- `report.tex`, `report.pdf`, `analysis.Rmd`, and `index.html`: updated paper, source, analysis notebook, and demonstration site.

### Reproduce the earlier three-day temporal experiment

```bash
python3 -m pip install -r requirements.txt
python3 test_temporal_haslemere.py
# Refit and run; larger runs can be slow depending on BLAS settings and hardware.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 temporal_haslemere.py --train 60 --heldout 16 --reps 15
# Or use the shipped fitted model to run 15 deterministic paired reps:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 temporal_haslemere.py --skip-fit --reps 15
python3 create_temporal_webpage.py
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

For smaller per-process jobs, use `--skip-fit --batch-start 0 --batch-count 3` and similarly for starts 3,6,9,12, followed by `--aggregate-batches`. The aggregation command reads every `results/temporal_batch_*.csv` and checks duplicate cells; **delete old batch files before changing model parameters**. To extend to 30 replications, add starts 15,18,21,24,27, then reaggregate. Each paired replicate uses the same initial infection set, uniform transmission innovations, and nested missing-dyad masks across policies and observation settings.

**Important assumptions:** Transmission hazard 0.060 per five-minute qualifying encounter, 25 initially hidden and four confirmed infections, daily recovery 0.12 and routine diagnosis 0.14 converted to five-minute probabilities, sensitivity 0.90, perfect specificity, instantaneous isolation after positive testing, no modeled transmission during the eight unrecorded nighttime hours, and fixed measured movement despite simulated isolation. These are not clinical estimates. No observed infection outcomes are in Haslemere. The three-day experiment targets infections during observation rather than long-run epidemic final size.

---

## Existing static-network studies (retained for reproducibility)


The earlier static-network experiment has a **factorized Bayesian SHCR filter**, **logistic infection-risk estimation from separate simulated epidemics**, and **budget-constrained one-step optimized testing**. It retains the original adaptive network heuristic and random, contact, and degree testing as four baseline methods. An additional filter-only optimizer is an ablation of risk estimation.

**Main deliverables**

- `report.tex` and `report.pdf`: paper with a **six-page main body** plus an expanded technical appendix, results, figures, references and reproducibility instructions.
- `analysis.Rmd`: R Markdown companion that reads saved replicate files, checks fitted-risk diagnostics, and regenerates comparison tables/figures. (R wasn't installed during creation, so this notebook was reviewed but not knitted here.)
- `index.html`: standalone project demonstration; open it alongside the `figures/` directory, or publish the whole folder.
- `model_based_core.py`: mean-field filtering recursion, Bayesian evidence updates, fitted-risk features, optimizer, and matched simulator.
- `fit_risk_model.py`: simulated training epidemics, trajectory-held-out validation, seven predictor coefficients, serialized fitted model.
- `run_optimized_experiments.py`: six-policy SBM and Haslemere comparisons with matched seeds and checkpointing.
- `create_model_based_assets.py` and `create_model_based_webpage.py`: results-to-report/page generation.
- `test_model_based.py`: assertions for posterior normalization, belief updates, budget, and equivalence of earlier policy outcomes.
- `results/optimized_*.csv`: executed **30 independent paired epidemics per network condition**; six SBM community-strength ratios and five observed-edge probabilities, plus Haslemere contact topology with five observation levels.
- `results/risk_model_*.csv`, `models/risk_model.joblib`: fitted model, 90 synthetic training trajectories, 24 held-out synthetic trajectories.
- The previous baseline simulation programs and outputs remain to permit complete replication of paired comparisons.

## Run from a clean environment

Python 3.10+ and the packages in `requirements.txt` are required. The original public **headerless** Haslemere proximity source file is now packaged: copy your `Kissler_DataS1.csv` from https://github.com/skissler/haslemere into `data/Kissler_DataS1.csv`. This is the **only external data input**; the ZIP contains the raw public CSV, saved simulated results, and fits.

```bash
python3 -m pip install -r requirements.txt
python3 test_model_based.py
# Refit individual-level risk estimator on simulated labeled trajectories
python3 fit_risk_model.py --csv data/Kissler_DataS1.csv --train-reps 90 --valid-reps 24
# Regenerate four original policy baselines with the matched evaluation seed
python3 run_structure_sensitivity.py --reps 30
python3 run_haslemere_visibility.py --csv data/Kissler_DataS1.csv --reps 30
# Add 2 optimization policies; reuse paired baseline run CSVs for speed
python3 run_optimized_experiments.py --csv data/Kissler_DataS1.csv --reps 30
python3 create_model_based_assets.py
python3 create_model_based_webpage.py
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

To re-execute the optimizers from scratch after a changed model, delete `results/optimized_sbm_checkpoint.csv` and `results/optimized_haslemere_checkpoint.csv` first. These checkpoint files are not automatically tied to the contents of the fitted model or Python code, so clear them if you change the model or any policy implementation.

The two original baseline files are generated by `run_structure_sensitivity.py` and `run_haslemere_visibility.py` with **the same** evaluation seed `2362026`, graph-generation uniforms, edge masks and individual epidemic innovations. The optimizer runner crossjoins those baseline rows with its own corresponding policies by replicate and network condition. `test_model_based.py` verifies baseline simulation equivalence. For a clean run, ensure these legacy baseline files are from the matching seed and settings; the optimized runner rejects mismatched seed values for its own run, but cannot independently authenticate externally replaced CSV files.

## Model and interpretation

- Simulator parameters: transmission hazard `tau=0.055`, spontaneous diagnosis `delta=0.025`, recovery probability `gamma=0.12`, test sensitivity `0.90`, perfect specificity, zero reporting delay, five initial infections, 50 days. These describe **fixed experimental assumptions**, not parameters estimated from observed infections.
- The filter updates individual `(S,H,C,R)` marginals after observed absence/presence of routine diagnoses and positive/negative test results, then predicts infection using observed neighbors' posterior hidden probabilities. It is a **mean-field approximation**, not an exact posterior over the joint disease process.
- Risk estimation fits standardized L2-regularized logistic regression (`C=1`) on only **synthetic person-day labels**, with grouping by epidemic trajectory (90 train, 24 held-out). The filter's hidden-state log-odds are one predictor. The training data include separate epidemics on the Haslemere topology; thus Haslemere evaluation is not generalization to an entirely unseen network, but *is* tested on separate outbreak realizations.
- On day `t`, the inferred value of testing `i` is `P(H_i) * 0.9 * (1-exp(-0.055)) * sum_j Aobs[i,j]*P(S_j)`. We choose the top 20 eligible individuals. This **exactly solves the additive one-day surrogate**, not the multi-day optimal control problem and not Huang et al.'s county-level Frank-Wolfe relaxation.
- Network availability is manipulated separately from true transmission: `A` is the simulator's true adjacency, `Aobs` is its edge-deleted observed version. When missingness is 75%, only 25% of edges are visible to the policy.
- The Haslemere **contact observations are real**, but all COVID-/disease-infection histories, tests and testing responses are simulated. Results therefore do not establish clinical benefits.

## Reported findings (30 paired simulated epidemics per cell)

- On fully observed **SBM** networks at community ratio 1, final infected fraction: adaptive heuristic **35.0%**, fitted optimizer **32.6%**; at ratio 12, **30.4%** and **30.0%**. Both fitted-minus-heuristic paired 95% Monte Carlo intervals include zero.
- On the **Haslemere** contact network at 100% edge visibility, heuristic **16.4%** versus fitted optimizer **16.1%**; at 25% visibility, heuristic **25.5%** versus fitted optimizer **25.0%**. There is no strong evidence from this experiment that the optimizer dominates the heuristic.
- Held-out synthetic-label risk model AUC **0.736** and Brier score **0.0455**, versus filter-only AUC **0.691** and Brier score **0.0476**.

**Project scope and standards:** This is a paired simulation-method comparison, not an empirical infection parameter estimation study or a proof of optimal epidemic control. Report main-body length stays below the 8-page guideline; additional technical details are placed in the appendix. 
