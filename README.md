# Network-Aware Testing Allocation for Infectious Disease Control Under Limited Resources

**The Role of Community Structure and Incomplete Contact Information**  
STAT 236 course project (Fall 2026)  
**Authors:** Yinpeng Cai, Aniket Jain, Annie Liang, Zidan Luo

[Project repository](https://github.com/annieliang111/STAT236Project)

## Overview

Under a limited testing budget, **whom should we test, and when**, when infections are hidden and contact-network information is incomplete? This project studies how **community structure**, **missing observed contacts**, and **temporal contact history** influence network-aware testing allocation.

We simulate epidemics using a susceptible--hidden infectious--confirmed--recovered (**SHCR**) model and compare simple testing rules with a model-based policy that:

1. Estimates each individual's hidden-infection probability using an approximate Bayesian (mean-field) filter.
2. Refines that estimate using an `L2`-regularized logistic regression trained on *separate simulated epidemics*.
3. Ranks eligible people by an additive approximation to the **immediate infections their testing could prevent**, then selects the highest-scoring individuals within the testing budget.

The optimization **exactly solves its one-step additive surrogate**, not the globally optimal 50-day testing problem. Policies have access to observed contacts, diagnoses, and their own earlier test results, **not** the simulator's true hidden infection states or future contacts.

The two primary experimental settings are:

| Setting | Network | Main question |
| --- | --- | --- |
| **Controlled SBM** | Static stochastic block-model networks | How do community separation and incomplete edge observation affect testing performance? |
| **Extended Haslemere** | Five-minute temporal contacts, simulated over 50 days | Does emphasizing recent contacts improve testing allocation over using cumulative past contact histories? |

The original three-day, measured-contact-only Haslemere simulation remains a **supplementary sensitivity experiment** (Appendix E of the report).

## Data and network construction

### Haslemere contact data

The public [Haslemere / BBC Pandemic dataset](https://github.com/skissler/haslemere) contains **102,831 pairwise distance records** involving **469 volunteers** across three consecutive days (12--14 October 2017). Measurements cover **7:00 a.m. to 11:00 p.m.**, at five-minute intervals: **192 slots/day**, or **576 observed slots** in total. We create an edge when two participants are recorded within **4 m** of each other, yielding **17,363 qualifying proximity records** across **1,262 distinct pairs**.

The data provide *proximity measurements*, not observed infections or a complete population contact network. Contacts with nonparticipants and overnight interactions are missing.

**Fifty-day temporal extension:** Days **1--3** use their original observed sequences. For every simulated day **4--50**, we sample one of the three complete observed daytime sequences uniformly with replacement. We preserve the original order of the 192 five-minute slots within the selected day. This generates **9,600 daytime simulation slots** across 50 days; the eight-hour overnight period allows diagnosis and recovery but has **no modeled transmission**. Sampled future day-blocks are matched across testing policies and are not revealed to a policy in advance.

**Days 4--50 are simulated, not newly observed Haslemere data.** The resampling model cannot create new social ties, capture long-term behavior change, or reconstruct unobserved nighttime contacts.

### Synthetic stochastic block-model networks

Each SBM network contains **180 people in three equally sized communities**. Edges are drawn independently conditional on community membership. The within-community probability `p_in` and between-community probability `p_out` are chosen so the expected degree is **8** while the community-separation ratio `r = p_in / p_out` varies over **1, 1.5, 2, 4, 8, 12**.

For incomplete observation, each existing network edge is retained for the testing policy with probability `q`; the static study uses **q = 0.25, 0.50, 0.60, 0.75, 1.00**. For temporal Haslemere, a missing-dyad mask is held fixed across time and we evaluate **50% and 100%** contact visibility in the main 50-day experiment. In every case, the simulator transmits disease over the *unthinned simulated network*, while the policy sees only the retained edges.

## Epidemic model and testing strategies

All infections, diagnoses, screening results, and disease outcomes are **simulated**. We use four states: susceptible (`S`), hidden infectious (`H`), confirmed/isolated (`C`), and removed/recovered (`R`). There is **no exposed (`E`) state**: newly infected people immediately enter `H`, an explicit biological simplification. Positive screening tests immediately isolate detected hidden-infectious people, preventing their subsequent transmission.

| Assumption | Static SBM | Extended temporal Haslemere |
| --- | --- | --- |
| Horizon | 50 days | 50 days; five-minute daytime transmission |
| Initial conditions | 5 hidden infections | 25 hidden + 4 confirmed cases |
| Transmission hazard | `tau = 0.055` per infectious neighbor per day | `tau5 = 0.060` per five-minute infectious contact |
| Spontaneous diagnosis | `delta = 0.025` per day | `delta = 0.14` per day, converted to slot probabilities |
| Recovery | `gamma = 0.12` per day | `gamma = 0.12` per day, converted to slot probabilities |
| Screening | 20 tests/day starting Day 8 | Four rounds/day, up to five tests/round (20/day) |
| Retesting | Five-day cooldown | 192 previously recorded slots cooldown |
| Test performance | Sensitivity 0.90; specificity 1 | Sensitivity 0.90; specificity 1 |

These epidemiological values are **fixed modeling assumptions**, **not estimates calibrated from clinical or Haslemere infection records**. The per-day and per-five-minute transmission hazards have different time units.

We compare **no testing**, **random selection**, **degree-based targeting**, **recent confirmed-case contact targeting**, a **hand-designed adaptive heuristic** (recent one-hop confirmed-case exposure, two-hop exposure, and degree), **mean-field-risk optimization**, and **fitted-risk optimization**. In the extended Haslemere experiment, two fitted-risk optimizers use the *same* risk model and objective but different past-only proxies for upcoming contacts:

- **Recency-weighted:** older observed contacts receive exponentially decreasing weight (12-hour e-folding time).
- **Cumulative-history:** every previously observed contact receives equal weight; the rate is normalized by the number of recorded slots.

**Both** use the *same genuinely time-varying transmission network*. The comparison tests the value of the **contact-history summary used in allocation**, not temporal transmission versus a fixed static transmission network.

## Experimental design

| Evaluation | Static SBM | Extended Haslemere |
| --- | ---: | ---: |
| Risk model training epidemics | 90 | 24 |
| Held-out risk-evaluation epidemics | 24 | 10 |
| Paired policy-evaluation epidemics | 30 per SBM condition | 12 per visibility setting |
| Main observation settings | 25--100% of true edges | 50% and 100% of observed dyads |

The primary policy outcome is the **percentage ever infected by Day 50**, including initial infections. **New infections** exclude those initially infected; fractional numbers in result tables are averages across simulated epidemics. Policies are compared with matched initial conditions, network/observation draws, and underlying epidemic random innovations. Their realized epidemics can diverge because the policies make different testing decisions.

Paired 95% **Monte Carlo intervals** reflect variability across simulated epidemic replications **conditional on the specified simulation model**. They are not confidence intervals for real-world intervention effects.

## Main results

### Hidden-infection prediction on held-out simulations

| Setting | Estimator | Brier score (lower is better) | Log loss (lower is better) | AUC (higher is better) |
| --- | --- | ---: | ---: | ---: |
| Static-network risk evaluation | Mean-field filter | 0.0476 | 0.218 | 0.691 |
| Static-network risk evaluation | Fitted logistic model | **0.0455** | **0.181** | **0.736** |
| Extended Haslemere risk evaluation | Mean-field filter | 0.0239 | 0.0876 | 0.875 |
| Extended Haslemere risk evaluation | Fitted logistic model | **0.0139** | **0.0591** | **0.898** |

The fitted models improve descriptive hidden-infection prediction. Better prediction **does not automatically imply** better decisions or fewer infections.

### Static SBM: community structure and incomplete observation

Mean ever infected by Day 50, **complete network observation** (30 paired epidemics per condition):

| Testing policy | `r = 1` | `r = 12` |
| --- | ---: | ---: |
| Random | 46.7% | 44.8% |
| Degree | 36.5% | 34.1% |
| Recent confirmed contacts | 37.9% | 31.3% |
| Adaptive heuristic | 35.0% | 30.4% |
| Mean-field optimizer | 38.1% | 36.3% |
| **Fitted-risk optimizer** | **32.6%** | **30.0%** |

The fitted optimizer has the lowest mean infection percentage in these two displayed conditions, but its paired advantage over the adaptive heuristic is not clearly resolved by Monte Carlo uncertainty, nor does its relative advantage increase monotonically with community separation. Reducing edge visibility generally worsens outcomes for network-based policies; it can also **reverse their ranking** in individual settings. Thus, neither fitted optimization nor the adaptive heuristic is shown to be systematically more robust to incomplete contacts.

### Extended 50-day dynamic Haslemere

Mean percentage ever infected by Day 50, **100% observed dyads**, across **12 paired epidemics**:

| Testing policy | Ever infected | Mean new infections | Mean tests used |
| --- | ---: | ---: | ---: |
| No testing | 39.98% | 158.5 | 0 |
| Random | 35.70% | 138.4 | 1,000 |
| Degree | 34.84% | 134.4 | 1,000 |
| Recent confirmed contacts | 24.70% | 86.8 | 1,000 |
| Adaptive heuristic | 25.69% | 91.5 | 1,000 |
| Mean-field optimizer | 28.70% | 105.6 | 1,000 |
| **Recency-weighted fitted optimizer** | **22.87%** | **78.2** | **1,000** |
| **Cumulative-history fitted optimizer** | **22.76%** | **77.8** | **1,000** |

The paired recency-minus-cumulative difference under full observation is **+0.11 percentage points** (95% MC interval **[-1.52, +1.73]**), giving no clear advantage to recency weighting. The recency-weighted fitted optimizer reduces the mean infection percentage by **12.83 percentage points** relative to random testing. Its advantage over the adaptive heuristic is less conclusive.

Under **50% dyad visibility**, the recency-weighted and cumulative-history fitted optimizers yield **27.27%** and **25.36%** ever infected, respectively (adaptive heuristic: **25.76%**). The paired recency-minus-cumulative difference is **+1.92 percentage points** (95% MC interval **[+0.15, +3.69]**). This result is exploratory, given the 12 replications and the assumed daily-block resampling process; it does not establish general superiority of cumulative histories outside this setup.

## Reproducing the experiments

Requirements: **Python 3.10+**, the packages in `requirements.txt`, and the public **headerless** Haslemere file at `data/Kissler_DataS1.csv`. The supplied project bundle includes this CSV; otherwise obtain it from the [original data repository](https://github.com/skissler/haslemere).

Run commands **from the project root**. Limiting BLAS threads can help keep resource usage predictable.

### Primary: extended 50-day Haslemere

```bash
python3 -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 test_extended_haslemere.py

# Re-evaluate using the supplied fitted risk model:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 extended_temporal_haslemere.py --skip-fit --days 50 --reps 12

# Alternatively, refit on 24 training and 10 held-out trajectories before evaluation:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 extended_temporal_haslemere.py --days 50 --train 24 --heldout 10 --reps 12

# Regenerate extended result tables and figures from saved outputs:
python3 create_extended_assets.py
```

Key saved artifacts include `models/extended_temporal_risk_model.joblib`, `results/extended_haslemere_*.csv`, `results/extended_risk_*.csv`, and `figures/extended_haslemere_visibility.png`.

### Primary: static SBM experiments

```bash
python3 test_model_based.py

# Refit the static-network risk estimator:
python3 fit_risk_model.py --csv data/Kissler_DataS1.csv --train-reps 90 --valid-reps 24

# Generate matched baseline runs and then model-based comparisons:
python3 run_structure_sensitivity.py --reps 30
python3 run_haslemere_visibility.py --csv data/Kissler_DataS1.csv --reps 30
python3 run_optimized_experiments.py --csv data/Kissler_DataS1.csv --reps 30

# Regenerate paper tables and SBM figures:
python3 create_model_based_assets.py
```

**Checkpoint caution:** If you change model coefficients, simulation parameters, or policy code, first remove stale `results/optimized_sbm_checkpoint.csv` and `results/optimized_haslemere_checkpoint.csv`. The optimized comparison runner expects compatible baseline results, with the matching evaluation seed and settings. Its outputs include `results/optimized_sbm_*.csv`, `models/risk_model.joblib`, and `figures/optimized_community.png` / `figures/optimized_sbm_visibility.png`.

### Supplementary: three-day measured-only Haslemere sensitivity (Appendix E)

```bash
python3 test_temporal_haslemere.py

# Refit the separate three-day risk model and evaluate 15 paired epidemics:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 temporal_haslemere.py --train 60 --heldout 16 --reps 15

# Or evaluate with the saved fitted three-day model:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 temporal_haslemere.py --skip-fit --reps 15
```

Optional smaller batch runs are supported using `--skip-fit --batch-start 0 --batch-count 3` (then starts 3, 6, 9, and 12) followed by `--aggregate-batches`. Clear existing `results/temporal_batch_*.csv` files before combining runs generated with different settings. The three-day results are **not** interchangeable with the primary 50-day outcomes.

## Project files

| Path | Purpose |
| --- | --- |
| `data/Kissler_DataS1.csv` | Public, headerless Haslemere proximity observations |
| `extended_temporal_haslemere.py` | Main 50-day temporal network, SHCR simulations, fitting, and policy comparisons |
| `temporal_haslemere.py` | Supplementary original three-day temporal experiment |
| `model_based_core.py` | Static-network simulator, factorized filter, and testing-allocation methods |
| `fit_risk_model.py` | Static-network fitted infection-risk model |
| `run_structure_sensitivity.py`, `run_optimized_experiments.py` | Controlled SBM experiments and model-based comparisons |
| `results/` | Saved per-replicate outcomes, contrasts, diagnostics, and LaTeX tables |
| `models/` | Saved separately fitted risk models |
| `figures/` | Paper figures and supplementary plots |
| `analysis.Rmd` | Companion analysis of saved outputs |
| `index.html` | Interactive project demonstration webpage |
| `create_model_based_assets.py`, `create_extended_assets.py` | Regenerate figures and result tables |
| `report.tex`, `references.bib` | LaTeX source and references in the project package |

## Interpretation and limitations

- **Simulation study, not a clinical effectiveness estimate.** Haslemere supplies observed contacts, while *all* disease states, diagnoses, testing results, and contacts after Day 3 are simulated.
- **Short, nonrepresentative contact sample.** Only three days from 469 volunteers inform the temporal extension; unrecorded people and nights, new ties, and changing behavior are not modeled.
- **Simplified disease progression.** The SHCR model assumes immediate infectiousness after infection, no latent exposed state, assumed rather than calibrated hazards, instantaneous test results, and no false-positive tests.
- **Stylized missingness.** Dyads are removed independently at random from the policy's observations, rather than modeling systematic missing contacts or noisy retained measurements.
- **Myopic optimization and limited simulation precision.** Additive expected-prevention scores ignore competing infectors, overlapping benefits, and downstream chains. The 12-replicate temporal comparisons especially warrant caution.

**Main takeaway:** Better risk prediction and more detailed temporal weighting need not yield a uniformly better testing policy. Network completeness matters, but the value of sophisticated optimization relative to a simple adaptive heuristic depends on the simulation setting.

## Data provenance and related work

- Kissler et al. (2018), *Sparking The BBC Four Pandemic: Leveraging citizen science and mobile phones to model the spread of disease*. [Data repository](https://github.com/skissler/haslemere).
- Firth et al. (2020), *Using a real-world network to model localized COVID-19 control strategies*. [DOI](https://doi.org/10.1038/s41591-020-1036-8).
- Huang, Ke, and Jin (2022), *Allocation of COVID-19 Testing Budget on a Commute Network of Counties*. [arXiv](https://arxiv.org/abs/2110.04381).

See the accompanying final course report for the full model specification, paired Monte Carlo contrasts, and appendices.
