# 1. Machine Learning

**Scope:** classical ML algorithms, statistics, evaluation, and feature engineering. **Tier:** 2 (Theory). All technical explanations for classical ML live here.

## Reading Order

Pick a thread based on what you need:

| If you're learning... | Read in order |
|-----------------------|---------------|
| Stats foundations | `01_fundamentals/01_statistics_foundations.md` → `01b` → `01c` |
| EDA → feature engineering | `01_fundamentals/02_eda.md` → `03_feature_engineering.md` |
| Model evaluation | `01_fundamentals/04_model_evaluation.md` (incl. conformal prediction, fairness) |
| Algorithms — start to finish | `02_algorithms/01_linear_models.md` → `02_tree_models` → `03_unsupervised` → `04_probabilistic` |
| Time series | `02_algorithms/05_time_series.md` → `05b_time_series_end_to_end.md` |
| Semi-supervised / RL | `06_semi_supervised_learning.md` → `07_reinforcement_learning.md` → `10_reinforcement_learning_deep.md` |
| EM / Gaussian Processes | `08_expectation_maximization.md` → `09_gaussian_processes.md` |

---

## Folder TOC

### `01_fundamentals/`

| File | Owns |
|------|------|
| `01_statistics_foundations.md` | Descriptive stats, distributions, hypothesis tests, CLT, Bayes, info theory + bootstrap (SSOT) |
| `01b_probability_and_bayes_end_to_end.md` | Worked example — conditional / joint / Bayes with numbers |
| `01c_statistics_end_to_end.md` | Worked examples — p-values, Bayesian A/B, sequential testing, bootstrap CI |
| `02_eda.md` | EDA framework + automated profiling + data validation + embedding-based EDA |
| `03_feature_engineering.md` | Scaling, encoding, imbalance, feature stores (Feast), embedding features |
| `04_model_evaluation.md` | Classification + regression metrics, CV, **conformal prediction**, fairness, LLM-eval cross-ref |

### `02_algorithms/`

| File | Owns |
|------|------|
| `01_linear_models.md` | Linear/Logistic/Ridge/Lasso/SVM + GLMs (Poisson, Tweedie) + quantile regression |
| `02_tree_models.md` | RF/XGBoost/LightGBM/CatBoost + monotonic constraints + modern tabular FM (TabPFN) |
| `03_unsupervised.md` | K-Means/DBSCAN/PCA/UMAP/Isolation Forest + FAISS clustering at scale |
| `04_probabilistic.md` | Naive Bayes / GMM / HMM / Bayesian regression / Probabilistic programming (PyMC, NumPyro) |
| `05_time_series.md` | ARIMA/SARIMA/Prophet/LightGBM-TS + Chronos / TimesFM (foundation models) |
| `05b_time_series_end_to_end.md` | Worked examples — ARIMA forecast with numbers |
| `06_semi_supervised_learning.md` | Self-training + label prop + co-training + FixMatch + foundation-model alternative |
| `07_reinforcement_learning.md` | MDP, Q-learning, REINFORCE, GridWorld (tabular RL) |
| `08_expectation_maximization.md` | EM algorithm depth — derivation, GMM, missing data |
| `09_gaussian_processes.md` | GP regression — kernels, posterior, sparse GPs |
| `10_reinforcement_learning_deep.md` | REINFORCE/A2C/PPO/GRPO/PPO algorithm depth — SSOT for DPO/RLHF RL framing |

---

## Connections

- **Deep Learning foundations** (universal building blocks): `../2.deep learning/`
- **Modern LLM alignment via DPO/GRPO** (production framing): `../6.llms/06_alignment_follow_ups.md`
- **Production ML** (drift, serving, monitoring): `../10.mlops/`
- **ML system design**: `../11.system_design/`

## Practice

This folder doesn't have a dedicated `code_practice/` phase — the practice sequence starts at notebooks in `../archive/1.machine learning/01_code/`
