# 05 — Time Series

## Quick Reference

| Method | Type | Best For |
|--------|------|----------|
| ARIMA | Statistical | Univariate, stationary, short-term |
| SARIMA | Statistical | Univariate with seasonality |
| Exponential Smoothing (ETS) | Statistical | Trend + seasonality, business forecasting |
| Prophet | Statistical + ML | Business time series, holiday effects, easy to use |
| LightGBM / XGBoost | ML | Large-scale, multivariate, feature-rich forecasting |
| LSTM / GRU | Deep Learning | Complex nonlinear patterns, long dependencies |
| N-BEATS / PatchTST | Deep Learning | SOTA for univariate/multivariate forecasting |

**Golden rule for time series:** Never random-shuffle for train/test split. Always train on past, test on future.

---

## 1. Core Concepts

### Stationarity

A time series is stationary if its statistical properties (mean, variance, autocorelation) don't change over time.

```
Weak stationarity:
  1. E[Xt] = μ  (constant mean)
  2. Var(Xt) = σ²  (constant variance)
  3. Cov(Xt, Xs) depends only on lag k, not t

Why it matters: most statistical forecasting methods (ARIMA) assume stationarity.
```

**Testing for stationarity:**

```python
from statsmodels.tsa.stattools import adfuller, kpss

# Augmented Dickey-Fuller test
# H0: series has a unit root (non-stationary)
# p < 0.05 → reject H0 → stationary
result = adfuller(df['value'], autolag='AIC')
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
print(f"Stationary: {result[1] < 0.05}")

# KPSS test (complementary)
# H0: series is stationary
# p < 0.05 → reject H0 → non-stationary
stat, p_value, lags, critical = kpss(df['value'], regression='c')
print(f"KPSS p-value: {p_value:.4f}")
```

### Making a series stationary:

```python
# Differencing (removes trend)
df['diff1'] = df['value'].diff(1)     # first difference: X - Xt-1
df['diff2'] = df['diff1'].diff(1)     # second difference (rarely needed)

# Seasonal differencing (removes seasonality)
df['seasonal_diff'] = df['value'].diff(12)  # lag=season period

# Log transform (stabilizes variance)
df['log_value'] = np.log(df['value'])

# Log + differencing (most common for multiplicative series)
df['log_diff'] = np.log(df['value']).diff(1)
```

### Autocorrelation

```
ACF (Autocorrelation Function): correlation of series with its own lagged values
  ACF(k) = Corr(Xt, Xt-k)

PACF (Partial Autocorrelation Function): direct correlation at lag k after removing
  the effect of intermediate lags

Used for: identifying ARIMA order (p, d, q)
```

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df['value'].dropna(), lags=40, ax=ax1)
plot_pacf(df['value'].dropna(), lags=40, ax=ax2)
plt.tight_layout()
plt.show()
```

---

## 2. ARIMA

### Components

```
AR(p): Autoregressive — linear combination of p past values
  Xt = c + φ1*Xt-1 + φ2*Xt-2 + ... + φp*Xt-p + ε

I(d): Integrated — number of differences needed for stationarity

MA(q): Moving Average — linear combination of q past errors
  Xt = c + ε + θ1*εt-1 + θ2*εt-2 + ... + θq*εt-q

ARIMA(p, d, q):
  p = AR order (PACF helps identify: significant spikes at lags 1..p)
  d = differencing order (0 if already stationary, 1 usually sufficient)
  q = MA order (ACF helps identify: significant spikes at lags 1..q)
```

### Reading ACF/PACF for Order Selection

```
AR(p) signature:
  ACF: gradual decay (tails off)
  PACF: sharp cutoff after lag p

MA(q) signature:
  ACF: sharp cutoff after lag q
  PACF: gradual decay (tails off)

ARMA(p,q): both tail off gradually
```

```python
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm   # auto_arima

# Manual ARIMA
model = ARIMA(df['value'], order=(2, 1, 1))  # p=2, d=1, q=1
fitted = model.fit()
print(fitted.summary())

# Forecast
forecast = fitted.forecast(steps=30)
conf_int = fitted.get_forecast(steps=30).conf_int()

# Auto ARIMA (searches for best p,d,q using AIC)
auto_model = pm.auto_arima(df['value'],
                           start_p=0, max_p=5,
                           start_q=0, max_q=5,
                           d=None,          # auto-detect differencing
                           information_criterion='aic',
                           trace=False,     # show search progress
                           error_action='ignore',
                           stepwise=True)
print(auto_model.order)
forecast = auto_model.predict(n_periods=30)
```

### SARIMA (Seasonal ARIMA)

```
SARIMA(p, d, q)(P, D, Q)s

p,d,q = non-seasonal ARIMA orders
P,D,Q = seasonal AR, differencing, MA orders
s     = seasonal period (12 for monthly, 7 for daily-with-weekly-season)
```

```python
# Monthly data with yearly seasonality
sarima = pm.auto_arima(df['value'],
                       seasonal=True, m=12,     # m=seasonal period
                       start_p=0, max_p=3,
                       start_q=0, max_q=3,
                       start_P=0, max_P=2,
                       start_Q=0, max_Q=2,
                       d=1, D=1,
                       information_criterion='aic',
                       trace=True, stepwise=True)
```

---

## 3. Exponential Smoothing (ETS)

More robust than ARIMA for business forecasting. Weighted average where recent observations get higher weight.

```
Simple Exponential Smoothing (SES): no trend, no seasonality
  S_t = α*Xt + (1-α)*St-1     α ∈ (0,1)
  α close to 1: mostly recent data
  α close to 0: long memory

Holt's: trend
Holt-Winters: trend + seasonality (additive or multiplicative)
```

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Holt-Winters with multiplicative seasonality
hw = ExponentialSmoothing(df['value'],
                          trend='add',        # 'add' or 'mul'
                          seasonal='mul',     # 'add' or 'mul'
                          seasonal_periods=12)  # monthly → 12
fitted_hw = hw.fit(optimized=True)   # auto-optimize alpha, beta, gamma

# Forecast
forecast = fitted_hw.forecast(steps=12)
```

**Additive vs Multiplicative seasonality:**
```
Additive: seasonal fluctuations are constant absolute magnitude
  + use when seasonal swing doesn't grow with trend
  Example: 100 units seasonal variation regardless of total volume

Multiplicative: seasonal fluctuations are proportional to level
  + use when seasonal swing grows with trend (more common in business data)
  Example: 10% seasonal variation regardless of base level
```

---

## 4. Prophet (Facebook/Meta)

Designed for business time series with daily/weekly/yearly seasonality and holiday effects. Robust to missing data and outliers.

```
y(t) = trend(t) + seasonality(t) + holidays(t) + ε

Trend: piecewise linear or logistic growth (with changepoints)
Seasonality: Fourier series decomposition
Holidays: custom indicator variables
```

```python
from prophet import Prophet
import pandas as pd

# Prophet requires columns: 'ds' (datetime) and 'y' (value)
df_prophet = df.rename(columns={'date': 'ds', 'value': 'y'})

# Basic model
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',   # or 'additive'
    changepoint_prior_scale=0.05,        # flexibility of trend changes (0.01=rigid, 0.5=flexible)
    seasonality_prior_scale=10,          # strength of seasonality
    holidays_prior_scale=10
)

# Add custom seasonality
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Add holidays
holidays = pd.DataFrame({'ds': ['2024-12-25', '2026-01-01'], 'holiday': ['christmas', 'new_year']})
m = Prophet(holidays=holidays)

m.fit(df_prophet)

# Forecast
future = m.make_future_dataframe(periods=365)   # 1 year ahead
forecast = m.predict(future)

# Components plot (trend, seasonality, holidays)
m.plot_components(forecast)
plt.show()
```

### When to Use Prophet vs ARIMA

```
Prophet:
  ✓ Strong seasonal patterns (weekly, yearly)
  ✓ Holiday effects matter
  ✓ Missing data present
  ✓ Interpretability needed (component decomposition)
  ✓ Easy API for non-experts

ARIMA:
  ✓ No clear seasonality
  ✓ Short memory processes
  ✓ Statistical inference on parameters needed
  ✓ Multivariate extensions (VAR)
```

---

## 5. ML-Based Forecasting (LightGBM/XGBoost)

Transform the time series forecasting problem into supervised learning by creating lag features.

### Feature Engineering for Time Series

```python
import pandas as pd
import numpy as np

def create_time_features(df, target_col, lags=[1,2,3,7,14,28], windows=[7,14,28]):
    df = df.copy()
    df = df.sort_index()

    # Lag features (most important)
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # Rolling window statistics
    for window in windows:
        df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}']  = df[target_col].shift(1).rolling(window=window).std()
        df[f'rolling_max_{window}']  = df[target_col].shift(1).rolling(window=window).max()
        df[f'rolling_min_{window}']  = df[target_col].shift(1).rolling(window=window).min()

    # Calendar features
    df['hour']       = df.index.hour
    df['dayofweek']  = df.index.dayofweek
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['month']      = df.index.month
    df['quarter']    = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Drop rows with NaN from lag creation
    df = df.dropna()
    return df

df_features = create_time_features(df, target_col='sales', lags=[1,7,14,28])

# Time-based train/test split (CRITICAL: no random shuffling)
cutoff = '2026-01-01'
train = df_features[df_features.index < cutoff]
test  = df_features[df_features.index >= cutoff]

X_train = train.drop(columns=['sales'])
y_train = train['sales']
X_test  = test.drop(columns=['sales'])
y_test  = test['sales']
```

### Multi-step Forecasting Strategies

```
Direct: train separate model for each forecast horizon h
  Model1: predict y_{t+1}, Model2: predict y_{t+2}, ...
  Pro: each model optimized for its horizon
  Con: n_horizons × training time

Recursive (one-step ahead iterated):
  Train one model to predict y_{t+1}
  Feed prediction as lag feature for y_{t+2}, etc.
  Pro: single model
  Con: error accumulates over horizon

MIMO (Multiple Input Multiple Output):
  Single model predicts all horizons simultaneously
  Pro: captures horizon correlations; single model
  Con: harder to optimize

Practical: use Direct or Recursive. LightGBM with recursive is often competitive.
```

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

scores = []
for train_idx, val_idx in tscv.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05,
                               num_leaves=31, verbose=-1)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    scores.append(mean_absolute_error(y_val, model.predict(X_val)))

print(f"CV MAE: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

---

## 5.5. Modern Deep Time Series (2022-2025)

Beyond LSTM/GRU — the SOTA for forecasting is now transformer-based or foundation-model-based. Worth knowing for senior interviews.

| Model | Year | Idea | Library |
|-------|------|------|---------|
| N-BEATS | 2019 | Stacked fully-connected blocks with backcast/forecast decomposition | darts, neuralforecast |
| N-HiTS | 2022 | N-BEATS + hierarchical multi-rate sampling — handles long horizons better | neuralforecast |
| PatchTST | 2022 | Time series → patches → transformer (like ViT for TS) | neuralforecast, darts |
| iTransformer | 2023 | Inverted transformer — attends across variates, not time | neuralforecast |
| Chronos | 2024 (Amazon) | TS foundation model — tokenize values, train TS → zero-shot forecast | chronos-forecasting |
| TimesFM | 2024 (Google) | Decoder-only TS foundation model, 200M params, zero-shot | timesfm |
| Moirai | 2024 (Salesforce) | Universal TS foundation model — any frequency, any number of variates | uni2ts |
| TimeGPT | 2023 (Nixtla) | Commercial TS foundation model API | Nixtla SDK |

```python
# Zero-shot Forecasting with Chronos (no fitting needed)
from chronos import ChronosPipeline
import torch

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # or -base / -large
    device_map="cuda", torch_dtype=torch.bfloat16,
)

context = torch.tensor(historical_series, dtype=torch.float32)
forecast = pipeline.predict(context, prediction_length=30, num_samples=20)
# forecast: (n_series, n_samples, prediction_length) — full predictive distribution
```

**Senior interview take:** Foundation models work when they're pretrained on (regular cadence, economic/business patterns). They underperform on highly specialized series (semiconductor manufacturing, niche IoT) where domain fine-tuning matters.

**Practical hierarchy in 2025:** naive baseline → ETS / Prophet → LightGBM with lag features → foundation model (Chronos / TimesFM) zero-shot → fine-tuned PatchTST / N-HiTS. Often LightGBM still wins on rich-feature tabular forecasting; foundation models win on cold-start / many-series problems.

### Conformal Prediction Intervals for Forecasts

Forecast point estimates aren't enough — stakeholders want calibrated intervals. **Conformal prediction** gives finite-sample coverage for ANY forecaster (statistical, ML, or deep), with no distributional assumptions.

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx

# Train with conformal prediction wrapping
nf = NeuralForecast(models=[
    NBEATSx(h=30, input_size=100, loss="MAE",
            # Conformal: train + calibration split, output quantile bands
            quantiles=[0.05, 0.5, 0.95])
])
nf.fit(df)
forecasts = nf.predict()  # includes 90% conformal prediction interval
```

Or use MAPIE on top of any forecaster — see `../01_fundamentals/04_model_evaluation.md §6.5`.

---

## 6. Decomposition

Decompose a time series into trend, seasonality, and residual components:

```python
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# Classical decomposition
result = seasonal_decompose(df['value'], model='multiplicative', period=12)
result.plot()
plt.show()

# STL (Seasonal and Trend decomposition using Loess) — more robust
stl = STL(df['value'], period=12, robust=True)
stl_result = stl.fit()
stl_result.plot()
plt.show()

# Extract components
trend    = stl_result.trend
seasonal = stl_result.seasonal
residual = stl_result.resid
```

---

## 7. Train/Test Split for Time Series

**NEVER random split. Always chronological.**

```python
# Chronological split
train = df[df.index < '2024-01-01']
test  = df[df.index >= '2024-01-01']

# Rolling window validation (mimics production deployment)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=0)
for train_idx, test_idx in tscv.split(df):
    # Each split: train on past, test on immediately following window
    pass

# With gap (avoid near-future leakage for lagged features)
tscv_gap = TimeSeriesSplit(n_splits=5, gap=7)  # 7-day gap between train and test
```

### Why Random Split Leaks

If you randomly split a time series: test set contains data from both before and after some training samples. A model predicting March 2024 sales using lag features that include April 2024 data — the model has seen the future — artificially inflated performance.

---

## 8. Forecasting Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | mean\|y-ŷ\| | Easy to interpret; same units |
| RMSE | √mean(y-ŷ)² | Penalizes large errors |
| MAPE | mean\|y-ŷ\|/y×100% | % error; breaks if y=0 |
| sMAPE | mean 2\|y-ŷ\|/(y+ŷ)×100% | Symmetric MAPE; bounded [0,200%] |
| MASE | MAE / MAE_naive | Scale-independent; > 1 means worse than naive |

**Naive forecast baseline:** predict y_{t+h} = y_{t} (last known value). Always compare your model to the naive baseline. A model with MASE > 1 is worse than just repeating yesterday's value.

```python
from sklearn.metrics import mean_absolute_error
import numpy as np

# Naive forecast (shift by 1)
naive_pred = y_test.shift(1).dropna()
naive_mae  = mean_absolute_error(y_test[1:], naive_pred)

# Model forecast
model_mae = mean_absolute_error(y_test, model.predict(X_test))

mase = model_mae / naive_mae
print(f"Model MAE: {model_mae:.3f}")
print(f"Naive MAE: {naive_mae:.3f}")
print(f"MASE: {mase:.3f}")  # < 1 means model beats naive baseline
```

---

## 9. When to Use What

| Scenario | Method | Why |
|----------|--------|-----|
| Univariate, no seasonality, <5K points | ARIMA | Statistical, interpretable |
| Univariate, strong seasonality | SARIMA / Holt-Winters / Prophet | Handles seasonal patterns |
| Business series, holiday effects | Prophet | Easy, interpretable, holiday API |
| Large-scale, many series (demand forecasting) | LightGBM / XGBoost | Feature-rich, multivariate, fast |
| Complex nonlinear patterns | LSTM / GRU | Captures long dependencies |
| SOTA benchmark comparison | N-BEATS, PatchTST, TimesFM | Modern deep learning SOTA |
| Quick baseline | Naive / ETS | Always start with these |

---

## 10. Gotchas

**Random train/test split is the #1 time series mistake.** Use chronological split always. Even CV must be time-aware (TimeSeriesSplit, not KFold).

**Lag features must use shift() — never the current value.** `df['lag_1'] = df['value'].shift(1)` — shift(1) means "yesterday's value available today." `df['lag_1'] = df['value'] - 1` — this is the current value, a direct leak.

**MAPE breaks when actual = 0.** For demand/inventory data where y=0 is common (no sales on some days), use sMAPE or MASE instead.

**ARIMA on non-stationary data gives garbage.** Always test stationarity (ADF test) before fitting ARIMA. Non-stationary → difference until stationary.

**Prophet changepoint_prior_scale controls overfitting.** Too high (>0.5) → Prophet fits every wiggle in training data → poor forecast. Too low (<0.001) → trend is too rigid → misses real trend changes. Default 0.05 is usually good.

**Window features must respect train/test boundary.** Rolling means computed using the test period's data → leakage. Always compute rolling features with `.shift(1)` first so you only use past data.

**ML models need retraining as new data arrives.** Unlike statistical models that update naturally, LightGBM/XGBoost need explicit retraining. Set up a retraining pipeline that runs weekly/monthly.

---

## 11. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| ADF test non-stationary | Trend or seasonality present | Apply differencing (d=1); seasonal differencing |
| ARIMA convergence warning | Wrong order or non-stationary input | Check stationarity; use auto_arima |
| Forecast constant / flat | d too high (over-differenced) | Reduce d; check if series was already stationary |
| LightGBM forecast diverges | Error accumulation in recursive forecasting | Use direct strategy; cap forecasts to reasonable range |
| High error at season boundaries | Model misses seasonal pattern | Add seasonal features; use SARIMA or Prophet |
| Model much worse than naive | Bad lag configuration or leakage | Check shift values; verify train/test split |
| Prophet underfits trend | changepoint_prior_scale too low | Increase to 0.1-0.5; add more changepoints |

---

## 12. Code Reference — Full Pipeline

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# 1. Load and prepare
df = pd.read_csv('sales.csv', parse_dates=['date'], index_col='date')
df = df.sort_index()
df = df.asfreq('D')                # ensure daily frequency
df['sales'] = df['sales'].fillna(0)  # fill missing dates

# 2. Feature engineering
df['lag_1']     = df['sales'].shift(1)
df['lag_7']     = df['sales'].shift(7)
df['lag_28']    = df['sales'].shift(28)
df['roll7_mean']= df['sales'].shift(1).rolling(7).mean()
df['roll28_mean']= df['sales'].shift(1).rolling(28).mean()
df['dayofweek'] = df.index.dayofweek
df['month']     = df.index.month
df = df.dropna()

# 3. Time-based split
train = df[df.index < '2023-12-31']
test  = df[df.index >= '2024-01-01']

features = [c for c in df.columns if c != 'sales']
X_train, y_train = train[features], train['sales']
X_test,  y_test  = test[features],  test['sales']

# 4. Cross-validation with TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=7)
cv_scores = []

for train_idx, val_idx in tscv.split(X_train):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                               num_leaves=31, verbose=-1)
    model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
              eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    cv_scores.append(mean_absolute_error(y_train.iloc[val_idx],
                     model.predict(X_train.iloc[val_idx])))

print(f"CV MAE: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

# 5. Final model and evaluation
final_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1)
final_model.fit(X_train, y_train)
test_mae = mean_absolute_error(y_test, final_model.predict(X_test))
print(f"Test MAE: {test_mae:.3f}")

# 6. MASE vs naive baseline
naive_pred = y_test.shift(1).dropna()
naive_mae  = mean_absolute_error(y_test[1:], naive_pred)
print(f"MASE: {test_mae / naive_mae:.3f}")  # < 1 means model beats naive baseline
```

---

## 13. Interview Q&A (Senior Level)

**Q: Why can't you use regular k-fold cross-validation for time series?**
A: Standard k-fold randomly shuffles data across folds, so a validation fold could contain data from both before and after some training fold data. This causes temporal leakage — the model effectively predicts the past from the future. In time series, the invariant is: the model can only use information available at the time of prediction. TimeSeriesSplit preserves chronological order — each validation fold is always strictly after all training data. Additionally, with random CV, autocorrelation between train and test samples (adjacent time steps are highly correlated) makes CV results overly optimistic.

**Q: When would you choose ARIMA over LightGBM for forecasting?**
A: ARIMA when: (1) univariate, short series (< 1K points) where lag features can't be reliably estimated, (2) you need statistical inference — confidence intervals on coefficients, formal hypothesis tests, (3) series is well-characterized by linear autocorrelation structure, (4) no exogenous variables available. LightGBM when: (1) many external features are available (price, promotions, weather), (2) you have many related series to forecast simultaneously (use global model), (3) series is long enough to support many lag features (> 1K points), (4) relationships are nonlinear. In practice, for business forecasting with rich features, LightGBM nearly always wins. For simple univariate cases, ARIMA/Prophet are often competitive with much less complexity.

**Q: What is the difference between additive and multiplicative seasonality in time series?**
A: Additive: seasonal effect has constant absolute magnitude regardless of the level — if sales are 1000 units in low season and 1500 in high season (seasonal spike = 500 units), then when the trend doubles to 2000 units the seasonal spike stays at 500 units. Multiplicative: seasonal effect is proportional to the level — if high season is 50% above low season, and the trend doubles, the absolute spike also doubles. Test: plot the series. If seasonal amplitude grows with the trend → multiplicative. If constant → additive. Log-transforming a multiplicative series converts it to additive: log(trend × seasonality) = log(trend) + log(seasonality). Most economic/business data is multiplicative.

---

## 14. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| LightGBM for forecasting | `02_tree_models.md` | Same model, time-series feature engineering |
| TimeSeriesSplit vs KFold | `../01_fundamentals/04_model_evaluation.md` | CV strategies |
| Feature engineering for lags | `../01_fundamentals/03_feature_engineering.md` | Lag/rolling features = time series FE |
| LSTM for sequences | `../../2.deep learning/02_architectures/03_rnn_lstm_gru.md` | DL alternative for complex TS |
| Temporal leakage | `../01_fundamentals/04_model_evaluation.md` | Leakage section — most common TS mistake |
| Conformal forecast intervals | `../01_fundamentals/04_model_evaluation.md#05-conformal-prediction-distribution-free-uncertainty` | Distribution-free intervals on any forecaster |
| Transformer architecture for PatchTST | `../../2.deep learning/02_architectures/04_transformer.md` | Background for modern deep TS |

---

## Key Takeaway

**Start with naive baseline → ETS/Prophet → ARIMA → LightGBM.** Never jump to LSTM before trying statistical and tree-based methods — they're often competitive with much less complexity.

**The three most important time series rules:** 1. Chronological train/test split — always. 2. Lag features must use `.shift(1)` — always. 3. Always benchmark against naive forecast (MASE < 1 is your minimum bar).

**Stationarity → ARIMA. Seasonality + holidays → Prophet. Many features + large data → LightGBM. Complex nonlinear → LSTM.**
