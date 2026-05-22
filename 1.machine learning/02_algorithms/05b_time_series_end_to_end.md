# 05b — Time Series End-to-End: Worked Examples

> Full forecasting pipeline traced with numbers. Use this for interviews, not the reference file.

## The Interview Question Pattern

Time series questions in interviews come in 3 forms:
```
1. "How would you forecast X?" → system design answer (which method + why)
2. "What is ARIMA / how does it work?" → conceptual + trace
3. "Walk me through how you'd build this end-to-end" → pipeline trace
```
This file covers all three.

---

## Part 1: ARIMA — Complete Dry Run

### Setup

```
Problem: Forecast next month's sales given 24 months of historical data.

Data (last 6 months for illustration):
Jan: 100, Feb: 98, Mar: 105, Apr: 102, May: 108, Jun: 106  (units: thousand)

Visual inspection: slight upward trend, no obvious seasonality.
Goal: Use ARIMA to forecast July.
```

### Step 1: Check Stationarity

```
Augmented Dickey-Fuller test:
  H0: series has a unit root (non-stationary)
  H1: series is stationary

Run ADF on the 24-month series:
  ADF statistic = -2.1
  p-value = 0.26  > 0.05 → FAIL to reject H0 → NON-STATIONARY

Action: Apply first differencing.
  ΔX_t = x_t - x_{t-1}

Differences:
  Feb-Jan = 98-100 = -2
  Mar-Feb = 105-98 = +7
  Apr-Mar = 102-105 = -3
  May-Apr = 108-102 = +6
  Jun-May = 106-108 = -4

Re-run ADF on differentiated series:
  ADF statistic = -3.8
  p-value = 0.003  < 0.05 → REJECT H0 → STATIONARY after d=1

So: d = 1 (one differencing needed)
```

### Step 2: Identify p and q from ACF/PACF

```
ACF of differenced series:
  Lag 1: -0.35 (significant)
  Lag 2:  0.05 (not significant)
  → ACF cuts off after lag 1 → q = 1 (MA term)

PACF of differenced series:
  Lag 1: -0.35 (significant)
  Lag 2:  0.08 (not significant)
  → PACF cuts off after lag 1 → p = 1 (AR term)

Model selected: ARIMA(1, 1, 1)
```

### Step 3: Fit the Model and Interpret

```
ARIMA(1,1,1) equation:
  (1 - φ1·B)(1 - B)·X_t = (1 + θ1·B)·ε_t

In plain English:
  ΔX_t = c + φ1·ΔX_{t-1} + ε_t + θ1·ε_{t-1}

where ΔX_t = X_t - X_{t-1}  (first difference)

Fitted parameters (from MLE):
  c   =  0.5   (slight upward drift in the differenced series)
  φ1  =  0.3   (AR coefficient — partial dependence on last month's change)
  θ1  = -0.2   (MA coefficient — partial dependence on last error)
  σ²  =  16    (noise variance)
```

### Step 4: Forecast July

```
Last observed value: Jun = 106
Last difference: ΔX_Jun = 106 - 108 = -2
Last residual: ε_Jun = ΔX_Jun - model_Jun = 0 (assume model fit well)

Forecast difference for July:
  ΔX_Jul = c + φ1·ΔX_Jun + θ1·ε_Jun
          = 0.5 + 0.3·(-2) + (-0.2)·0
          = 0.5 - 0.6 + 0
          = -0.1

Forecast for July:
  X_Jul = X_Jun + ΔX_Jul = 106 + (-0.1) = 105.9

95% Prediction Interval:
  ± 1.96 * σ = 1.96 * √16 = 7.8
  Interval: [105.9 - 7.8, 105.9 + 7.8] = [98.1, 113.7]

Answer: "We forecast 105.9 thousand units for July, with a 95% interval of [98.1, 113.7]."
```

### Step 5: Model Diagnostics

```
After fitting, always check residuals:

1. Residuals should be white noise (no autocorrelation)
   • Ljung-Box test: H0 = no autocorrelation in residuals
     p > 0.05 = good fit

2. Residuals should be approximately Normal
   • Q-Q plot or Shapiro-Wilk test

3. No heteroskedasticity (constant variance)
   • Plot residuals over time; no pattern

If residuals show autocorrelation → model order wrong, increase p or q.
```

---

## Part 2: Feature Engineering for ML Forecasting — Full Trace

### Setup

```
Problem: Predict daily restaurant sales for next 7 days.
         You have 2 years of daily sales data + day-of-week + month.

Strategy: Convert to supervised learning with lag features + use LightGBM.
```

### Step 1: Raw Data

```
Date        Sales  DayOfWeek  Month
2024-01-01  1200   0          1    ← Monday
2024-01-02  1350   1          1    ← Tuesday
2024-01-03  1100   2          1    ← Wednesday
...
2024-01-08  1180   0          1    ← Monday (next week)
```

### Step 2: Create Features

```
# For each row (date), create features visible on that date

For 2024-01-08 (Monday, week 2), features are:

  lag_1  = sales on 2024-01-07 (Sunday)  = 950  ← previous day
  lag_7  = sales on 2024-01-01 (Monday)  = 1200  ← same day last week + MOST IMPORTANT
  lag_14 = sales on 2023-12-25 (Monday)  = 1050  ← same day 2 weeks ago

  roll7_mean  = mean(sales days 1-7)  = mean(1200,1350,1100,1400,1250,1100,950) = 1193
  roll7_std   = std of same window = 163
  roll28_mean = mean of last 28 days = 1180

  dayofweek = 0  ← Monday
  month     = 1  ← January
  is_weekend = 0  ← weekday

Target: sales on 2024-01-08 = 1180
```

```python
# Critical: All lag/rolling features use .shift(1) minimum:
df['lag_1']  = df['sales'].shift(1)           # yesterday's sales
df['roll7']  = df['sales'].shift(1).rolling(7).mean()  # NOT .rolling(7).mean() directly
                                                        # that would include today, which is leakage
```

### Step 3: Time-Based Split

```
Training:   2022-01-01 to 2023-12-31  (730 days)
Validation: 2024-01-01 to 2024-03-31  (90 days)  ← strictly after training
Test:       2024-04-01 to 2024-04-30  (30 days)  ← strictly after validation

WRONG: random_state=42 with train_test_split → mixes past and future → leakage
CORRECT: df[df.index < '2024-01-01'] for training
```

### Step 4: Cross-Validation

```
TimeSeriesSplit with n_splits=5:

Split 1: Train [Jan 2022 - Aug 2022],  Val [Sep 2022 - Oct 2022]
Split 2: Train [Jan 2022 - Oct 2022],  Val [Nov 2022 - Dec 2022]
Split 3: Train [Jan 2022 - Dec 2022],  Val [Jan 2023 - Feb 2023]
Split 4: Train [Jan 2022 - Feb 2023],  Val [Mar 2023 - Apr 2023]
Split 5: Train [Jan 2022 - Apr 2023],  Val [May 2023 - Jun 2023]

Each validation fold is ALWAYS AFTER its training fold.
CV MAE = mean(MAE across 5 folds)
```

### Step 5: Evaluation

```
Test set (April 2024):
  Naive forecast (yesterday's value):  MAE = 180
  LightGBM forecast:                   MAE = 95

MASE = Model MAE / Naive MAE = 95 / 180 = 0.53

Interpretation: LightGBM is 47% better than the naive baseline.
MASE < 1 → model beats naive → worth deploying.

MAPE = mean |actual - predicted| / actual × 100
     = 95 / 1150 × 100 = 8.3%  → "we're off by about 8% on average"
```

---

## Part 3: Stationarity — Why It Matters

### Visual Intuition

```
Non-stationary series (DON'T fit ARIMA directly):
  Sales: 100, 108, 110, 115, 120  ← trending upward
  Mean changes over time → violates stationarity

After first differencing:
  Δ: +5, +5, +5, +5  ← constant mean = 5 (growth rate)
  Now stationary → can model this with ARIMA
```

### ADF Test Interpretation

```
ADF test outputs:
  ADF statistic: -1.5
  p-value: 0.52
  Critical values: 1%: -3.48,  5%: -2.87,  10%: -2.57

Decision:
  p-value = 0.52 > 0.05 → fail to reject H0 → NON-STATIONARY
  Also: ADF statistic (-1.5) is greater than 5% critical value (-2.87)
    → same conclusion

Action: apply differencing, retest.

After differencing:
  ADF statistic: -4.2
  p-value: 0.001

  p-value = 0.001 < 0.05 → reject H0 → STATIONARY ✓
```

---

## Part 4: Seasonality — Worked Example

### Identifying Seasonality

```
Problem: Monthly ice cream sales (in thousands):
  Jan: 20, Feb: 22, Mar: 35, Apr: 55, May: 80, Jun: 110,
  Jul: 130, Aug: 125, Sep: 90, Oct: 60, Nov: 35, Dec: 25

Observations:
  - Peaks in Jun-Aug (summer)
  - Troughs in Dec-Feb (winter)
  - Period = 12 months (yearly seasonality)
  - Amplitude grows with level → MULTIPLICATIVE seasonality
```

### STL Decomposition (Manual Trace)

```
Observed = Trend × Seasonal × Residual  (multiplicative)

Step 1: Estimate trend using moving average
  Trend_Jul = mean(Jan..Dec) = (20+22+35+55+80+110+130+125+90+60+35+25)/12 = 65.6

Step 2: Detrend
  Detrended_Jul = Observed_Jul / Trend_Jul = 130 / 65.6 = 1.98

Step 3: Seasonal indices (average detrended values for each month across years)
  SI_Jul = avg(detrended Jul values across all years) = 1.98

Step 4: Residual
  Residual_Jul = Observed_Jul / (Trend_Jul × SI_Jul) = 1.0 (perfect fit → residual = 1)

Seasonal indices for all months (roughly):
  Jan: 0.30, Feb: 0.33, Mar: 0.53, Apr: 0.84, May: 1.22, Jun: 1.68
  Jul: 1.98, Aug: 1.91, Sep: 1.37, Oct: 0.91, Nov: 0.53, Dec: 0.38

Sum of monthly SI = 12 ✓ (they must sum to number of periods)
```

### SARIMA Order Selection

```
For monthly data with yearly seasonality:
  s = 12 (period)

Non-seasonal: look at ACF/PACF of differenced series → get p, d, q
Seasonal:     look at ACF/PACF at multiples of 12 (lag 12, 24, 36) → get P, D, Q

Common starting point for monthly business data:
  SARIMA(1,1,1)(1,1,1)s..

Meaning: one AR term + one differencing + one MA term (non-seasonal)
         one seasonal AR + one seasonal differencing + one seasonal MA (at lag 12)

Use auto_arima to search the space; treat it as a starting point and verify diagnostics.
```

---

## Part 5: How to Answer "Build a Forecasting System" in an Interview

### The Answer Template (45 seconds)

```
"I'd approach this in three phases:

First, data and exploration. I'd decompose the series (trend, seasonality, residual),
check stationarity with the ADF test, and look at ACF/PACF plots to understand the
autocorrelation structure.

Second, model selection. I'd start with a naive baseline — just use last week's value.
Then try ETS or Prophet for interpretability, then ARIMA if the series has linear
autocorrelation, and finally LightGBM with lag features if we have many external
features like weather, promotions, or events.

Third, evaluation. I'd use TimeSeriesSplit cross-validation — never random split —
and report MASE relative to the naive baseline. MASE < 1 means we beat naive,
which is the minimum bar for deployment. I'd also monitor MAPE weekly in production
and retrain monthly or when drift is detected."
```

### The Decision Table (memorize this)

| Scenario | Best Choice |
|----------|------------|
| Single series, < 1K points | ETS or ARIMA |
| Single series, strong seasonality | SARIMA or Holt-Winters |
| Business series, holiday effects | Prophet |
| Many features (weather, promos, etc.) | LightGBM with lag features |
| Many similar series (global model) | LightGBM (one model for all) |
| Complex nonlinear, very long series | LSTM / Temporal Fusion Transformer |
| SOTA benchmark comparison | N-BEATS, PatchTST, TimesFM |
| Always first | Naive / always start with naive baseline |

---

## Part 6: Top 5 Time Series Interview Questions

**Q1: Why can't you use k-fold CV on time series?**

K-fold randomly shuffles → validation fold contains data from before training samples → temporal leakage. The model learns to "predict the past from the future." Use TimeSeriesSplit: each validation fold is always AFTER its training fold. Also: adjacent time steps are autocorrelated → random CV gives overly optimistic estimates.

**Q2: What is stationarity and why does ARIMA need it?**

Stationarity = constant mean, constant variance, covariance depends only on lag not time. ARIMA assumes the patterns (autocorrelation structure) are stable over time. If mean is trending, yesterday's pattern ≠ today's pattern → model can't generalize. Fix: differencing removes trend (d=1 usually sufficient). ADF test to verify.

**Q3: When would you use Prophet over ARIMA?**

Prophet: strong seasonal patterns (weekly + yearly), holiday effects matter, missing data present, need interpretable component decomposition, non-expert users reading the output. ARIMA: no clear seasonality, need statistical inference on parameters, multivariate extension needed (VAR model), short series. In practice: try both, compare MASE on held-out test set, pick winner.

**Q4: What is MASE and why use it over MAPE?**

MASE = Model MAE / Naive MAE. MASE < 1 = model beats naive baseline (minimum bar for deployment). MASE = 0.6 means model error is 40% lower than naive. MAPE breaks when actual = 0 (division by zero). Common for demand data (no sales on some days). MASE is scale-free and works with zero actuals.

**Q5: A client says "my ARIMA forecast is flat/constant" — what happened?**

Over-differenced. If d is too high, you removed too much signal. Check: did you apply differencing when series was already stationary? Fix: reduce d; use ADF test to determine correct differencing order. Other causes: AR coefficients near zero (φ ≈ 0) → model predicts constant mean. MA coefficients cancel the signal. Start with d=1, verify stationarity, check ACF/PACF after differencing.

---

## Key Takeaway

**Time Series Pipeline:**
```
1. Plot + decompose (trend, seasonal, residual)
2. Test stationarity (ADF) + difference if needed + set d
3. Read ACF/PACF → set p and q
4. Fit ARIMA(p,d,q) + check residuals (Ljung-Box, Q-Q plot)
5. Forecast + report with prediction intervals
```

**For ML approach:**
```
1. Create lag features with .shift() — NEVER use current value
2. TimeSeriesSplit CV — NEVER random split
3. Benchmark against naive (MASE < 1 = minimum bar)
4. MAPE for stakeholders, MASE for model selection
```

**ARIMA assumes stationarity. Prophet handles seasonality + holidays. LightGBM wins when you have many features and large data. Always start with the naive baseline.**
