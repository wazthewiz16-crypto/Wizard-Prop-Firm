# Instructions.md: The Wizwave Quant App

## 1. Project Goal
Build a Python-based proprietary trading application that backtests and executes the "Wizwave" strategy. The app must overlay a Machine Learning "Meta-Labeling" layer to filter false positives, maximizing the Sharpe Ratio for a $100k prop firm account.

## 2. Core Strategy Logic (The "Primary Model")
*Based on the provided 'wizwave_strategy.txt'*
* **Indicators:**
    * `Mango_D1`: Weighted Moving Average (WMA, period 29).
    * `Mango_D2`: Exponential Moving Average (EMA, period 27).
    * `Zone`: EMA High/Low (period 22).
    * `Slope`: % change of D1 over 1 bar.
* **Signal Logic:**
    * **Long:** (D1 > D2) AND (Close > D2) AND (Close > Zone_Mid) AND (Slope > Threshold).
    * **Short:** (D1 < D2) AND (Close < D2) AND (Close < Zone_Mid) AND (Slope < -Threshold).

## 3. Financial Machine Learning Enhancements
*Strict adherence to "Advances in Financial Machine Learning" (López de Prado)* 

### A. Feature Engineering (Stationarity)
* Do not feed raw prices to the model.
* **Fractional Differentiation:** Apply fractional differentiation to preserve memory while achieving stationarity.
* **Input Features:** Volatility (rolling std dev), Log-returns, Slope strength, D1/D2 divergence.

### B. Labeling (The Truth Value)
* **Triple Barrier Method:** * Barrier 1: Profit Take (based on volatility σ).
    * Barrier 2: Stop Loss (based on volatility σ).
    * Barrier 3: Time expiration (e.g., 10 bars).
* *Labeling:* If price hits Barrier 1 first, Label = 1. Else, Label = 0.

### C. Meta-Labeling (Secondary Model)
* **Concept:** The mechanical strategy provides the *side* (Long/Short). The ML model (Random Forest or LSTM) is trained on the features + the Primary Signal.
* **Output:** The model predicts the *probability* (confidence) that the primary signal will hit the profit barrier before the stop loss.
* **Filter:** Only execute trades where Model Confidence > 0.6 (tunable).

## 4. Backtesting & Metrics
* **Engine:** `vectorbt` or `backtrader`.
* **Split:** Chronological Train/Test split (no random shuffling) to prevent Look-Ahead Bias.
* **Metrics:** * Win Rate %
    * Total Net Profit (PnL on $100k)
    * Max Drawdown %
    * Profit Factor
    * Equity Curve Visualization

## 5. UI/UX
* **Framework:** Streamlit or Dash.
* **Controls:** Sliders for "Slope Threshold", "Volatility Multiplier", and "ML Confidence Level".
