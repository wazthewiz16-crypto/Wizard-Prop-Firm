import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from backtest import BacktestEngine
from data_loader import DataLoader

# Setup
st.set_page_config(page_title="Wizwave Quant App", layout="wide", page_icon="📈")
st.title("🧙‍♂️ Wizwave Quant App")
st.markdown("### Momentum Strategy + ML Meta-Labeling")

# Sidebar Config
st.sidebar.header("Configuration")

# 1. Asset & Data
st.sidebar.subheader("Asset Selection")
ticker = st.sidebar.text_input("Ticker", value="BTC/USDT")
source = st.sidebar.selectbox("Source", ["ccxt", "yfinance"], index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d", "2d", "4d", "1w"], index=1)

# 2. Strategy Params
st.sidebar.subheader("Strategy Logic")
d1_period = st.sidebar.slider("Mango D1 (WMA)", 10, 100, 29)
d2_period = st.sidebar.slider("Mango D2 (EMA)", 10, 100, 27)
min_slope = st.sidebar.slider("Min Slope (%)", 0.0, 0.1, 0.02, step=0.01)

# 3. ML & Risk Params
st.sidebar.subheader("ML & Risk")
confidence = st.sidebar.slider("ML Confidence Threshold", 0.5, 0.9, 0.60)
stop_loss = st.sidebar.slider("Stop Loss Multiplier", 0.5, 5.0, 2.0)
take_profit = st.sidebar.slider("Take Profit Multiplier", 0.5, 10.0, 2.0)
vol_window = st.sidebar.slider("Volatility Window", 5, 50, 20)

if st.sidebar.button("Run Backtest"):
    # Build Config
    config = {
        "strategy": {
            "d1_period": d1_period,
            "d2_period": d2_period,
            "min_slope": min_slope,
            "zone_period": 22
        },
        "risk": {
            "volatility_window": vol_window,
            "pt_multiplier": take_profit,
            "sl_multiplier": stop_loss,
            "time_limit_bars": 12 
        },
        "ml": {
            "model_type": "RandomForest",
            "confidence_threshold": confidence,
            "test_size": 0.2
        },
        "data": {}
    }
    
    # Initialize Engine
    loader = DataLoader()
    engine = BacktestEngine(loader, config)
    
    with st.spinner("Fetching Data & Training Model..."):
        try:
            results = engine.run(ticker=ticker, source=source, timeframe=timeframe)
            
            if "error" in results:
                st.error(results["error"])
            else:
                # Dispay Metrics
                metrics = results["metrics"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Net Profit", f"${metrics['Net Profit']:.2f}")
                c2.metric("Win Rate", f"{metrics['Win Rate']*100:.2f}%")
                c3.metric("Max Drawdown", f"${metrics['Max Drawdown']:.2f}")
                c4.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
                
                # --- NEW CHART: Price + Signals ---
                st.subheader("Price & Signals")
                sim_data = results["sim_data"]
                
                # Plot with Matplotlib for control
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Plot Close Price
                ax.plot(sim_data.index, sim_data['Close'], label='Close', color='gray', alpha=0.5)
                
                # Plot Buy Signals (Longs where trade taken)
                # We need to filter sim_data where valid trade occurred
                valid_trades = results['trades']
                if not valid_trades.empty:
                    longs = valid_trades[valid_trades['Type'] == 'Long']
                    shorts = valid_trades[valid_trades['Type'] == 'Short']
                    
                    if not longs.empty:
                         ax.scatter(longs['Entry Time'], longs['Entry Price'], marker='^', color='green', s=100, label='Buy', zorder=5)
                    if not shorts.empty:
                         ax.scatter(shorts['Entry Time'], shorts['Entry Price'], marker='v', color='red', s=100, label='Sell', zorder=5)
                
                ax.set_title(f"{ticker} Trades")
                ax.set_ylabel("Price")
                ax.legend()
                ax.grid(True, alpha=0.2)
                
                # Use Streamlit's native matplotlib support (or plotly if preferred)
                st.pyplot(fig)
                
                # Equity Curve (Date Axis)
                st.subheader("Equity Curve")
                # Ensure it's treated as a Time Series
                st.line_chart(results["equity_curve"])
                
                # Trade Log
                st.subheader("Trade Log")
                trades = results["trades"]
                st.dataframe(trades, use_container_width=True)
                
                # Debug Data (Optional)
                with st.expander("View signals data"):
                    st.dataframe(results["sim_data"].tail(100))
                    
        except Exception as e:
            st.error(f"Execution failed: {e}")
            raise e
