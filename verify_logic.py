import pandas as pd
import numpy as np
from strategy import WizwaveStrategy
from ml_utils import get_triple_barrier_labels, get_features
from backtest import BacktestEngine
from data_loader import DataLoader

def test_strategy_logic():
    print("Testing Strategy Logic...")
    # Create synthetic data with a known trend
    # 100 bars
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    close = np.linspace(100, 200, 100) # Strong uptrend
    # Add some noise for High/Low
    df = pd.DataFrame({
        'Close': close,
        'High': close + 1,
        'Low': close - 1,
        'Open': close,
        'Volume': 1000
    }, index=dates)
    
    config = {
        "d1_period": 10,
        "d2_period": 20,
        "min_slope": 0.001,
        "zone_period": 5
    }
    
    strat = WizwaveStrategy(config)
    df = strat.calculate_indicators(df)
    df = strat.generate_signals(df)
    
    # Check if we have signals
    # In a strong uptrend with short D1 (10) > D2 (20), we should have Longs (1)
    # Slope should be positive
    
    signals = df['Primary_Signal'].value_counts()
    print(f"Signals Distribution:\n{signals}")
    
    if 1 not in df['Primary_Signal'].values:
        print("FAIL: No Long signals detected in perfect uptrend.")
    else:
        print("PASS: Long signals detected.")

def test_triple_barrier():
    print("\nTesting Triple Barrier Labeling...")
    # 10 bars
    prices = pd.Series([100, 101, 102, 103, 105, 104, 102, 100, 99, 98])
    # T1 events (barrier 2 bars later)
    events = pd.DataFrame(index=[0, 1])
    events['t1'] = [2, 3] # Indices
    events['trgt'] = 0.01 # 1% target
    events['side'] = 1 # Long
    
    # Run
    # Row 0: P=100, Trgt=1, PT=2 (Target 102). Path: 101, 102. Hit 102 at index 2. Should be 1.
    # Row 1: P=101, Trgt=1, PT=2 (Target 103.02). Path: 102, 103. Hit 103 at index 3. Should be ~1 if close enough or 0 if strict.
    
    labels = get_triple_barrier_labels(prices, events, sl=2, pt=2, molecule=events.index)
    print("Labels:\n", labels)
    
    if labels.iloc[0]['bin'] == 1:
        print("PASS: Labeling works.")
    else:
        print("FAIL: Labeling incorrect.")

def test_full_pipeline():
    print("\nTesting Full Backtest Pipeline...")
    # Mock Loader
    class MockLoader:
        def fetch_data(self, t, source, timeframe):
            dates = pd.date_range(start='2023-01-01', periods=200, freq='1h')
            # Random walk with trend
            np.random.seed(42)
            returns = np.random.normal(0.001, 0.01, 200)
            price = 100 * np.exp(np.cumsum(returns))
            
            return pd.DataFrame({
                'Close': price,
                'High': price * 1.005,
                'Low': price * 0.995,
                'Open': price, 
                'Volume': 1000
            }, index=dates)
            
    loader = MockLoader()
    
    config = {
        "strategy": {"d1_period":10, "d2_period":20, "min_slope":0.0, "zone_period":5, "adx_threshold": 10},
        "risk": {"volatility_window":10, "pt_multiplier":1, "sl_multiplier":1, "time_limit_bars":5, "risk_per_trade": 0.01},
        "ml": {"model_type":"RandomForest", "confidence_threshold":0.5, "test_size":0.5},
        "data": {}
    }
    
    engine = BacktestEngine(loader, config)
    res = engine.run()
    
    if "error" in res:
        print(f"FAIL: Pipeline Error {res['error']}")
    else:
        print("PASS: Pipeline finished.")
        print("Metrics:", res['metrics'])
        # Check Equity Curve Structure
        eq = res['equity_curve']
        if not eq.empty and 'Equity' in eq.columns and 'Benchmark' in eq.columns:
             print("PASS: Equity Curve has 'Equity' and 'Benchmark'.")
        else:
             print("FAIL: Equity Curve missing columns or empty.")
             
        # Check if ADX was calculated
        if 'ADX' in res['sim_data'].columns:
             print(f"PASS: ADX Calculated (Mean: {res['sim_data']['ADX'].mean():.2f})")
        else:
             print("FAIL: ADX column missing.")

if __name__ == "__main__":
    try:
        test_strategy_logic()
        test_triple_barrier()
        test_full_pipeline()
    except Exception as e:
        print(f"CRITICAL FAIL: {e}")
        import traceback
        traceback.print_exc()
