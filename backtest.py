import pandas as pd
import numpy as np
from strategy import WizwaveStrategy
from ml_utils import get_features, get_triple_barrier_labels, MetaLabelingModel, get_volatility

class BacktestEngine:
    def __init__(self, data_loader, config):
        self.loader = data_loader
        self.config = config
        self.strategy = WizwaveStrategy(config['strategy'])
        self.ml_model = MetaLabelingModel(model_type=config['ml'].get('model_type', 'RandomForest'))
        
    def run(self, ticker="BTC/USDT", source="ccxt", timeframe="1h", start_date=None, end_date=None):
        """
        Full pipeline execution: Data -> Strategy -> ML -> Backtest.
        """
        # 1. Fetch Data
        df = self.loader.fetch_data(ticker, source=source, timeframe=timeframe)
        if df.empty:
            raise ValueError("No data found.")
            
        # Filter dates if provided (simple version, loader could handle too)
        # Assuming df has datetime index
        
        # 2. Run Strategy (Primary Signals)
        df = self.strategy.calculate_indicators(df)
        df = self.strategy.generate_signals(df)
        
        # 3. Setup ML Data
        # We need to Label the historical signals to train the model.
        # Volatility for barriers
        vol = get_volatility(df['Close'], span0=self.config['risk']['volatility_window'])
        
        # Get Triple Barrier Labels
        # We only care about labeling when we have a signal.
        # But get_triple_barrier_labels needs to look forward.
        
        # Define events: T1 is expiration.
        # We need a 't1' series. using 'time_limit_bars'
        t_limit = self.config['risk']['time_limit_bars']
        
        # Create events DataFrame for the labeler
        # We only label points where Primary_Signal != 0
        signal_indices = df[df['Primary_Signal'] != 0].index
        
        if len(signal_indices) == 0:
            return {"error": "No primary signals generated."}
            
        events = pd.DataFrame(index=signal_indices)
        events['t1'] = events.index + pd.Timedelta(hours=t_limit) # Approx if 1h, use shift logic for real robustness
        # For non-pandas-freq aware:
        # events['t1'] = pd.Series(df.index, index=df.index).shift(-t_limit).loc[signal_indices] 
        # Using shift on full index is safer
        full_t1 = pd.Series(df.index, index=df.index).shift(-t_limit)
        events['t1'] = full_t1.loc[signal_indices]
        
        events['trgt'] = vol.loc[signal_indices]
        events['side'] = df.loc[signal_indices, 'Primary_Signal']
        
        # Run Labeling
        labels = get_triple_barrier_labels(
            df['Close'], 
            events, 
            sl=self.config['risk']['sl_multiplier'], 
            pt=self.config['risk']['pt_multiplier'], 
            molecule=events.index
        )
        
        # Join Labels back to features
        # Features should be calculated on ALL data, then subset
        features = get_features(df)
        
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index, 'bin']
        
        if X.empty:
             return {"error": "Not enough data for ML features."}

        # 4. Train/Test Split (Chronological)
        # Split point
        split_pct = 1 - self.config['ml']['test_size'] # e.g. 0.8
        split_idx = int(len(X) * split_pct)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train Model
        self.ml_model.train(X_train, y_train)
        
        # Predict on Test Set
        # We want the probability of class 1
        probs = self.ml_model.predict_proba(X_test)
        # Handle if model only sees 1 class
        if probs.shape[1] > 1:
            prob_side_1 = probs[:, 1]
        else:
            prob_side_1 = probs[:, 0] # Fallback
            
        # 5. Simulate Trading on TEST set
        # Reconstruct the test dataframe subset
        # We need the original info (Close, Side) aligned with these predictions
        test_indices = X_test.index
        
        # Simulation loop
        # Start Capital
        capital = 100000
        equity_curve = [] # Initialize empty to avoid mixed types (float vs dict)
        
        # Add initial state if we have the time? 
        # We'll just track from first bar for simplicity or use the first index.
        if not test_indices.empty:
             equity_curve.append({'time': test_indices[0], 'equity': capital})
        
        trades = []
        
        position = 0
        entry_price = 0
        
        # Threshold
        conf_thresh = self.config['ml']['confidence_threshold']
        
        # Combine data for simulation
        sim_data = df.loc[test_indices].copy()
        sim_data['ML_Prob'] = prob_side_1
        # Add the 'ret' from labels (the actual outcome of the Triple Barrier logic)
        # This simplifies backtest to "what happened" according to barriers
        # Real backtest would step bar-by-bar, but Triple Barrier outcome is sufficient for 'Signal Quality'
        sim_data['Actual_Ret'] = labels.loc[test_indices, 'ret']
        
        for idx, row in sim_data.iterrows():
            signal = row['Primary_Signal']
            prob = row['ML_Prob']
            outcome_ret = row['Actual_Ret']
            
            # Logic: IF Signal != 0 AND Prob > Threshold -> TAKE TRADE
            if signal != 0 and prob > conf_thresh:
                # Trade Executed
                # PnL = Capital * outcome_ret (assuming 100% equity allocation for simplicity/compounding)
                # Or Fixed Size? Prompt says "$100k prop firm account"
                # Let's assume fixed risk per trade or fixed allocation. 
                # Let's use 10% allocation per trade to be safe/realistic props.
                alloc = capital * 0.10
                pnl = alloc * outcome_ret
                
                # Estimate Prices for Log
                entry_px = row['Close'] # Approx
                # Return r = (Exit - Entry) / Entry  => Exit = Entry * (1 + r)
                # Note: direction matters. 
                # Long: r = (Exit - Entry)/Entry -> Exit = Entry(1+r)
                # Short: r = (Entry - Exit)/Entry -> Exit = Entry(1-r)
                
                if signal == 1:
                     exit_px = entry_px * (1 + outcome_ret)
                else:
                     exit_px = entry_px * (1 - outcome_ret)
                
                capital += pnl
                trades.append({
                    'Entry Time': idx,
                    'Type': 'Long' if signal == 1 else 'Short',
                    'Prob': prob,
                    'Entry Price': entry_px,
                    'Exit Price': exit_px,
                    'Return': outcome_ret,
                    'PnL': pnl
                })
            
            # Record Equity with Time
            equity_curve.append({'time': idx, 'equity': capital})
            
        # Metrics
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve).set_index('time') if equity_curve else pd.DataFrame()
        
        if not trades_df.empty:
            total_net_profit = capital - 100000
            win_rate = len(trades_df[trades_df['PnL'] > 0]) / len(trades_df)
            max_dd = (equity_df['equity'].cummax() - equity_df['equity']).max()
            profit_factor = trades_df[trades_df['PnL'] > 0]['PnL'].sum() / abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum())
        else:
            total_net_profit = 0
            win_rate = 0
            max_dd = 0
            profit_factor = 0
            
        return {
            "metrics": {
                "Net Profit": total_net_profit,
                "Win Rate": win_rate,
                "Max Drawdown": max_dd,
                "Profit Factor": profit_factor,
                "Total Trades": len(trades_df)
            },
            "equity_curve": equity_df, # Returned as DataFrame with Date Index
            "trades": trades_df,
            "sim_data": sim_data
        }
