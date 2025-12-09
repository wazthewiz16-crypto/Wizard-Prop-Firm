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
            
        # 2. Run Strategy (Primary Signals)
        df = self.strategy.calculate_indicators(df)
        df = self.strategy.generate_signals(df)
        
        # 3. Setup ML Data
        vol = get_volatility(df['Close'], span0=self.config['risk']['volatility_window'])
        
        # Define events
        t_limit = self.config['risk']['time_limit_bars']
        signal_indices = df[df['Primary_Signal'] != 0].index
        
        if len(signal_indices) == 0:
            return {"error": "No primary signals generated."}
            
        events = pd.DataFrame(index=signal_indices)
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
        
        # Features
        features = get_features(df)
        
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index, 'bin']
        
        if X.empty:
             return {"error": "Not enough data for ML features."}

        # 4. Train/Test Split (Chronological)
        split_pct = 1 - self.config['ml']['test_size']
        split_idx = int(len(X) * split_pct)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train = y.iloc[:split_idx] # y_test not needed for train
        
        # Train
        self.ml_model.train(X_train, y_train)
        
        # Predict
        probs = self.ml_model.predict_proba(X_test)
        if probs.shape[1] > 1:
            prob_side_1 = probs[:, 1]
        else:
            prob_side_1 = probs[:, 0]
            
        # 5. Simulate Trading with MtM
        test_indices = X_test.index
        # Get subset of DF corresponding to test period (from first test signal to end)
        if len(test_indices) > 0:
            start_simulation = test_indices[0]
            sim_data = df.loc[start_simulation:].copy()
        else:
            return {"error": "No test data."}

        # Enrich sim_data with Signals/Probs
        sim_data['ML_Prob'] = 0.0
        sim_data.loc[test_indices, 'ML_Prob'] = prob_side_1
        
        # We also need to know the 'outcome' or 't1_real' for the trades to know when they close
        # But we want proper MtM.
        # So we will replicate the Triple Barrier logic 'live' or use the pre-calced exit time?
        # Using pre-calced 't1_real' is safer/faster.
        sim_data['Trade_Exit_Time'] = pd.NaT
        # Map labels info
        # common_test = test_indices.intersection(labels.index)
        # sim_data.loc[common_test, 'Trade_Exit_Time'] = labels.loc[common_test, 't1_real']
        # Actually 't1_real' in labels is the timestamp of exit.
        
        # Start Capital
        initial_capital = 100000
        capital = initial_capital
        equity_curve = [] 
        trades = []
        
        # Track active trade: Only 1 at a time for simplicity in this MVP?
        # Or list of trades. Let's do list.
        active_trades = [] # List of dicts: {'entry_price', 'size', 'side', 'exit_time', 'stop_price', 'take_price'}
        
        # Conf Threshold
        conf_thresh = self.config['ml']['confidence_threshold']
        
        # Benchmark
        # Buy & Hold from start
        start_price = sim_data.iloc[0]['Close']
        bh_size = initial_capital / start_price
        
        for idx, row in sim_data.iterrows():
            current_price = row['Close']
            
            # 1. Manage Active Trades (Mark to Market & Exits)
            unrealized_pnl = 0
            remaining_trades = []
            
            for trade in active_trades:
                entry_px = trade['entry_price']
                side = trade['side'] # 1 or -1
                size = trade['size']
                
                # Check Exit
                # We used 'labels' to determine specific exit outcome.
                # If idx == trade['exit_time'], close it.
                # Note: 't1_real' from labels is the BAR timestamp where barrier hit.
                
                if idx >= trade['exit_time']:
                    # Trade Closes NOW.
                    # PnL realized.
                    # Calculate final return based on price 
                    # (Or use the 'ret' from labels if we want to be exact to barrier logic)
                    # Let's use current_price for MtM consistency, assuming idx matches t1_real
                    
                    if side == 1:
                        pnl = (current_price - entry_px) * size
                    else:
                        pnl = (entry_px - current_price) * size
                        
                    capital += pnl
                    # Log Trade
                    trades.append({
                        'Entry Time': trade['entry_time'],
                        'Exit Time': idx,
                        'Type': 'Long' if side == 1 else 'Short',
                        'Entry Price': entry_px,
                        'Exit Price': current_price,
                        'PnL': pnl,
                        'Return': (pnl / (entry_px * size)) if entry_px else 0
                    })
                    # Trade removed
                else:
                    # Trade still open, calc unrealized
                    if side == 1:
                        upnl = (current_price - entry_px) * size
                    else:
                        upnl = (entry_px - current_price) * size
                    unrealized_pnl += upnl
                    remaining_trades.append(trade)
            
            active_trades = remaining_trades
            
            # 2. Check New Entries
            # Only if index is in test_indices (meaning it has a signal calculation)
            if idx in test_indices:
                # We need to look up if this specific signal is valid
                # Signal is in df['Primary_Signal'] but we need to check the Prob
                # sim_data has 'ML_Prob' populated for these indices
                prob = row['ML_Prob']
                signal = df.loc[idx, 'Primary_Signal']
                
                if signal != 0 and prob > conf_thresh:
                    # Check if we already have a trade? (Pyramiding?)
                    # Let's limit to 1 active trade for clarity/risk
                    if len(active_trades) == 0:
                        # Risk Management: Fixed Fractional Risk
                        # Risk Amount = Current Equity * Risk Per Trade (e.g. 1%)
                        risk_pct = self.config['risk'].get('risk_per_trade', 0.01)
                        risk_amount = capital * risk_pct
                        
                        # Calculate Stop Loss Distance
                        # We used Triple Barrier logic roughly:
                        # SL Distance = Price * Volatility * Multiplier
                        if idx in events.index:
                            # Use volatility at entry time to estimate stop distance
                            vol_at_entry = vol.loc[idx]
                            sl_mult = self.config['risk']['sl_multiplier']
                            sl_dist_pct = vol_at_entry * sl_mult
                            
                            # Sanity check: Minimum stop distance?
                            if sl_dist_pct < 0.002: # Minimum 0.2% stop
                                sl_dist_pct = 0.002
                                
                            entry_price = current_price
                            stop_price_dist = entry_price * sl_dist_pct
                            
                            # Position Size (Units) = Risk Amount / Stop Price Dist
                            size = risk_amount / stop_price_dist
                            
                            # Cap size? Not more than leverage allows. 
                            # Prop firms typically 1:30 or 1:100.
                            # Check notional value vs Equity
                            notional = size * entry_price
                            if notional > capital * 20: # Max 20x leverage cap safety
                                size = (capital * 20) / entry_price
                            
                            # Find exit time from labels
                            if idx in labels.index:
                                exit_time = labels.loc[idx, 't1_real']
                                # If NaT (no exit found?), default to end of sim
                                if pd.isna(exit_time):
                                    exit_time = sim_data.index[-1]
                                
                                active_trades.append({
                                    'entry_time': idx,
                                    'entry_price': current_price,
                                    'side': signal,
                                    'size': size,
                                    'exit_time': exit_time
                                })
            
            # 3. Record Equity
            total_equity = capital + unrealized_pnl
            
            # Benchmark PnL
            bh_equity = bh_size * current_price
            
            equity_curve.append({
                'time': idx, 
                'Equity': total_equity,
                'Benchmark': bh_equity
            })
            
        # Metrics
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve).set_index('time') if equity_curve else pd.DataFrame()
        
        if not trades_df.empty:
            total_net_profit = capital - 100000
            win_rate = len(trades_df[trades_df['PnL'] > 0]) / len(trades_df)
            max_dd = (equity_df['Equity'].cummax() - equity_df['Equity']).max()
            profit_factor = trades_df[trades_df['PnL'] > 0]['PnL'].sum() / abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum()) if len(trades_df[trades_df['PnL'] < 0]) > 0 else float('inf')
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
            "equity_curve": equity_df,
            "trades": trades_df,
            "sim_data": sim_data
        }
