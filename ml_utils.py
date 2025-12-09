import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pandas_ta as ta

def get_volatility(close, span0=20):
    """
    Computes daily volatility using exponentially weighted moving standard deviation.
    """
    # Simple Returns
    df0 = close.pct_change()
    # EWM Std Dev
    df0 = df0.ewm(span=span0).std()
    return df0

def get_triple_barrier_labels(prices, events, sl, pt, molecule, min_ret=0, check_pt=True, check_sl=True):
    """
    Simplified Triple Barrier Method.
    
    Args:
        prices (pd.Series): Close prices.
        events (pd.DataFrame): Index of events (t1: vertical barrier timestamp, trgt: target volatility).
        sl (float): Stop Loss multiplier.
        pt (float): Profit Take multiplier.
        molecule (list): Indices to process.
        
    Returns:
        pd.DataFrame: Label (1=profit, 0=loss/time) and completion time.
    """
    # events: index=timestamp, columns=['t1', 'trgt', 'side'] (side is optional here if we just label return touch)
    
    # 1. Apply PT/SL to target volatility
    # If pt > 0 then upper barrier is event['trgt'] * pt
    # If sl > 0 then lower barrier is event['trgt'] * sl
    
    out = pd.DataFrame(index=events.index)
    
    # Loop for simplicity/readability over vectorization for complex path dependent barrier
    # In production, this can be slow, but safe for 1h candles over 2 years.
    
    for loc, row in events.loc[molecule].iterrows():
        t0 = loc
        t1 = row['t1']
        trgt = row['trgt']
        
        # Slice prices from t0 to t1
        # t0 is start, t1 is vertical barrier (time expiry)
        if pd.isna(t1):
            path_prices = prices.loc[t0:]
        else:
            path_prices = prices.loc[t0:t1]
            
        # Normalize path to starting price
        path_returns = (path_prices / prices[loc]) - 1
        
        # Barriers
        # We need to know the 'side' (Long/1 or Short/-1) to define Profit/Loss direction
        side = row.get('side', 1) 
        
        # Setup Barriers
        # Profit Take = trgt * pt
        # Stop Loss = trgt * sl
        
        # If Side is 1 (Long): Profit is UP (+), Loss is DOWN (-)
        # If Side is -1 (Short): Profit is DOWN (-), Loss is UP (+)
        
        out_label = 0 # Default to 0 (Loss/Time limit)
        time_touch = t1
        
        barrier_hit = False
        
        # Iterate path
        for t, r in path_returns.items():
            if t == t0: continue # Skip first
            
            # Check Profit (Barrier 1)
            # Long: r > trgt*pt
            # Short: r < -trgt*pt
            if check_pt and ((side==1 and r > trgt*pt) or (side==-1 and r < -trgt*pt)):
                out_label = 1
                time_touch = t
                barrier_hit = True
                break
                
            # Check Stop Loss (Barrier 2)
            # Long: r < -trgt*sl
            # Short: r > trgt*sl
            if check_sl and ((side==1 and r < -trgt*sl) or (side==-1 and r > trgt*sl)):
                out_label = 0
                time_touch = t
                barrier_hit = True
                break
        
        out.loc[loc, 'ret'] = path_returns.iloc[-1] if not barrier_hit else (trgt*pt if out_label==1 else -trgt*sl) # Approx return
        out.loc[loc, 'bin'] = out_label
        out.loc[loc, 't1_real'] = time_touch
        
    return out

def get_features(df):
    """
    Generates stationary features for ML.
    """
    df = df.copy()
    
    # 1. Log Returns (Stationary)
    df['log_ret'] = np.log(df['Close']).diff()
    
    # 2. Volatility (Stationary-ish)
    df['volatility_20'] = df['log_ret'].rolling(20).std()
    
    # 3. RSI (Stationary bounded 0-100)
    df['rsi'] = ta.rsi(df['Close'], length=14)
    
    # 4. Slope (Already calc in strategy, ensure specific features for ML)
    # We will compute it fresh or rely on what's passed.
    
    # 5. D1/D2 Dist (Stationary relative diff)
    # (D1 - D2) / Close
    # Need to be careful if we don't have columns. Assuming this runs AFTER strategy.
    if 'Mango_D1' in df.columns:
        df['d1_d2_diff'] = (df['Mango_D1'] - df['Mango_D2']) / df['Close']
        
    # Drop NaNs
    df = df.dropna()
    
    # Return only stationary features
    # Ensure 'Slope' is included if it exists (from strategy)
    cols = ['log_ret', 'volatility_20', 'rsi', 'd1_d2_diff']
    if 'Slope' in df.columns:
        cols.append('Slope')
        
    return df[cols]

class MetaLabelingModel:
    def __init__(self, model_type="RandomForest", **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
        
    def train(self, X, y):
        # Time-Series Split (Chronological)
        # We don't do cross-val grid search in this simple MVP, just train.
        # But we should respect order.
        self.model.fit(X, y)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
