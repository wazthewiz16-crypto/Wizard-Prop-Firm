import pandas as pd
import numpy as np
import pandas_ta as ta

class WizwaveStrategy:
    """
    Implements the 'Wizwave' mechanical strategy.
    
    Indicators:
    - Mango_D1: WMA(29)
    - Mango_D2: EMA(27)
    - Zone: EMA(22) of High and Low
    - Slope: % Change of D1
    
    Signals:
    - 1 (Long): D1 > D2 & Close > D2 & Close > Zone_Mid & Slope > min_slope
    - -1 (Short): D1 < D2 & Close < D2 & Close < Zone_Mid & Slope < -min_slope
    """
    
    def __init__(self, config):
        """
        Args:
            config (dict): Strategy config dictionary (e.g. from config.json 'strategy' key)
        """
        self.d1_period = config.get('d1_period', 29)
        self.d2_period = config.get('d2_period', 27)
        self.zone_period = config.get('zone_period', 22)
        self.min_slope = config.get('min_slope', 0.02)
        self.adx_threshold = config.get('adx_threshold', 25) # New ADX param

    def calculate_indicators(self, df):
        """
        Adds indicator columns to the DataFrame.
        """
        # Ensure we don't modify the original implicitly
        df = df.copy()
        
        # 1. Mango D1 (WMA 29)
        try:
            df['Mango_D1'] = ta.wma(df['Close'], length=self.d1_period)
        except Exception:
            weights = np.arange(1, self.d1_period + 1)
            df['Mango_D1'] = df['Close'].rolling(self.d1_period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )

        # 2. Mango D2 (EMA 27)
        df['Mango_D2'] = ta.ema(df['Close'], length=self.d2_period)

        # 3. Zone (EMA 22 of High/Low)
        df['Zone_High'] = ta.ema(df['High'], length=self.zone_period)
        df['Zone_Low'] = ta.ema(df['Low'], length=self.zone_period)
        df['Zone_Mid'] = (df['Zone_High'] + df['Zone_Low']) / 2

        # 4. Slope (Percent change of D1 over 1 bar)
        df['Slope'] = df['Mango_D1'].pct_change() * 100
        
        # 5. ADX (Trend Strength)
        # Check if enough data for ADX (usually needs 14 bars + period)
        try:
            adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            # pandas_ta returns ADX_14, DMP_14, DMN_14 usually
            # We just need ADX value
            if adx_df is not None and not adx_df.empty:
                # Find the ADX column (usually first or named ADX_14)
                adx_col = [c for c in adx_df.columns if c.startswith('ADX')][0]
                df['ADX'] = adx_df[adx_col]
            else:
                df['ADX'] = 0
        except Exception:
            df['ADX'] = 0

        return df

    def generate_signals(self, df):
        """
        Generates 'Primary_Signal' column: 1 (Long), -1 (Short), 0 (Neutral).
        """
        # Ensure indicators exist
        if 'Mango_D1' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Logic Conditions
        # Long
        long_cond = (
            (df['Mango_D1'] > df['Mango_D2']) &
            (df['Close'] > df['Mango_D2']) &
            (df['Close'] > df['Zone_Mid']) &
            (df['Slope'] > self.min_slope) &
            (df['ADX'] > self.adx_threshold) # ADX Filter
        )
        
        # Short
        short_cond = (
            (df['Mango_D1'] < df['Mango_D2']) &
            (df['Close'] < df['Mango_D2']) &
            (df['Close'] < df['Zone_Mid']) &
            (df['Slope'] < -self.min_slope) &
            (df['ADX'] > self.adx_threshold) # ADX Filter
        )
        
        # Initialize Signal column
        df['Primary_Signal'] = 0
        
        # Apply
        df.loc[long_cond, 'Primary_Signal'] = 1
        df.loc[short_cond, 'Primary_Signal'] = -1
        
        return df

if __name__ == "__main__":
    # Test Code
    import json
    
    # Load config mock
    mock_config = {
        "d1_period": 29,
        "d2_period": 27,
        "min_slope": 0.02,
        "zone_period": 22
    }
    
    # Create Dummy Data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    data = {
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 105,
        'Low': np.random.rand(100) * 95,
        'Close': np.linspace(100, 150, 100) # Uptrend
    }
    # Add some noise to make High > Close > Low
    data['High'] = data['Close'] + 1
    data['Low'] = data['Close'] - 1
    
    df_test = pd.DataFrame(data, index=dates)
    
    strat = WizwaveStrategy(mock_config)
    df_res = strat.generate_signals(df_test)
    
    print("Indicators calculated:", df_res.columns.tolist())
    print("Signals head:", df_res['Primary_Signal'].tail())
