import ccxt
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class DataLoader:
    """
    Handles data fetching from Crypto Exchanges (via ccxt), Traditional Markets (via yfinance),
    and local CSV files. Supports automatic resampling for non-standard timeframes.
    """
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Initialize safe default exchange (Binance) for public data
        # Try Binance US first if in US? Or handle dynamic.
        # We'll stick to binance but handle errors.
        try:
             self.exchange = ccxt.binance({'enableRateLimit': True})
             self.exchange.load_markets()
        except Exception:
             # Fallback to Binance US
             print("Binance.com failed (likely restricted). Trying Binance US...")
             self.exchange = ccxt.binanceus({'enableRateLimit': True})

    def fetch_data(self, ticker, source='ccxt', timeframe='1h', days=730, csv_path=None):
        """
        Main entry point to fetch data.
        
        Args:
            ticker (str): Symbol (e.g., 'BTC/USDT', '^GSPC').
            source (str): 'ccxt', 'yfinance', or 'csv'.
            timeframe (str): '1h', '4h', '1d', '2d', '4d', '1w'.
            days (int): Lookback period in days (default ~2 years).
            csv_path (str): Path to CSV if source is 'csv'.
            
        Returns:
            pd.DataFrame: OHLCV data with DatetimeIndex.
        """
        print(f"Fetching {ticker} via {source} [{timeframe}]...")
        
        # 1. Handle CSV
        if source == 'csv':
            if not csv_path or not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV path not found: {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
            return self._validate_columns(df)

        # 2. Determine base timeframe for fetching vs resampling
        # If timeframe is standard, fetch directly. If not (e.g. 2d), fetch 1d and resample.
        base_tf, resample_rule = self._parse_timeframe(timeframe)
        
        df = None
        
        # 3. Fetch from Source
        if source == 'ccxt':
            try:
                df = self._fetch_ccxt(ticker, base_tf, days)
                if df is None or df.empty:
                    raise ValueError("CCXT returned empty data.")
            except Exception as e:
                print(f"CCXT Fetch failed: {e}. Falling back to YFinance.")
                # Try to convert ticker for YFinance (BTC/USDT -> BTC-USD)
                yf_ticker = ticker.replace('/', '-')
                # Remove :USDT if present (CCXT format sometimes)
                if ':' in yf_ticker:
                     yf_ticker = yf_ticker.split(':')[0]
                
                df = self._fetch_yfinance(yf_ticker, base_tf, days)
                
        elif source == 'yfinance':
            df = self._fetch_yfinance(ticker, base_tf, days)
        else:
            raise ValueError(f"Unknown source: {source}")
            
        # 4. Resample if necessary
        if resample_rule and df is not None:
             df = self._resample_data(df, resample_rule)

        return self._validate_columns(df)

    def _fetch_ccxt(self, symbol, timeframe, days):
        """Fetches data using CCXT (defaulting to Binance)."""
        # Normalize symbol: CCXT requires BASE/QUOTE (e.g. BTC/USDT)
        # If user input 'BTCUSDT', try to fix it.
        if '/' not in symbol and len(symbol) > 3:
            # Simple heuristic: Assume USDT is the quote if it ends with it, OR try to guess.
            # Most common case: BTCUSDT -> BTC/USDT
            if symbol.upper().endswith('USDT'):
                symbol = f"{symbol[:-4].upper()}/{symbol[-4:].upper()}"
            elif symbol.upper().endswith('USD'):
                symbol = f"{symbol[:-3].upper()}/{symbol[-3:].upper()}"
        
        # Calculate start timestamp (ms)
        since = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
        
        all_ohlcv = []
        limit = 1000  # Binance limit per req
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                # Update 'since' to the last timestamp + 1 timeframe duration
                last_ts = ohlcv[-1][0]
                since = last_ts + 1 
                
                # Safety break if we reached current time
                if last_ts >= self.exchange.milliseconds():
                    break
                    
                if len(ohlcv) < limit:
                    break
                    
            except Exception as e:
                print(f"Error fetching ccxt data: {e}")
                break
                
        if not all_ohlcv:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def _fetch_yfinance(self, ticker, interval, days):
        """Fetches data using yfinance."""
        # YFinance interval mapping
        # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        # We assume 'base_tf' passed here is compatible or mapped.
        
        yf_interval_map = {
            '1h': '1h',
            '1d': '1d',
            '1w': '1wk'
        }
        
        yf_int = yf_interval_map.get(interval, interval) # Fallback to literal
        
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = yf.download(ticker, start=start_date, interval=yf_int, progress=False, multi_level_index=False)
        
        # Clean up YF noise
        if isinstance(df.columns, pd.MultiIndex):
             # Flatten if needed (rare with multi_level_index=False but widely reported YF bug)
             df.columns = df.columns.get_level_values(0)

        # Rename to standard
        df = df.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
        })
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
             df.index = pd.to_datetime(df.index)

        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def _parse_timeframe(self, timeframe):
        """
        Returns (fetch_timeframe, resample_rule).
        If no resampling needed, resample_rule is None.
        """
        # Standard exchange timeframes allowed usually: 1m, 1h, 4h, 1d, 1w
        # We need to handle 2d, 4d specifically as requested.
        
        if timeframe in ['1h', '4h', '1d', '1w']:
            return timeframe, None
        
        if timeframe == '2d':
            return '1d', '2D'
        if timeframe == '4d':
            return '1d', '4D'
            
        # Fallback
        return timeframe, None

    def _resample_data(self, df, rule):
        """Resamples OHLCV data."""
        logic = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        return df.resample(rule).agg(logic).dropna()

    def _validate_columns(self, df):
        """Ensures consistent column naming."""
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if df is None or df.empty:
            return pd.DataFrame(columns=required)
            
        # Capitalize columns just in case
        df.columns = [c.capitalize() for c in df.columns]
        
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Dataframe missing required columns: {missing}")
            
        return df[required]

if __name__ == "__main__":
    # Quick Test
    loader = DataLoader()
    try:
        print("Testing BTC/USDT fetch...")
        df_btc = loader.fetch_data('BTC/USDT', source='ccxt', timeframe='4h', days=10)
        print(df_btc.tail())
        
        print("\nTesting SPX fetch...")
        df_spx = loader.fetch_data('^GSPC', source='yfinance', timeframe='1d', days=10)
        print(df_spx.tail())
    except Exception as e:
        print(f"Test failed: {e}")
