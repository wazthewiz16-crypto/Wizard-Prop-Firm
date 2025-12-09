import ccxt
import pandas as pd
import sys

def test_ccxt():
    print("Python Version:", sys.version)
    print("CCXT Version:", ccxt.__version__)
    
    # Init Exchange
    exchange = ccxt.binance({'enableRateLimit': True})
    
    print("\n1. Testing Connectivity...")
    try:
        # Load Markets
        markets = exchange.load_markets()
        print(f"Success! Loaded {len(markets)} markets.")
    except Exception as e:
        print(f"CRITICAL: Failed to load markets. Error: {e}")
        return

    # Check Symbols
    test_symbols = ['BTC/USDT', 'BTCUSDT', 'BTC/USDT:USDT', 'ETH/USDT']
    print("\n2. Checking Symbol Availability...")
    for sym in test_symbols:
        present = sym in markets
        print(f"  - '{sym}' available? {present}")
        
    # Try Fetching
    print("\n3. Trying fetch_ohlcv with 'BTC/USDT'...")
    try:
        data = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=5)
        if data:
            print(f"Success! Fetched {len(data)} bars.")
            print(data[0])
        else:
            print("Warning: fetch_ohlcv returned empty list.")
    except Exception as e:
        print(f"Error fetching 'BTC/USDT': {e}")
        
    print("\n4. Trying fetch_ohlcv with 'BTCUSDT' (Raw)...")
    try:
        data = exchange.fetch_ohlcv('BTCUSDT', '1d', limit=5)
        if data:
            print(f"Success! Fetched {len(data)} bars.")
        else:
            print("Warning: fetch_ohlcv returned empty list.")
    except Exception as e:
        print(f"Error fetching 'BTCUSDT': {e}")

if __name__ == "__main__":
    test_ccxt()
