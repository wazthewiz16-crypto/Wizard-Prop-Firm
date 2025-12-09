from data_loader import DataLoader
import pandas as pd

def test_loader():
    loader = DataLoader() # This might print "Binance.com failed..."
    
    print("\nTesting Data Fetch (Expect Fallback if restricted)...")
    try:
        # Use BTCUSDT to test the normalization + fallback chain
        df = loader.fetch_data("BTCUSDT", source='ccxt', timeframe='1d', days=5)
        
        if df is not None and not df.empty:
            print(f"SUCCESS: Fetched {len(df)} bars.")
            print(df.tail())
        else:
            print("FAIL: DataFrame is empty.")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_loader()
