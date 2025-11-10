import yfinance as yf
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
# 🧭 Primary + fallback tickers
# -----------------------------------------------------------------------------
tickers = {
    "Momentum": ["ICICIMOMENTUM.NS", "^NDX"],      # Nifty Momentum ETF → Nasdaq 100
    "Value": ["ICICIVALUE.NS", "^DJI"],            # Nifty Value ETF → Dow Jones
    "Quality": ["ICICIQUAL.NS", "^GSPC"],          # Nifty Quality ETF → S&P 500
    "SmallCap": ["NIPPONSMALL.NS", "^RUT"],        # Nifty SmallCap ETF → Russell 2000
}

Path("data").mkdir(exist_ok=True)
all_data = []

def try_download(symbol):
    """Try fetching one symbol safely."""
    try:
        df = yf.download(symbol, start="2018-01-01", interval="1mo", progress=False)
        if not df.empty:
            print(f"✅ Data fetched for {symbol} ({len(df)} rows)")
            return df
        print(f"⚠️ Empty data for {symbol}")
        return None
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None


# -----------------------------------------------------------------------------
# 🌐 Try each ticker with fallback
# -----------------------------------------------------------------------------
for label, options in tickers.items():
    print(f"\n📈 Fetching {label} ...")
    df = None
    for sym in options:
        df = try_download(sym)
        if df is not None:
            df = df.reset_index()[["Date", "Close"]]
            df["index"] = label
            df.rename(columns={"Close": "price"}, inplace=True)
            all_data.append(df)
            break
    if df is None:
        print(f"🚫 Failed all sources for {label}")

if not all_data:
    raise RuntimeError("❌ No data fetched! Check connection or symbols.")

# -----------------------------------------------------------------------------
# 💾 Save results
# -----------------------------------------------------------------------------
merged = pd.concat(all_data)
merged.sort_values(["index", "Date"], inplace=True)
merged.to_csv("data/index_prices.csv", index=False)

print(f"\n✅ Saved → data/index_prices.csv with {len(merged)} rows")
print("🧱 Index breakdown:\n", merged.groupby("index").size())
