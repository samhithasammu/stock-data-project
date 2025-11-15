import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Config ----------
RAW_CSV = "stock_market.csv"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
SCR_DIR = OUT_DIR / "screenshots"
SCR_DIR.mkdir(exist_ok=True)

# ---------- Helpers ----------
def snake_case(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_]", "", s)
    return s.lower()

MISSING_TOKENS = {"", "na", "n/a", "null", "-", "none", "nan", "NA", "NaN"}
def normalize_missing(x):
    if isinstance(x, str) and x.strip() in MISSING_TOKENS:
        return np.nan
    return x

def parse_validated(x):
    if pd.isna(x): return pd.NA
    x = str(x).strip().lower()
    if x in {"yes","y","true","1"}: return True
    if x in {"no","n","false","0"}: return False
    return pd.NA

# ---------- 1) Load raw CSV ----------
print("Loading:", RAW_CSV)
df = pd.read_csv(RAW_CSV, quotechar='"', skipinitialspace=True, dtype=str, keep_default_na=False)

# ---------- 2) Clean / Normalize ----------
# headers -> snake_case
df.columns = [snake_case(c) for c in df.columns]

# trim whitespace for strings
df = df.applymap(lambda v: v.strip() if isinstance(v, str) else v)

# normalize obvious missing tokens
df = df.applymap(normalize_missing)

# parse trade_date (expect mm/dd/YYYY)
if 'trade_date' in df.columns:
    df['trade_date'] = pd.to_datetime(df['trade_date'], format="%m/%d/%Y", errors='coerce')
else:
    raise KeyError("Expected 'Trade Date' column (becomes trade_date).")

# ticker: uppercase and normalize
if 'ticker' in df.columns:
    df['ticker'] = df['ticker'].astype(str).str.strip().replace({'nan': None})
    df.loc[df['ticker'].isin([None, 'None', '']), 'ticker'] = np.nan
    df['ticker'] = df['ticker'].str.upper()

# numeric conversions
for col in ('open_price', 'close_price'):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'volume' in df.columns:
    # keep pandas nullable integer
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64')

# sector/currency/exchange/notes tidy
for c in ('sector','currency','exchange','notes'):
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace({'nan': None})
        if c == 'sector':
            df[c] = df[c].where(df[c].isna(), df[c].str.title())

# validated -> boolean-ish
if 'validated' in df.columns:
    df['validated'] = df['validated'].apply(parse_validated).astype('boolean')

# drop rows lacking both date & ticker
df = df[~(df['trade_date'].isna() & df['ticker'].isna())].copy()

# deduplicate
df = df.drop_duplicates()

# reorder columns if present
preferred = ['trade_date','ticker','open_price','close_price','volume','sector','validated','currency','exchange','notes']
cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
df = df[cols]

# ---------- 3) Save cleaned outputs ----------
cleaned_csv = OUT_DIR / "cleaned.csv"
df.to_csv(cleaned_csv, index=False, date_format="%Y-%m-%d")
print("Wrote cleaned CSV:", cleaned_csv)

cleaned_parquet = OUT_DIR / "cleaned.parquet"
try:
    df.to_parquet(cleaned_parquet, index=False)
    print("Wrote cleaned Parquet:", cleaned_parquet)
except Exception as e:
    print("Parquet write failed (pyarrow/fastparquet missing). CSV saved as fallback.")
    cleaned_parquet = None

# ---------- 4) ANALYSIS A: Daily average close by sector (time series)
# We'll compute avg close per sector per date and plot top-3 sectors by overall row count
agg_a = (df.dropna(subset=['close_price','sector'])
          .groupby(['trade_date','sector'], as_index=False)
          .agg(avg_close=('close_price','mean'),
               n=('close_price','size')))

# choose top 3 sectors by data availability
top_sectors = (agg_a.groupby('sector')['n'].sum().sort_values(ascending=False).head(3).index.tolist())
plot_a = agg_a[agg_a['sector'].isin(top_sectors)].pivot(index='trade_date', columns='sector', values='avg_close').sort_index()

# save agg file
agg_a_csv = OUT_DIR / "agg_daily_avg_close_by_sector.csv"
agg_a.to_csv(agg_a_csv, index=False, date_format="%Y-%m-%d")
try:
    agg_a.to_parquet(OUT_DIR / "agg_daily_avg_close_by_sector.parquet", index=False)
except: pass

# plot top 3 sectors time series
plt.figure(figsize=(10,5))
plot_a.plot(marker='o', linewidth=1.5)
plt.title("Daily Average Close — Top 3 Sectors")
plt.xlabel("Date")
plt.ylabel("Average Close Price")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
chart_a = SCR_DIR / "chart_daily_avg_close_top3_sectors.png"
plt.savefig(chart_a)
plt.close()
print("Saved chart A:", chart_a)

# ---------- 5) ANALYSIS B: Top tickers by average volume (bar chart)
agg_b = (df.dropna(subset=['volume'])
           .groupby('ticker', as_index=False)
           .agg(avg_volume=('volume','mean'), count=('volume','size'))
           .sort_values('avg_volume', ascending=False))

agg_b_csv = OUT_DIR / "agg_top_tickers_by_avg_volume.csv"
agg_b.to_csv(agg_b_csv, index=False)
try:
    agg_b.to_parquet(OUT_DIR / "agg_top_tickers_by_avg_volume.parquet", index=False)
except: pass

# take top 10 tickers (or fewer)
top_n = agg_b.head(10)
plt.figure(figsize=(8,5))
plt.bar(top_n['ticker'], top_n['avg_volume'])
plt.title("Top 10 Tickers by Average Volume")
plt.xlabel("Ticker")
plt.ylabel("Average Volume")
plt.xticks(rotation=45)
plt.tight_layout()
chart_b = SCR_DIR / "chart_top10_tickers_avg_volume.png"
plt.savefig(chart_b)
plt.close()
print("Saved chart B:", chart_b)

# ---------- 6) ANALYSIS C: Rolling 7-day volatility of daily returns for top ticker by volume
# prepare price series per ticker, compute daily return, then 7-day rolling std (volatility)
# choose top ticker by avg volume from agg_b
if not agg_b.empty:
    top_ticker = agg_b.iloc[0]['ticker']
    ticker_prices = (df.dropna(subset=['close_price'])
                       .loc[lambda d: d['ticker']==top_ticker, ['trade_date','close_price']]
                       .sort_values('trade_date'))
    ticker_prices = ticker_prices.set_index('trade_date').asfreq('D')  # fills missing dates with NaN (won't break pct_change)
    ticker_prices['close_price'] = ticker_prices['close_price'].astype(float)
    ticker_prices['daily_return'] = ticker_prices['close_price'].pct_change()
    ticker_prices['rolling_vol_7d'] = ticker_prices['daily_return'].rolling(window=7, min_periods=3).std()

    agg_c = ticker_prices.reset_index()[['trade_date','close_price','daily_return','rolling_vol_7d']].dropna(subset=['rolling_vol_7d'])
    agg_c_csv = OUT_DIR / f"agg_rolling_vol_7d_{top_ticker}.csv"
    agg_c.to_csv(agg_c_csv, index=False, date_format="%Y-%m-%d")
    try:
        agg_c.to_parquet(OUT_DIR / f"agg_rolling_vol_7d_{top_ticker}.parquet", index=False)
    except: pass

    # plot rolling volatility
    plt.figure(figsize=(10,4))
    plt.plot(agg_c['trade_date'], agg_c['rolling_vol_7d'], marker='o', linewidth=1)
    plt.title(f"7-day Rolling Volatility (std of returns) — {top_ticker}")
    plt.xlabel("Date")
    plt.ylabel("7-day rolling std (daily returns)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    chart_c = SCR_DIR / f"chart_rolling_volatility_{top_ticker}.png"
    plt.savefig(chart_c)
    plt.close()
    print("Saved chart C:", chart_c)
else:
    print("No volume data available for analysis C (skipping).")

# ---------- 7) Summary of outputs ----------
print("\n--- Done ---")
print("Outputs folder:", OUT_DIR.resolve())
print(" - cleaned.csv")
if cleaned_parquet:
    print(" - cleaned.parquet")
print(" - agg_daily_avg_close_by_sector.csv (and parquet if available)")
print(" - agg_top_tickers_by_avg_volume.csv")
print(f" - agg_rolling_vol_7d_{(''+top_ticker) if not agg_b.empty else 'NA'}.csv (if available)")
print(" - screenshots in:", SCR_DIR.resolve())
