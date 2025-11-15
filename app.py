import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.title("📈 Stock Market Dashboard")

# Load cleaned data
data_path_parquet = Path("outputs/cleaned.parquet")
data_path_csv = Path("outputs/cleaned.csv")

if data_path_parquet.exists():
    df = pd.read_parquet(data_path_parquet)
else:
    df = pd.read_csv(data_path_csv, parse_dates=["trade_date"])

st.success("Cleaned dataset loaded!")

# Sidebar filters
ticker_list = sorted(df["ticker"].dropna().unique())
sector_list = sorted(df["sector"].dropna().unique())

st.sidebar.header("Filters")
selected_ticker = st.sidebar.multiselect("Select ticker(s)", ticker_list)
selected_sector = st.sidebar.multiselect("Select sector(s)", sector_list)

# Apply filters
filtered = df.copy()
if selected_ticker:
    filtered = filtered[filtered["ticker"].isin(selected_ticker)]
if selected_sector:
    filtered = filtered[filtered["sector"].isin(selected_sector)]

st.write("### 🔍 Filtered Data Preview")
st.dataframe(filtered.head(20))

# Chart 1 — Close price over time
st.write("### 📊 Close Price Over Time")

if not filtered.empty:
    fig, ax = plt.subplots()
    for t in filtered["ticker"].dropna().unique():
        temp = filtered[filtered["ticker"] == t]
        temp = temp.sort_values("trade_date")
        ax.plot(temp["trade_date"], temp["close_price"], label=t)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    st.pyplot(fig)
else:
    st.info("Please select at least one ticker with data.")

# Chart 2 — Volume over time
st.write("### 📦 Volume Over Time")

if not filtered.empty:
    fig, ax = plt.subplots()
    for t in filtered["ticker"].dropna().unique():
        temp = filtered[filtered["ticker"] == t]
        temp = temp.sort_values("trade_date")
        ax.plot(temp["trade_date"], temp["volume"], label=t)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Volume")
    st.pyplot(fig)
