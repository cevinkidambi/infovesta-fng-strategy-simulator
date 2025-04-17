import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("testing3.xlsx")
df['date'] = pd.to_datetime(df['date'])
df = df.rename(columns={"FnG_Prev": "FnG_Score"})
df = df[['date', 'FnG_Score', 'IDX80', 'LQ45']].dropna().sort_values('date').reset_index(drop=True)

# Inject custom CSS
st.markdown("""
    <style>
    /* Force content up by overriding Streamlit layout spacing */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }

    .main > div:first-child {
        margin-top: -3rem !important;
    }

    /* Hide Streamlit UI elements */
    #MainMenu, header, footer {
        visibility: hidden;
    }

    /* Background color and text */
    html, body, [class*="css"]  {
        background-color: #121212;
        color: #f5f5f5;
    }

    /* Title font customization */
    .stTitle h1 {
        font-family: 'Orbitron', sans-serif !important;
        color: #FF4B4B !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
        margin-top: 0rem;
    }

    /* Sidebar title + markdown */
    .sidebar .css-1v0mbdj, .sidebar .css-1x8cf1d {
        font-family: 'Inter', sans-serif;
        color: #f0f0f0;
    }

    /* Markdown headings and data labels */
    h2, h3, h4 {
        color: #f5f5f5;
        font-family: 'Inter', sans-serif;
    }

    /* Dataframe & Table formatting */
    .dataframe th, .dataframe td {
        font-size: 0.9rem;
        text-align: center;
        color: white !important;
        background-color: #1e1e1e !important;
    }

    /* Make logo more prominent */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 0.5rem;
        margin-top: 0rem;
    }

    .logo-container img {
        background-color: #fff;
        padding: 4px;
        border-radius: 8px;
    }
    </style>

    <!-- Orbitron Google Font -->
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Sidebar
logo_base64 = base64.b64encode(open("logo_infovesta.png", "rb").read()).decode()
st.markdown(f"""
<div class="logo-container">
    <img src="data:image/png;base64,{logo_base64}" width="300">
</div>
""", unsafe_allow_html=True)

st.title("📈 Infovesta FnG Risk-Adjusted Strategy Simulator")

# Define min/max date from dataset
min_date = df['date'].min().date()
max_date = df['date'].max().date()

# Layout: Index and Risk Mode
with st.sidebar.expander("ℹ️ About this app"):
    st.markdown("""
    **Infovesta FnG Risk-Adjusted Strategy Simulator**

    This tool allows you to simulate performance of various DCA-based investment strategies on **IDX80** and **LQ45**, guided by the **Infovesta Fear & Greed Index**.

    📅 **Data Range**: 4 Feb 2020 — 10 Apr 2025  
    🧠 **Strategies Covered**:  
    - Risk-adjusted using FnG signals  
    - Buy & Hold  
    - Daily & Weekly DCA  
    """)
with st.sidebar.expander("ℹ️ About Infovesta Fear & Greed Index"):
    st.markdown("""
    **What is it?**

    The Infovesta Fear and Greed (FnG) Index is a sentiment indicator ranging from 0 (extreme fear) to 100 (extreme greed), which is inspired by CNN’s Fear & Greed Index. While CNN uses 7 Fear & Greed indicators, the Infovesta FnG Index is an aggregate of 9 distinct indicators, each gauging a specific dimension of stock market activity. These indicators include BTC/IDR, JCI Foreign Flow, GOLD Spot, Market Momentum, IDX80 Mcclellan Summation Index (MSI), IDX80 Mcclellan Volume Summation Index (MVSI), Stock-Bond Spread, USD/IDR Spot, and Stock Price Strength. The Index assesses the deviation of each individual indicator, in relation to its typical fluctuations. The Index assigns equal weight to each indicator to avoid outliers. Each indicator is updated as soon as new data is available, including in weekends and after hours.

    ### 🧠 FnG Index Levels
    Below is the official interpretation of Fear & Greed levels:
    
    | **Levels**         | **Thresholds** |
    |--------------------|----------------|
    | Extreme Greed      | 76–100         |
    | Greed              | 60–75          |
    | Neutral            | 45–59          |
    | Fear               | 26–45          |
    | Extreme Fear       | 0–25           |
    """)
col1, col2 = st.columns(2)
with col1:
    benchmark_index = st.selectbox("📊 Select Benchmark Index", ["IDX80", "LQ45"])
with col2:
    risk_mode = st.selectbox("⚖️ Risk-Adjusted Strategy", ["Aggressive", "Moderate", "Conservative"])

# Layout: Start and End Dates
col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input("📅 Start Date", value=min_date, min_value=min_date, max_value=max_date)
with col4:
    end_date = st.date_input("📅 End Date", value=max_date, min_value=min_date, max_value=max_date)

# Layout: DCA and Thresholds
col5, col6 = st.columns(2)
with col5:
    dca_amount = st.number_input("💰 DCA Amount (Rp)", value=200_000, step=10_000)
with col6:
    include_cash = st.checkbox("Include Uninvested Cash?", value=True)

# Advanced thresholds in expander
with st.expander("⚙️ Fine-tune FnG Thresholds"):
    col7, col8 = st.columns(2)
    with col7:
        buy_threshold = st.slider("Buy threshold (FnG ≤)", 0, 50, value=25, step=1)
    with col8:
        sell_threshold = st.slider("Sell threshold (FnG >)", 50, 100, value=75, step=1)


# Strategy sell rates based on risk mode
sell_rate = {"Aggressive": 0.02, "Moderate": 0.05, "Conservative": 0.10}[risk_mode]

# Filter data
df_filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))].copy()

# Initialize FnG strategy
shares_held, cash_available, cash_invested = 0, 0, 0
fng_values, fng_invested = [], []

for i, row in df_filtered.iterrows():
    fnG, idx_price = row['FnG_Score'], row[benchmark_index]
    if fnG <= buy_threshold:
        buy_amount = dca_amount + cash_available
        shares_bought = buy_amount / idx_price
        shares_held += shares_bought
        cash_invested += dca_amount
        cash_available = 0
    elif fnG > sell_threshold:
        shares_sold = shares_held * sell_rate
        cash_available += shares_sold * idx_price
        shares_held -= shares_sold

    total_value = shares_held * idx_price + cash_available
    fng_values.append(total_value)
    fng_invested.append(cash_invested)

# Buy & Hold
bh_shares = dca_amount / df_filtered[benchmark_index].iloc[0]
df_filtered['BuyHold_Value'] = bh_shares * df_filtered[benchmark_index]
df_filtered['BuyHold_Invested'] = dca_amount

# Daily DCA
daily_shares, daily_invested = 0, 0
daily_vals, daily_invested_list = [], []
for _, row in df_filtered.iterrows():
    daily_shares += dca_amount / row[benchmark_index]
    daily_invested += dca_amount
    daily_vals.append(daily_shares * row[benchmark_index])
    daily_invested_list.append(daily_invested)

df_filtered['DCA_Daily_Value'] = daily_vals
df_filtered['DCA_Daily_Invested'] = daily_invested_list

# Weekly DCA
weekly_shares, weekly_invested = 0, 0
weekly_vals, weekly_invested_list = [], []
for i, row in df_filtered.iterrows():
    if i % 7 == 0:
        weekly_shares += dca_amount / row[benchmark_index]
        weekly_invested += dca_amount
    weekly_vals.append(weekly_shares * row[benchmark_index])
    weekly_invested_list.append(weekly_invested)

df_filtered['DCA_Weekly_Value'] = weekly_vals
df_filtered['DCA_Weekly_Invested'] = weekly_invested_list

# Add FnG values
df_filtered['FnG_Value'] = fng_values
df_filtered['FnG_Invested'] = fng_invested

# Calculate returns
df_filtered['FnG_Return'] = (df_filtered['FnG_Value'] - df_filtered['FnG_Invested']) / df_filtered['FnG_Invested'] * 100
df_filtered['BuyHold_Return'] = (df_filtered['BuyHold_Value'] - df_filtered['BuyHold_Invested']) / df_filtered['BuyHold_Invested'] * 100
df_filtered['DCA_Daily_Return'] = (df_filtered['DCA_Daily_Value'] - df_filtered['DCA_Daily_Invested']) / df_filtered['DCA_Daily_Invested'] * 100
df_filtered['DCA_Weekly_Return'] = (df_filtered['DCA_Weekly_Value'] - df_filtered['DCA_Weekly_Invested']) / df_filtered['DCA_Weekly_Invested'] * 100

# Plot
mpl.rcParams.update({
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': 'white',
    'axes.titlecolor': 'white',
    'legend.edgecolor': 'white',
    'legend.facecolor': 'none',
    'axes.facecolor': 'none',
    'figure.facecolor': 'none'
})
st.subheader("📊 Cumulative Return Comparison")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_filtered['date'], df_filtered['FnG_Return'], label='Risk-Adjusted FnG Strategy')
ax.plot(df_filtered['date'], df_filtered['BuyHold_Return'], label='Buy & Hold')
ax.plot(df_filtered['date'], df_filtered['DCA_Daily_Return'], label='DCA Daily')
ax.plot(df_filtered['date'], df_filtered['DCA_Weekly_Return'], label='DCA Weekly')
ax.set_title(f"Cumulative Return Comparison ({benchmark_index})")
ax.set_ylabel("Return (%)")
ax.set_xlabel("Date")
ax.axhline(0, linestyle="--", color="gray")
ax.legend()
fig.patch.set_alpha(0)
ax.set_facecolor('none')
st.pyplot(fig)


# Final stats
final = df_filtered.iloc[-1]
summary = pd.DataFrame({
    "Strategy": ["FnG (Risk-Adjusted)", "Buy & Hold", "DCA Daily", "DCA Weekly"],
    "Total Invested": [
        final['FnG_Invested'],
        final['BuyHold_Invested'],
        final['DCA_Daily_Invested'],
        final['DCA_Weekly_Invested']
    ],
    "Final Value": [
        final['FnG_Value'],
        final['BuyHold_Value'],
        final['DCA_Daily_Value'],
        final['DCA_Weekly_Value']
    ],
    "Return (%)": [
        final['FnG_Return'],
        final['BuyHold_Return'],
        final['DCA_Daily_Return'],
        final['DCA_Weekly_Return']
    ]
})
# Reward/Risk Calculation
strategies = {
    'FnG (Risk-Adjusted)': df_filtered['FnG_Return'],
    'Buy & Hold': df_filtered['BuyHold_Return'],
    'DCA Daily': df_filtered['DCA_Daily_Return'],
    'DCA Weekly': df_filtered['DCA_Weekly_Return']
}

reward_risk_data = []
for strategy_name in summary["Strategy"]:
    returns = strategies[strategy_name]
    max_return = returns.max()
    min_return = returns.min()
    reward_risk = max_return / abs(min_return) if min_return != 0 else np.nan
    reward_risk_data.append(reward_risk)

summary["Reward/Risk"] = reward_risk_data


def calculate_cagr(final_value, invested, start, end):
    years = (end - start).days / 365.25
    return ((final_value / invested) ** (1 / years) - 1) * 100

summary["CAGR (%)"] = [
    calculate_cagr(final['FnG_Value'], final['FnG_Invested'], start_date, end_date),
    calculate_cagr(final['BuyHold_Value'], final['BuyHold_Invested'], start_date, end_date),
    calculate_cagr(final['DCA_Daily_Value'], final['DCA_Daily_Invested'], start_date, end_date),
    calculate_cagr(final['DCA_Weekly_Value'], final['DCA_Weekly_Invested'], start_date, end_date)
]

# Compute Max Drawdown, Max Downside, Max Upside
max_downsides = []
max_downside_dates = []
max_drawdowns = []
max_drawdown_dates = []
max_upsides = []
max_upside_dates = []

def calculate_max_drawdown(return_series):
    equity_curve = return_series / 100 + 1
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_dd = drawdown.min() * 100
    max_dd_date = return_series.index[drawdown.argmin()]
    return max_dd, df_filtered.loc[max_dd_date, 'date']

for strategy_name in summary["Strategy"]:
    returns = strategies[strategy_name]
    
    # Max Downside
    min_return = returns.min()
    min_date = df_filtered.loc[returns.idxmin(), 'date']
    max_downsides.append(min_return)
    max_downside_dates.append(min_date.strftime('%Y-%m-%d'))

    # Max Upside
    max_return = returns.max()
    max_date = df_filtered.loc[returns.idxmax(), 'date']
    max_upsides.append(max_return)
    max_upside_dates.append(max_date.strftime('%Y-%m-%d'))

    # Max Drawdown
    dd_value, dd_date = calculate_max_drawdown(returns)
    max_drawdowns.append(dd_value)
    max_drawdown_dates.append(dd_date.strftime('%Y-%m-%d'))

# Append to summary
summary["Max Upside (%)"] = max_upsides
summary["Max Upside Date"] = max_upside_dates
summary["Max Downside (%)"] = max_downsides
summary["Max Downside Date"] = max_downside_dates
summary["Max Drawdown (%)"] = max_drawdowns
summary["Max Drawdown Date"] = max_drawdown_dates

# Update table display
st.subheader("📋 Final Strategy Summary")
st.dataframe(summary.set_index("Strategy").style.format({
    "Total Invested": "Rp{:,.0f}",
    "Final Value": "Rp{:,.0f}",
    "Return (%)": "{:.2f}",
    "Reward/Risk": "{:.2f}",
    "Max Upside (%)": "{:.2f}",
    "Max Downside (%)": "{:.2f}",
    "Max Drawdown (%)": "{:.2f}",
    "CAGR (%)": "{:.2f}"

}))
