import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from polygon import RESTClient
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Constants ---
TRADING_DAYS_PER_YEAR = 252
BENCHMARK_SYMBOL = ".SPX"  # S&P 500 on Polygon

# --- Session State ---
for key in ["polygon_api_key", "instruments_df", "benchmark_hist", "custom_index_history", "comparison_df", "metrics"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "metrics" else {}

# --- Load Polygon API Key ---
def load_polygon_key():
    try:
        return st.secrets["polygon"]["api_key"]
    except KeyError:
        st.error("❌ Missing `polygon.api_key` in `.streamlit/secrets.toml`")
        st.stop()

POLYGON_API_KEY = load_polygon_key()

@st.cache_resource(ttl=3600)
def get_polygon_client(api_key: str) -> RESTClient:
    return RESTClient(api_key=api_key)

client = get_polygon_client(POLYGON_API_KEY)

# --- Data Fetching Functions ---
@st.cache_data(ttl=86400)
def load_us_tickers() -> pd.DataFrame:
    """Load all active US stock tickers."""
    tickers = []
    for t in client.list_tickers(market="stocks", active=True, limit=10000):
        tickers.append({
            "symbol": t.ticker,
            "name": t.name or t.ticker,
            "exchange": t.primary_exchange or "US",
            "type": t.type
        })
    return pd.DataFrame(tickers)

@st.cache_data(ttl=3600)
def fetch_historical(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Fetch daily OHLCV from Polygon."""
    aggs = []
    for a in client.list_aggs(
        ticker=symbol,
        multiplier=1,
        timespan="day",
        from_=start.isoformat(),
        to=end.isoformat(),
        limit=50000
    ):
        aggs.append({
            "date": datetime.fromtimestamp(a.timestamp / 1000),
            "open": a.open,
            "high": a.high,
            "low": a.low,
            "close": a.close,
            "volume": a.volume
        })
    df = pd.DataFrame(aggs)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df[["open", "high", "low", "close", "volume"]]

@st.cache_data(ttl=60)
def get_ltp(symbol: str) -> float | None:
    """Get near-real-time last price."""
    try:
        snap = client.get_snapshot(symbol=symbol)
        return snap.ticker.last_trade.price if snap and snap.ticker and snap.ticker.last_trade else None
    except:
        return None

# --- Performance Metrics ---
def calculate_metrics(returns: pd.Series, benchmark_returns: pd.Series = None, risk_free=0.04) -> dict:
    if returns.empty: return {}
    ann_ret = ((1 + returns).prod() ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1) * 100
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    sharpe = (ann_ret / 100 - risk_free) / (ann_vol / 100) if ann_vol > 0 else np.nan

    # Drawdown
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum / peak - 1) * 100
    max_dd = dd.min()

    # Beta/Alpha (if benchmark provided)
    beta = alpha = np.nan
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        cov = np.cov(returns, benchmark_returns)[0, 1]
        var = benchmark_returns.var()
        if var > 0:
            beta = cov / var
            bench_ann = ((1 + benchmark_returns).prod() ** (TRADING_DAYS_PER_YEAR / len(benchmark_returns)) - 1)
            alpha = (ann_ret / 100 - (risk_free + beta * (bench_ann - risk_free))) * 100

    return {
        "Annualized Return (%)": round(ann_ret, 2),
        "Annualized Volatility (%)": round(ann_vol, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Beta (vs SPX)": round(beta, 2),
        "Alpha (%)": round(alpha, 2)
    }

# --- Index Construction ---
def build_index_history(constituents: pd.DataFrame, start: datetime.date, end: datetime.date) -> pd.Series:
    """
    constituents: DataFrame with columns ['symbol', 'weight']
    Returns normalized index value series (base=100).
    """
    price_data = {}
    for _, row in constituents.iterrows():
        hist = fetch_historical(row["symbol"], start, end)
        if not hist.empty:
            price_data[row["symbol"]] = hist["close"]
    
    if not price_data:
        raise ValueError("No historical data for any constituent.")
    
    prices_df = pd.DataFrame(price_data).ffill().bfill()
    weights = constituents.set_index("symbol")["weight"]
    common_syms = weights.index.intersection(prices_df.columns)
    if common_syms.empty:
        raise ValueError("No overlapping symbols between weights and price data.")
    
    aligned_prices = prices_df[common_syms]
    aligned_weights = weights[common_syms]
    index_raw = (aligned_prices * aligned_weights).sum(axis=1)
    return (index_raw / index_raw.iloc[0]) * 100

# --- Streamlit UI ---
st.title("🇺🇸 US Index Construction & Benchmarking")
st.markdown("Upload constituents, build a custom index, and benchmark against the S&P 500.")

# --- Step 1: Upload Constituents ---
st.subheader("1. Upload Index Constituents")
uploaded = st.file_uploader("CSV with columns: `symbol`, `weight`", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    if not {"symbol", "weight"}.issubset(df.columns):
        st.error("❌ CSV must contain `symbol` and `weight` columns.")
    else:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df = df.dropna(subset=["symbol", "weight"])
        total_w = df["weight"].sum()
        if total_w <= 0:
            st.error("❌ Total weight must be positive.")
        else:
            df["weight"] = df["weight"] / total_w
            st.session_state.constituents = df[["symbol", "weight"]]
            st.success(f"✅ Loaded {len(df)} constituents. Weights normalized to sum = 1.")

# --- Step 2: Configure & Build ---
if "constituents" in st.session_state:
    st.subheader("2. Build Index History")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today().date() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today().date())
    
    if st.button("🚀 Build Index & Benchmark"):
        try:
            # Fetch benchmark
            bench_hist = fetch_historical(BENCHMARK_SYMBOL, start_date, end_date)
            if bench_hist.empty:
                st.error(f"❌ No data for benchmark {BENCHMARK_SYMBOL}")
            else:
                st.session_state.benchmark_hist = bench_hist["close"]
                bench_norm = (bench_hist["close"] / bench_hist["close"].iloc[0]) * 100

                # Build custom index
                index_series = build_index_history(st.session_state.constituents, start_date, end_date)
                st.session_state.custom_index_history = index_series

                # Align and combine
                combined = pd.DataFrame({
                    "Custom Index": index_series,
                    "S&P 500 (.SPX)": bench_norm
                }).dropna()
                st.session_state.comparison_df = combined

                # Metrics
                custom_ret = combined["Custom Index"].pct_change().dropna()
                bench_ret = combined["S&P 500 (.SPX)"].pct_change().dropna()
                st.session_state.metrics = calculate_metrics(custom_ret, bench_ret)

                st.success("✅ Index and benchmark built successfully!")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# --- Step 3: Results ---
if "comparison_df" in st.session_state and not st.session_state.comparison_df.empty:
    st.subheader("3. Performance Comparison")
    
    # Chart
    fig = go.Figure()
    for col in st.session_state.comparison_df.columns:
        fig.add_trace(go.Scatter(
            x=st.session_state.comparison_df.index,
            y=st.session_state.comparison_df[col],
            mode='lines',
            name=col
        ))
    fig.update_layout(
        title="Cumulative Performance (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Index Value",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    if st.session_state.metrics:
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame([st.session_state.metrics]).T
        st.dataframe(metrics_df.style.format("{:.2f}"), use_container_width=True)

    # Constituents with Live Prices
    st.subheader("Constituents (Live Prices)")
    constituents = st.session_state.constituents.copy()
    constituents["Live Price"] = constituents["symbol"].apply(get_ltp)
    constituents["Weighted Contribution"] = constituents["Live Price"] * constituents["weight"]
    total_value = constituents["Weighted Contribution"].sum()
    st.dataframe(constituents.style.format({
        "weight": "{:.2%}",
        "Live Price": "${:.2f}",
        "Weighted Contribution": "${:.2f}"
    }), use_container_width=True)
    st.metric("Current Index Value (Estimate)", f"${total_value:,.2f}")

    # Download
    csv = st.session_state.comparison_df.to_csv().encode("utf-8")
    st.download_button("📥 Download Performance Data", csv, "index_comparison.csv", "text/csv")
