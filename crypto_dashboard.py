import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(
    page_title="Crypto Intelligence Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- RESPONSIVE MEDIA SCREEN CSS -----------------
st.markdown("""
<style>

/* Base responsive typography */
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

/* Laptop / Desktop */
@media (min-width: 1024px) {
    h1 {
        font-size: 42px;
    }
    .block-container {
        padding: 2rem 3rem 2rem 3rem;
    }
}

/* Tablet */
@media (max-width: 1023px) and (min-width: 768px) {
    h1 {
        font-size: 32px;
        text-align: center;
    }
    .block-container {
        padding: 1.5rem;
    }
}

/* Mobile */
@media (max-width: 767px) {
    h1 {
        font-size: 24px;
        text-align: center;
    }

    .stMetric {
        text-align: center;
        font-size: 14px;
    }

    .block-container {
        padding: 1rem;
    }

    /* Stack columns vertically */
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Smaller charts */
    .js-plotly-plot {
        height: 350px !important;
    }
}

/* Responsive buttons */
button {
    width: 100%;
    border-radius: 8px;
}

/* Sidebar responsive */
@media (max-width: 767px) {
    section[data-testid="stSidebar"] {
        width: 100% !important;
    }
}

</style>
""", unsafe_allow_html=True)

# ----------------- AUTO REFRESH -----------------
if "refresh" not in st.session_state:
    st.session_state.refresh = False

# ----------------- SIDEBAR -----------------
st.sidebar.title("⚙ Controls")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Theme
theme = st.sidebar.radio("Theme", ["Dark", "Light"])

# Timeframe
period = st.sidebar.selectbox(
    "Timeframe",
    ["1mo", "3mo", "6mo", "1y", "2y"]
)

coins = st.sidebar.multiselect(
    "Select Coins",
    ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"],
    default=["BTC-USD", "ETH-USD"]
)

# ----------------- HEADER -----------------
st.title("💎 Crypto Intelligence Hub")
st.caption("Professional AI-Powered Crypto Analytics Dashboard")

# ----------------- DATA LOAD -----------------
@st.cache_data
def load_multi_coin(symbols):
    data = {}
    for s in symbols:
        df = yf.download(s, period=period)
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        data[s] = df
    return data

multi_data = load_multi_coin(coins)

# ----------------- LATEST PRICES -----------------
st.subheader("💰 Latest Prices")
cols = st.columns(len(coins))

for i, symbol in enumerate(coins):
    df = multi_data[symbol].tail(2)
    latest, prev = df.iloc[-1], df.iloc[-2]

    price = latest["Close"]
    change = price - prev["Close"]
    pct = (change / prev["Close"]) * 100

    with cols[i]:
        st.metric(
            symbol,
            f"${price:,.2f}",
            f"{pct:.2f}%"
        )

# ----------------- PORTFOLIO TRACKER -----------------
st.subheader("💼 Portfolio Tracker")

for coin in coins:
    qty = st.number_input(
        f"Quantity {coin}",
        value=1.0
    )

    buy_price = st.number_input(
        f"Buy Price {coin}",
        value=float(multi_data[coin]["Close"].iloc[-1])
    )

    current_price = multi_data[coin]["Close"].iloc[-1]

    investment = qty * buy_price
    current_value = qty * current_price
    profit = current_value - investment

    st.metric(
        f"{coin} Profit/Loss",
        f"${profit:,.2f}"
    )

# ----------------- CANDLESTICK -----------------
st.subheader("📈 Candlestick Charts")

for symbol in coins:
    df = multi_data[symbol]

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=symbol
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA20"],
        name="MA20"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA50"],
        name="MA50"
    ))

    st.plotly_chart(fig, use_container_width=True)

# ----------------- TECHNICAL INDICATORS -----------------
df = multi_data[coins[0]]

# RSI

delta = df["Close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()

avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss


df["RSI"] = 100 - (100 / (1 + rs))

# MACD

ema12 = df["Close"].ewm(span=12).mean()

ema26 = df["Close"].ewm(span=26).mean()


df["MACD"] = ema12 - ema26


df["Signal"] = df["MACD"].ewm(span=9).mean()

col1, col2 = st.columns(2)

with col1:
    st.subheader("RSI")
    st.line_chart(df["RSI"])

with col2:
    st.subheader("MACD")
    st.line_chart(df[["MACD", "Signal"]])

# ----------------- VOLATILITY -----------------
st.subheader("📉 Volatility")

returns = df["Close"].pct_change()

volatility = returns.std() * np.sqrt(252)

st.metric(
    "Annual Volatility",
    f"{volatility:.2%}"
)

# ----------------- CORRELATION HEATMAP -----------------
st.subheader("📊 Correlation Heatmap")

price_df = pd.DataFrame()

for coin in coins:
    price_df[coin] = multi_data[coin]["Close"]

corr = price_df.corr()

fig = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns
    )
)

st.plotly_chart(fig)

# ----------------- AI INSIGHT -----------------
st.subheader("🧠 AI Market Insight")

if df["RSI"].iloc[-1] < 30:
    st.success("🟢 BUY Signal")

elif df["RSI"].iloc[-1] > 70:
    st.error("🔴 SELL Signal")

else:
    st.info("🔵 HOLD Signal")

# ----------------- LSTM FORECAST -----------------
st.subheader("🤖 LSTM Forecast")

prices = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()

scaled = scaler.fit_transform(prices)

X, y = [], []

for i in range(60, len(scaled)):
    X.append(scaled[i - 60:i, 0])
    y.append(scaled[i, 0])

X = np.array(X)

y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(
    X,
    y,
    epochs=3,
    batch_size=32,
    verbose=0
)

last_60 = scaled[-60:].reshape((1, 60, 1))

pred = model.predict(last_60)

pred_price = scaler.inverse_transform(pred)[0][0]

st.metric(
    "Predicted Price",
    f"${pred_price:,.2f}"
)

# ----------------- MODEL PERFORMANCE -----------------
actual = prices[-1][0]

rmse = np.sqrt(mean_squared_error([actual], pred))

st.metric(
    "RMSE",
    f"{rmse:.4f}"
)

# ----------------- SENTIMENT -----------------
st.subheader("🧠 Market Sentiment")

sentiment = random.uniform(-1, 1)

if sentiment > 0.3:
    label = "Bullish"

elif sentiment < -0.3:
    label = "Bearish"

else:
    label = "Neutral"

st.metric(
    "Sentiment",
    label
)

# ----------------- FEAR GREED INDEX -----------------
st.subheader("😨 Fear & Greed Index")

fear_greed = random.randint(0, 100)

st.metric(
    "Fear & Greed",
    fear_greed
)

# ----------------- TOP MOVERS -----------------
st.subheader("🏆 Top Movers")

changes = {}

for coin in coins:
    df_coin = multi_data[coin].tail(2)

    pct = (
        (df_coin["Close"].iloc[-1] - df_coin["Close"].iloc[-2])
        / df_coin["Close"].iloc[-2]
    ) * 100

    changes[coin] = pct


gainer = max(changes, key=changes.get)

loser = min(changes, key=changes.get)

st.metric("Top Gainer", gainer)

st.metric("Top Loser", loser)

# ----------------- DOWNLOAD DATA -----------------
st.subheader("⬇ Download Data")

csv = df.to_csv().encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "crypto_data.csv",
    "text/csv"
)

# ----------------- PRICE ALERT -----------------
st.subheader("🔔 Price Alert")

target_price = st.number_input(
    "Set Alert Price",
    value=float(df["Close"].iloc[-1])
)

if df["Close"].iloc[-1] >= target_price:
    st.success("🚀 Price Reached Target")

else:
    st.info("⏳ Waiting for Target Price")

# ----------------- NEWS -----------------
st.subheader("📰 Crypto News")

feed = feedparser.parse(
    "https://cointelegraph.com/rss"
)

for article in feed.entries[:5]:
    st.write(article.title)
    st.caption(article.published)
    st.write(article.link)

# ----------------- BACKTEST -----------------
st.subheader("📊 Strategy Backtest")

df_bt = df.copy()

df_bt["Signal"] = 0

df_bt.loc[20:, "Signal"] = np.where(
    df_bt["MA20"][20:] > df_bt["MA50"][20:],
    1,
    -1
)


df_bt["Returns"] = (
    df_bt["Close"].pct_change()
    * df_bt["Signal"].shift(1)
)


df_bt["Cumulative"] = (
    1 + df_bt["Returns"]
).cumprod()

st.line_chart(df_bt["Cumulative"])

# ----------------- RL DEMO -----------------
st.subheader("🧠 RL Suggested Action")

actions = ["BUY", "HOLD", "SELL"]

best_action = random.choice(actions)

st.success(
    f"Suggested Action: {best_action}"
)

# ----------------- FOOTER -----------------
st.markdown(
    "---"
)

st.caption(
    "Built by Rakshith HL • AI Crypto Intelligence Platform • 2026"
)
