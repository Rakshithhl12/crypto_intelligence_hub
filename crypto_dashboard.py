# crypto_dashboard.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random

def dashboard():
    # ----------------- THEME -----------------
    theme = st.sidebar.radio("Theme", ["Dark", "Light"])
    if theme == "Dark":
        bg = "#0B1D33"
        text_color = "#F0F5F9"
        sub_color = "#A0AEC0"
        card_bg = "rgba(255,255,255,0.05)"
        gradient = "linear-gradient(90deg,#0D47A1,#1976D2,#1DE9B6)"
    else:
        bg = "#F5F7FA"
        text_color = "#1F2937"
        sub_color = "#4B5563"
        card_bg = "rgba(0,0,0,0.03)"
        gradient = "linear-gradient(90deg,#3B82F6,#10B981,#06B6D4)"

    # ----------------- CUSTOM CSS -----------------
    st.markdown(f"""
    <style>
    body {{ background-color: {bg}; font-family:'Segoe UI',sans-serif; }}
    h1, h2, h3 {{ font-weight:bold; }}
    h4 {{ color:{sub_color}; margin-top:-10px; }}

    /* Section Titles */
    .section-title {{
        background: {gradient};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size:28px;
        font-weight:bold;
        text-align:center;
        margin-bottom:20px;
    }}

    /* Metric Cards */
    .metric-box {{
        background: {card_bg};
        backdrop-filter: blur(14px);
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align:center;
        margin-bottom: 20px;
    }}
    .metric-box:hover {{
        transform: translateY(-6px);
        box-shadow: 0 16px 40px rgba(0,0,0,0.6);
    }}
    .metric-value {{
        font-size:32px; 
        font-weight:bold; 
        color:#0D1B2A; 
        background: #ffffff88; 
        padding: 5px 12px;
        border-radius: 12px;
        display:inline-block;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }}
    .metric-label {{ color:{sub_color}; font-size:16px; font-weight:600; }}
    .positive {{ color:#10B981; font-weight:bold; }}
    .negative {{ color:#EF4444; font-weight:bold; }}

    /* Buttons */
    .gradient-btn {{
        background: {gradient};
        padding: 10px 25px;
        border-radius: 10px;
        color: #fff;
        font-weight:bold;
        transition: 0.3s;
        display:inline-block;
        margin-top:10px;
        text-decoration:none;
    }}
    .gradient-btn:hover {{ opacity:0.85; }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: {gradient};
        color: #fff;
        padding: 15px;
        border-radius: 20px;
    }}
    [data-testid="stSidebar"] .css-1d391kg {{ color:#fff; font-weight:bold; }}

    /* Header Animation */
    @keyframes fadeIn {{
        0% {{opacity: 0; transform: translateY(-20px);}}
        100% {{opacity: 1; transform: translateY(0);}}
    }}
    </style>
    """, unsafe_allow_html=True)

    # ----------------- HEADER -----------------
    st.markdown(f"""
    <div style="
        background: {gradient};
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 8px 40px rgba(0,0,0,0.5);
        animation: fadeIn 1.5s ease-in-out;
    ">
        <h1 style="
            font-size: 48px;
            font-weight: 900;
            color: white;
            text-shadow: 2px 2px 20px rgba(0,0,0,0.5);
            margin-bottom: 10px;
        ">💎 Crypto Intelligence Hub</h1>
        <h4 style="
            color: #E0E0E0;
            font-weight: 500;
            font-size: 20px;
            text-shadow: 1px 1px 10px rgba(0,0,0,0.3);
        ">Professional Real-Time Crypto Dashboard</h4>
    </div>
    """, unsafe_allow_html=True)

    # ----------------- SIDEBAR CONTROLS -----------------
    coins = st.sidebar.multiselect("Select Cryptocurrencies", ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"], default=["BTC-USD","ETH-USD"])
    days = st.sidebar.slider("Time Range (Days)", 30, 365, 120)
    show_ma = st.sidebar.checkbox("Show Moving Averages", True)

    # ----------------- DATA LOAD -----------------
    @st.cache_data
    def load_multi_coin(symbols):
        data = {}
        for s in symbols:
            df = yf.download(s, start="2020-01-01")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df["MA20"] = df["Close"].rolling(20).mean()
            df["MA50"] = df["Close"].rolling(50).mean()
            data[s] = df
        return data

    with st.spinner("Loading market data..."):
        multi_data = load_multi_coin(coins)

    # ----------------- LATEST PRICES -----------------
    st.markdown("<h2 class='section-title'>💰 Latest Prices</h2>", unsafe_allow_html=True)
    cols = st.columns(len(coins))
    for i, symbol in enumerate(coins):
        df = multi_data[symbol].tail(2)
        latest, prev = df.iloc[-1], df.iloc[-2]
        price = latest["Close"]
        change = price - prev["Close"]
        pct = (change / prev["Close"])*100
        arrow = "▲" if pct>=0 else "▼"
        cls = "positive" if pct>=0 else "negative"
        with cols[i]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{symbol}</div>
                <div class="metric-value">${price:,.2f}</div>
                <div class="{cls}">{arrow} {pct:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ----------------- CANDLESTICK -----------------
    st.markdown("<h2 class='section-title'>📈 Candlestick Charts</h2>", unsafe_allow_html=True)
    for symbol in coins:
        df = multi_data[symbol].tail(days)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=symbol
        ))
        if show_ma:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color="#10B981",width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(color="#3B82F6",width=2)))
        fig.update_layout(template="plotly_dark" if theme=="Dark" else "plotly_white", height=600, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ----------------- TECHNICAL INDICATORS -----------------
    df = multi_data[coins[0]].tail(days)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100/(1+rs))
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    st.markdown("<h2 class='section-title'>📊 Technical Indicators</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='metric-box'><h3>RSI</h3></div>", unsafe_allow_html=True)
        st.line_chart(df["RSI"])
    with col2:
        st.markdown("<div class='metric-box'><h3>MACD</h3></div>", unsafe_allow_html=True)
        st.line_chart(df[["MACD","Signal"]])

    # ----------------- AI INSIGHT -----------------
    st.markdown("<h2 class='section-title'>🧠 AI Market Insight</h2>", unsafe_allow_html=True)
    latest = df.iloc[-1]
    if df["RSI"].iloc[-1]<30 and df["MACD"].iloc[-1]>df["Signal"].iloc[-1]:
        ai_signal, ai_color, ai_msg = "🟢 BUY","#10B981","Strong momentum with oversold recovery"
    elif df["RSI"].iloc[-1]>70:
        ai_signal, ai_color, ai_msg = "🔴 SELL","#EF4444","Overbought zone – pullback likely"
    else:
        ai_signal, ai_color, ai_msg = "🔵 HOLD","#3B82F6","Market stable – wait for confirmation"
    st.markdown(f"""
    <div class="metric-box" style="background:{gradient};">
        <h2 style="color:{ai_color};">{ai_signal}</h2>
        <p style="color:#FAFAFA;">{ai_msg}</p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------- LSTM FORECAST -----------------
    st.markdown("<h2 class='section-title'>🤖 LSTM 7-Day Forecast</h2>", unsafe_allow_html=True)
    data_scaled = df["Close"].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data_scaled)
    X, y = [], []
    for i in range(60,len(scaled)):
        X.append(scaled[i-60:i,0])
        y.append(scaled[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0],X.shape[1],1))
    model = Sequential([
        LSTM(50,return_sequences=True,input_shape=(60,1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam",loss="mse")
    model.fit(X,y,epochs=5,batch_size=32,verbose=0)
    last_60 = scaled[-60:].reshape((1,60,1))
    pred = model.predict(last_60)
    pred_price = scaler.inverse_transform(pred)[0][0]
    st.markdown(f"<div class='metric-box'><h3>Predicted Price</h3><h2>${pred_price:,.2f}</h2></div>", unsafe_allow_html=True)

    # ----------------- PRICE ALERT -----------------
    st.markdown("<h2 class='section-title'>🔔 Price Alerts</h2>", unsafe_allow_html=True)
    target_price = st.number_input("Set alert price", value=float(latest["Close"]))
    if latest["Close"]>=target_price:
        st.success(f"🚀 Alert! {coins[0]} crossed ${target_price:,.2f}")
    else:
        st.info(f"⏳ Waiting for {coins[0]} to reach ${target_price:,.2f}")

    # ----------------- NEWS -----------------
    st.markdown("<h2 class='section-title'>📰 Crypto News</h2>", unsafe_allow_html=True)
    feed = feedparser.parse("https://cointelegraph.com/rss")
    for article in feed.entries[:5]:
        st.markdown(f"""
        <div class="metric-box">
            <h4>{article.title}</h4>
            <p style="color:#A0AEC0;font-size:14px;">{article.published}</p>
            <a href="{article.link}" target="_blank" class="gradient-btn">Read full story →</a>
        </div>
        """, unsafe_allow_html=True)

    # ----------------- BACKTEST -----------------
    st.markdown("<h2 class='section-title'>📊 MA Crossover Backtest</h2>", unsafe_allow_html=True)
    df_bt = df.copy()
    df_bt["Signal"]=0
    df_bt["Signal"][20:] = np.where(df_bt["MA20"][20:]>df_bt["MA50"][20:],1,-1)
    df_bt["Returns"]=df_bt["Close"].pct_change()*df_bt["Signal"].shift(1)
    df_bt["Cumulative"]=(1+df_bt["Returns"]).cumprod()
    st.line_chart(df_bt["Cumulative"])

    # ----------------- RL DEMO -----------------
    st.markdown("<h2 class='section-title'>🧠 RL Suggested Action</h2>", unsafe_allow_html=True)
    actions = ["BUY","HOLD","SELL"]
    state = (latest["RSI"]>70, latest["RSI"]<30)
    Q = {}
    for _ in range(100):
        s = random.choice([True,False])
        a = random.choice(actions)
        reward = random.choice([-1,0,1])
        Q[(s,a)] = Q.get((s,a),0) + 0.1*(reward - Q.get((s,a),0))
    best_action = max(range(len(actions)), key=lambda i: Q.get((state,actions[i]),0))
    st.markdown(f"<div class='metric-box'><h3>RL Action: {actions[best_action]}</h3></div>", unsafe_allow_html=True)

    # ----------------- FOOTER -----------------
    st.markdown(f"<hr><p style='text-align:center;color:{sub_color};font-size:14px;'>💎 Built by Rakshith HL • Crypto Intelligence Platform • 2026</p>", unsafe_allow_html=True)
