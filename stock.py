import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Aplikacja Giełdowa AI", layout="wide")

# ===== Funkcje pomocnicze =====
def load_data(symbol, period="1y", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    df = df.rename(columns={
        "Close": "CLOSE",
        "Open": "OPEN",
        "High": "HIGH",
        "Low": "LOW",
        "Volume": "VOLUME"
    })
    df["SMA_50"] = df["CLOSE"].rolling(50).mean()
    df["SMA_200"] = df["CLOSE"].rolling(200).mean()
    df["RSI_14"] = compute_rsi(df["CLOSE"])
    df["EMA_12"] = df["CLOSE"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["CLOSE"].ewm(span=26, adjust=False).mean()
    df["MACD_12_26_9"] = df["EMA_12"] - df["EMA_26"]
    df["MACDs_12_26_9"] = df["MACD_12_26_9"].ewm(span=9, adjust=False).mean()
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(data):
    features = ["SMA_50", "SMA_200", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9"]
    X = data[features]
    y = np.where(data["CLOSE"].shift(-1) > data["CLOSE"], 1, -1)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[:-1], y[:-1])
    return model

def plot_chart(df):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df["CLOSE"], label="CLOSE", color="black")
    ax.plot(df.index, df["SMA_50"], label="SMA 50", color="blue")
    ax.plot(df.index, df["SMA_200"], label="SMA 200", color="orange")

    # Sygnały wejścia/wyjścia
    buy_signals = df[(df["SMA_50"] > df["SMA_200"]) & (df["RSI_14"] < 70)]
    sell_signals = df[(df["SMA_50"] < df["SMA_200"]) & (df["RSI_14"] > 30)]

    ax.scatter(buy_signals.index, buy_signals["CLOSE"], marker="^", color="green", label="Kupno")
    ax.scatter(sell_signals.index, sell_signals["CLOSE"], marker="v", color="red", label="Sprzedaż")

    ax.legend()
    ax.grid(True)
    return fig

# ===== UI =====
st.title("📈 Inteligentna Aplikacja Giełdowa")

symbol = st.text_input("Podaj symbol (np. AAPL, TSLA, EURUSD=X):", "AAPL")
period = st.selectbox("Okres:", ["6mo", "1y", "2y", "5y"])
interval = st.selectbox("Interwał:", ["1d", "1h", "30m"])

if st.button("🔍 Pobierz dane"):
    df = load_data(symbol, period, interval)

    tab1, tab2, tab3 = st.tabs(["📊 Wykres", "⚡ Sygnały", "🤖 AI Predykcja"])

    with tab1:
        st.pyplot(plot_chart(df))

    with tab2:
        last = df.iloc[-1]
        if last["SMA_50"] > last["SMA_200"] and last["RSI_14"] < 70:
            st.success(f"📈 Sygnał: KUPNO (RSI={last['RSI_14']:.2f})")
        elif last["SMA_50"] < last["SMA_200"] and last["RSI_14"] > 30:
            st.error(f"📉 Sygnał: SPRZEDAŻ (RSI={last['RSI_14']:.2f})")
        else:
            st.warning(f"⏸ Brak jednoznacznego sygnału (RSI={last['RSI_14']:.2f})")

    with tab3:
        model = train_model(df)
        features = ["SMA_50", "SMA_200", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9"]
        pred = model.predict(df[features].tail(1))[0]
        if pred == 1:
            st.success("🤖 AI przewiduje WZROST 🚀")
        else:
            st.error("🤖 AI przewiduje SPADEK 📉")
