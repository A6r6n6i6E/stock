import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ------------------------------
# KONFIGURACJA STRONY
# ------------------------------
st.set_page_config(layout="wide", page_title="WIG20 AI Trader", page_icon="📈")
st.markdown("""
<style>
@media (max-width: 768px) {
    .stPlotlyChart { height: 70vh !important; }
    .stButton button { font-size: 18px; padding: 12px 24px; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# LISTA SPÓŁEK
# ------------------------------
wig20_dict = {
    "ALIOR": "ALR.WA", "ALLEGRO": "ALE.WA", "BUDIMEX": "BDX.WA", "CCC": "CCC.WA",
    "CDPROJEKT": "CDR.WA", "DINOPL": "DNP.WA", "KETY": "KTY.WA", "KGHM": "KGH.WA",
    "KRUK": "KRK.WA", "LPP": "LPP.WA", "MBANK": "MBK.WA", "ORANGEPL": "OPL.WA",
    "PEKAO": "PEO.WA", "PEPCO": "PCO.WA", "PGE": "PGE.WA", "PKNORLEN": "PKN.WA",
    "PKOBP": "PKO.WA", "PZU": "PZU.WA", "SANPL": "SAN.WA", "ZABKA": "ZAB.WA"
}

ticker = st.sidebar.selectbox("📊 Wybierz spółkę z WIG20", list(wig20_dict.keys()))
symbol = wig20_dict[ticker]
period = st.sidebar.selectbox("⏳ Zakres czasowy", ["5d", "1mo", "3mo", "6mo", "1y", "2y"])
interval = st.sidebar.selectbox("⏱ Interwał", ["1d", "1h"])
num_sessions = st.sidebar.number_input("📅 Ile ostatnich sesji wyświetlić?", min_value=10, max_value=500, value=50)

# ------------------------------
# FUNKCJE
# ------------------------------
def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col if i]).upper() for col in df.columns]
    else:
        df.columns = [str(col).upper() for col in df.columns]
    return df

@st.cache_data
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        return df
    df = flatten_columns(df)

    # Wybór kolumny CLOSE
    close_col = [c for c in df.columns if c.startswith("CLOSE")][0]

    # Wskaźniki
    df.ta.sma(close=close_col, length=50, append=True)
    df.ta.sma(close=close_col, length=200, append=True)
    df.ta.rsi(close=close_col, length=14, append=True)
    df.ta.macd(close=close_col, append=True)

    return df

def find_crossovers(short_sma, long_sma):
    signals = pd.Series(0, index=short_sma.index)
    signals[(short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1))] = 1
    signals[(short_sma < long_sma) & (short_sma.shift(1) >= long_sma.shift(1))] = -1
    return signals

def find_ohlc_columns(df):
    cols = {}
    for prefix in ['OPEN', 'HIGH', 'LOW', 'CLOSE']:
        matches = [col for col in df.columns if col.startswith(prefix)]
        cols[prefix] = matches[0] if matches else None
    return cols

# ------------------------------
# WCZYTANIE DANYCH
# ------------------------------
df = load_data(symbol, period, interval)

if df.empty:
    st.warning("Brak danych dla wybranego zakresu i interwału.")
    st.stop()

df = df.tail(num_sessions)
cols = find_ohlc_columns(df)
if None in cols.values():
    st.error(f"Brak wymaganych kolumn OHLC: {cols}")
    st.stop()

# ------------------------------
# WYSZUKIWANIE NAZW WSKAŹNIKÓW
# ------------------------------
sma_50_col = [c for c in df.columns if "SMA_50" in c][0]
sma_200_col = [c for c in df.columns if "SMA_200" in c][0]
rsi_col = [c for c in df.columns if "RSI_14" in c][0]
macd_col = [c for c in df.columns if "MACD_12_26_9" in c][0]
macd_signal_col = [c for c in df.columns if "MACDs_12_26_9" in c][0]

# ------------------------------
# SYGNAŁY ZŁOŻONE (SMA + RSI + MACD)
# ------------------------------
signals = find_crossovers(df[sma_50_col], df[sma_200_col])
rsi_buy = df[rsi_col] < 30
rsi_sell = df[rsi_col] > 70
macd_buy = df[macd_col] > df[macd_signal_col]
macd_sell = df[macd_col] < df[macd_signal_col]

final_buy = (signals == 1) & rsi_buy & macd_buy
final_sell = (signals == -1) & rsi_sell & macd_sell

buy_signals = df[final_buy]
sell_signals = df[final_sell]

# ------------------------------
# OSTATNI SYGNAŁ
# ------------------------------
if not buy_signals.empty and (sell_signals.empty or buy_signals.index[-1] > sell_signals.index[-1]):
    st.success(f"🟢 Ostatni sygnał KUPNA: {buy_signals.index[-1].strftime('%Y-%m-%d')} przy cenie {buy_signals[cols['CLOSE']].iloc[-1]:.2f} zł")
elif not sell_signals.empty:
    st.error(f"🔴 Ostatni sygnał SPRZEDAŻY: {sell_signals.index[-1].strftime('%Y-%m-%d')} przy cenie {sell_signals[cols['CLOSE']].iloc[-1]:.2f} zł")

# ------------------------------
# WYKRES CEN
# ------------------------------
fig_price = go.Figure()
fig_price.add_trace(go.Candlestick(
    x=df.index,
    open=df[cols['OPEN']],
    high=df[cols['HIGH']],
    low=df[cols['LOW']],
    close=df[cols['CLOSE']],
    increasing_line_color='green',
    decreasing_line_color='red',
    name='Świece'
))

fig_price.add_trace(go.Scatter(x=df.index, y=df[sma_50_col], mode='lines', name='SMA 50', line=dict(color='blue', width=2)))
fig_price.add_trace(go.Scatter(x=df.index, y=df[sma_200_col], mode='lines', name='SMA 200', line=dict(color='cyan', width=2)))

fig_price.add_trace(go.Scatter(
    x=buy_signals.index, y=buy_signals[cols['CLOSE']],
    mode='markers', name='Sygnał kupna',
    marker=dict(color='lime', size=14, symbol='triangle-up')
))
fig_price.add_trace(go.Scatter(
    x=sell_signals.index, y=sell_signals[cols['CLOSE']],
    mode='markers', name='Sygnał sprzedaży',
    marker=dict(color='red', size=14, symbol='triangle-down')
))

fig_price.update_layout(
    title=f"📈 Notowania: {ticker} ({symbol})",
    xaxis_title="Data", yaxis_title="Cena",
    xaxis_rangeslider_visible=False, height=600
)

# ------------------------------
# WYKRES RSI
# ------------------------------
fig_rsi = go.Figure()
rsi_data = df[rsi_col].dropna()
fig_rsi.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data, mode='lines', name='RSI 14', line=dict(color='purple')))
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.update_layout(
    title="RSI (Relative Strength Index)",
    yaxis=dict(range=[0, 100]), height=300, xaxis_rangeslider_visible=False
)

# ------------------------------
# MACHINE LEARNING — PROGNOZA
# ------------------------------
features = df[[sma_50_col, sma_200_col, rsi_col, macd_col, macd_signal_col]].dropna()
labels = (df[cols['CLOSE']].shift(-1) > df[cols['CLOSE']]).astype(int).loc[features.index]
if len(features) > 20:
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict([features.iloc[-1]])[0]
    if pred == 1:
        st.info("🤖 AI przewiduje: **Wzrost** jutro 📈")
    else:
        st.warning("🤖 AI przewiduje: **Spadek** jutro 📉")

# ------------------------------
# WYŚWIETLENIE WYKRESÓW
# ------------------------------
st.plotly_chart(fig_price, use_container_width=True)
st.plotly_chart(fig_rsi, use_container_width=True)
