# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import pandas_ta as ta
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =========================
# USTAWIENIA APLIKACJI
# =========================
st.set_page_config(page_title="WIG20 – sygnały i AI", layout="wide")

# Prosty CSS pod telefon: większe fonty, większe hitboxy
st.markdown("""
<style>
/* większe fonty na telefonie */
html, body, [class*="css"]  { font-size: 16px; }
div.block-container { padding-top: 1rem; padding-bottom: 1rem; }
.stButton>button { padding: 0.6rem 1rem; border-radius: 0.8rem; }
.stSelectbox label, .stNumberInput label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =========================
# LISTA SPÓŁEK
# =========================
wig20_dict = {
    "ALIOR": "ALR.WA",
    "ALLEGRO": "ALE.WA",
    "BUDIMEX": "BDX.WA",
    "CCC": "CCC.WA",
    "CDPROJEKT": "CDR.WA",
    "DINOPL": "DNP.WA",
    "KETY": "KTY.WA",
    "KGHM": "KGH.WA",
    "KRUK": "KRK.WA",
    "LPP": "LPP.WA",
    "MBANK": "MBK.WA",
    "ORANGEPL": "OPL.WA",
    "PEKAO": "PEO.WA",
    "PEPCO": "PCO.WA",
    "PGE": "PGE.WA",
    "PKNORLEN": "PKN.WA",
    "PKOBP": "PKO.WA",
    "PZU": "PZU.WA",
    "SANPL": "SAN.WA",
    "ZABKA": "ZAB.WA"
}

# =========================
# KONTROLKI U GÓRY (MOBILNIE)
# =========================
c1, c2 = st.columns([1.1, 1.2])
with c1:
    ticker = st.selectbox("Spółka WIG20", list(wig20_dict.keys()), index=11)  # domyślnie ORANGEPL
with c2:
    interval = st.selectbox("Interwał", ["1d", "1h"], index=0)

c3, c4, c5 = st.columns([1, 1, 1])
with c3:
    period = st.selectbox("Zakres", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
with c4:
    num_sessions = st.number_input("Ile ostatnich świec", min_value=50, max_value=2000, value=300, step=50)
with c5:
    use_ai = st.toggle("Włącz AI (RandomForest)", value=True)

symbol = wig20_dict[ticker]

# =========================
# POMOCNICZE
# =========================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col if i]).upper() for col in df.columns]
    else:
        df.columns = [str(col).upper() for col in df.columns]
    return df

def has_data(df, col):
    return col in df.columns and df[col].notna().any()

def find_crossovers(short_sma, long_sma):
    signals = pd.Series(0, index=short_sma.index)
    signals[(short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1))] = 1
    signals[(short_sma < long_sma) & (short_sma.shift(1) >= long_sma.shift(1))] = -1
    return signals

def detect_trendlines(extrema_idx, prices, tolerance=5):
    used = set()
    lines = []
    for i in range(len(extrema_idx)):
        i1 = extrema_idx[i]
        if i1 in used:
            continue
        for j in range(i+1, len(extrema_idx)):
            i2 = extrema_idx[j]
            if i2 in used:
                continue
            for k in range(j+1, len(extrema_idx)):
                i3 = extrema_idx[k]
                if i3 in used:
                    continue
                x = np.array([i1, i2, i3])
                y = prices[x]
                coeffs = np.polyfit(x, y, 1)
                y_fit = np.poly1d(coeffs)(x)
                if np.all(np.abs(y - y_fit) < tolerance):
                    used.update([i1, i2, i3])
                    lines.append((i1, i3, coeffs))
                    break
    return lines

# =========================
# DANE + INDIKATORY
# =========================
@st.cache_data(show_spinner=False)
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False)
    if df.empty:
        return df
    df = flatten_columns(df)
    # Indykatory
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    # Dodatkowe cechy
    df["RET_1"] = df["CLOSE"].pct_change()
    df["RET_5"] = df["CLOSE"].pct_change(5)
    df["VOL_5"] = df["RET_1"].rolling(5).std()
    df["VOL_20"] = df["RET_1"].rolling(20).std()
    df["MA_GAP"] = (df["CLOSE"] - df["SMA_50"]) / df["CLOSE"]
    # Etykiety do ML na podstawie zwrotu w przód (unikamy lookahead w cechach)
    df["FWD_RET_1"] = df["CLOSE"].shift(-1) / df["CLOSE"] - 1.0
    # Klasy: 1 kupno, -1 sprzedaż, 0 brak (strefa martwa)
    thr = 0.001 if interval == "1d" else 0.0005
    df["TARGET"] = 0
    df.loc[df["FWD_RET_1"] > thr, "TARGET"] = 1
    df.loc[df["FWD_RET_1"] < -thr, "TARGET"] = -1
    return df

df = load_data(symbol, period, interval)
if df.empty:
    st.warning("Brak danych dla wybranego zakresu i interwału.")
    st.stop()

df = df.tail(num_sessions).copy()

# =========================
# WYKRES CENY + SYGNAŁY HEURYSTYCZNE
# =========================
close_col, high_col, low_col, open_col = "CLOSE", "HIGH", "LOW", "OPEN"

fig_price = go.Figure()
fig_price.add_trace(go.Candlestick(
    x=df.index,
    open=df[open_col],
    high=df[high_col],
    low=df[low_col],
    close=df[close_col],
    increasing_line_color='green',
    decreasing_line_color='red',
    name='Świece'
))

# SMA
if has_data(df, 'SMA_50'):
    fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines',
                                   name='SMA 50', line=dict(width=2)))
if has_data(df, 'SMA_200'):
    fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], mode='lines',
                                   name='SMA 200', line=dict(width=2)))

# Prosta linia trendu ceny (regresja liniowa po indeksie)
z_price = np.polyfit(np.arange(len(df)), df[close_col].values, 1)
p_price = np.poly1d(z_price)
fig_price.add_trace(go.Scatter(
    x=df.index, y=p_price(np.arange(len(df))), mode='lines',
    n
