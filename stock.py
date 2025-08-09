import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import pandas_ta as ta
from scipy.signal import argrelextrema

st.set_page_config(layout="wide")

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

ticker = st.sidebar.selectbox("Wybierz spółkę z WIG20", list(wig20_dict.keys()))
symbol = wig20_dict[ticker]

period = st.sidebar.selectbox("Zakres czasowy", ["5d", "1mo", "3mo", "6mo", "1y", "2y"])
interval = st.sidebar.selectbox("Interwał", ["1d", "1h"])
num_sessions = st.sidebar.number_input("Ile ostatnich sesji wyświetlić?", min_value=10, max_value=500, value=50)

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
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    return df

def has_data(df, col):
    return col in df.columns and df[col].notna().any()

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

def detect_trendlines(extrema_idx, prices, kind='high', tolerance=5):
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

df = load_data(symbol, period, interval)

if df.empty:
    st.warning("Brak danych dla wybranego zakresu i interwału.")
else:
    df = df.tail(num_sessions)
    st.dataframe(df)

    cols = find_ohlc_columns(df)

    if None in cols.values():
        st.error(f"Brak wymaganych kolumn OHLC: {cols}")
        st.stop()

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

    if has_data(df, 'SMA_50'):
        fig_price.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=2)
        ))

    if has_data(df, 'SMA_200'):
        fig_price.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_200'],
            mode='lines',
            name='SMA 200',
            line=dict(color='cyan', width=2)
        ))

    if has_data(df, cols['CLOSE']):
        z_price = np.polyfit(np.arange(len(df)), df[cols['CLOSE']], 1)
        p_price = np.poly1d(z_price)
        fig_price.add_trace(go.Scatter(
            x=df.index,
            y=p_price(np.arange(len(df))),
            mode='lines',
            name='Linia trendu ceny',
            line=dict(color='purple', dash='dash')
        ))

    if has_data(df, 'SMA_50') and has_data(df, 'SMA_200'):
        signals = find_crossovers(df['SMA_50'], df['SMA_200'])
        buy_signals = df[signals == 1]
        sell_signals = df[signals == -1]

        fig_price.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals[cols['CLOSE']],
            mode='markers',
            name='Sygnał kupna',
            marker=dict(color='green', size=12, symbol='triangle-up')
        ))

        fig_price.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals[cols['CLOSE']],
            mode='markers',
            name='Sygnał sprzedaży',
            marker=dict(color='red', size=12, symbol='triangle-down')
        ))

    close_prices = df[cols['CLOSE']].values
    local_max_idx = argrelextrema(close_prices, np.greater)[0]
    local_min_idx = argrelextrema(close_prices, np.less)[0]

    max_lines = detect_trendlines(local_max_idx, close_prices, kind='high')
    min_lines = detect_trendlines(local_min_idx, close_prices, kind='low')

    fig_price.update_layout(
        title=f"Notowania: {ticker} ({symbol}) z sygnałami SMA i liniami trendu",
        xaxis_title="Data",
        yaxis_title="Cena / Wskaźniki",
        xaxis_rangeslider_visible=False,
        height=600
    )

    fig_rsi = go.Figure()
    if has_data(df, 'RSI_14'):
        rsi_data = df['RSI_14'].dropna()
        fig_rsi.add_trace(go.Scatter(
            x=rsi_data.index,
            y=rsi_data,
            mode='lines',
            name='RSI 14',
            line=dict(color='purple')
        ))
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="30", annotation_position="bottom right")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="70", annotation_position="top right")

    fig_rsi.update_layout(
        title="RSI (Relative Strength Index)",
        yaxis=dict(range=[0, 100]),
        height=300,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig_price, use_container_width=True)
    st.plotly_chart(fig_rsi, use_container_width=True)

