# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
import yfinance as yf

# ---------------------------
# Ustawienia strony
# ---------------------------
st.set_page_config(page_title="Wykres z tooltipami (mpld3)", layout="wide")

st.title("📈 Interaktywny wykres z tooltipami (matplotlib + mpld3)")

# ---------------------------
# Funkcje pomocnicze
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(symbol="AAPL", period="6mo", interval="1d") -> pd.DataFrame:
    """
    Pobiera dane z Yahoo Finance i wylicza podstawowe wskaźniki.
    Zwraca ramkę o kolumnach: Date, Open, High, Low, Close, Volume, SMA20, SMA50, EMA20, RSI14
    """
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        return df

    # przenosimy DatetimeIndex do kolumny Date i upewniamy się, że to datetime64[ns]
    df = df.reset_index().rename(columns=str)
    if "Date" not in df.columns:  # (czasem nazwa bywa inna)
        # poszukajmy kolumny z datetime
        dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
        if dt_cols:
            df = df.rename(columns={dt_cols[0]: "Date"})
        else:
            # jeśli brak, spróbujmy skonwertować
            df["Date"] = pd.to_datetime(df.index)

    df["Date"] = pd.to_datetime(df["Date"])

    # Wskaźniki
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["RSI14"] = compute_rsi(df["Close"], period=14)

    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Prosta implementacja RSI (Wilders).
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Średnie kroczące wykładnicze z alpha = 1/period (przybliżenie Wildersa)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def fmt_num(x, dec=2):
    """Ładne formatowanie liczby z obsługą NaN."""
    return "—" if pd.isna(x) else f"{x:.{dec}f}"


def plot_with_tooltips(df: pd.DataFrame, title: str = "Wykres z interaktywnymi tooltipami") -> str:
    """
    Tworzy wykres matplotlib z tooltipami (mpld3) i zwraca gotowy HTML jako string.
    Rozwiązuje problem .strftime na Series – używa .dt.strftime wektorowo.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Linie na wykresie (Close + średnie)
    line_close, = ax.plot(df["Date"], df["Close"], marker="o", linestyle="-", label="Close")
    ax.plot(df["Date"], df["SMA20"], linestyle="--", label="SMA20")
    ax.plot(df["Date"], df["SMA50"], linestyle="--", label="SMA50")
    ax.plot(df["Date"], df["EMA20"], linestyle=":", label="EMA20")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Data")
    ax.set_ylabel("Cena zamknięcia")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    # --- TOOLTIPY ---
    # >>> KLUCZOWA POPRAWKA <<<
    # Zamiast row['Date'].strftime(...), korzystamy z wektorowego .dt.strftime na całej kolumnie:
    date_str = df["Date"].dt.strftime("%Y-%m-%d")

    labels = []
    for d, c, s20, s50, e20, r in zip(
            date_str, df["Close"], df["SMA20"], df["SMA50"], df["EMA20"], df["RSI14"]
    ):
        html = (
            f"<div style='font-size:12px'>"
            f"<b>{d}</b><br>"
            f"Close: {fmt_num(c, 2)}<br>"
            f"SMA20: {fmt_num(s20, 2)}<br>"
            f"SMA50: {fmt_num(s50, 2)}<br>"
            f"EMA20: {fmt_num(e20, 2)}<br>"
            f"RSI14: {fmt_num(r, 1)}"
            f"</div>"
        )
        labels.append(html)

    # Tooltipy do punktów linii Close (jeden tooltip na każdy marker)
    tooltip = plugins.PointHTMLTooltip(line_close, labels, voffset=10, hoffset=10, css=None)
    plugins.connect(fig, tooltip, plugins.Zoom(), plugins.Reset(), plugins.MousePosition(fontsize=10))

    # Zwróć gotowy HTML (mpld3)
    html = mpld3.fig_to_html(fig)
    plt.close(fig)
    return html


# ---------------------------
# UI
# ---------------------------
with st.sidebar:
    st.header("Ustawienia")
    symbol = st.text_input("Symbol", value="AAPL")
    period = st.selectbox("Okres", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Interwał", ["1d", "1h", "30m"], index=0)
    run = st.button("🔍 Pobierz i narysuj")

if run:
    df = load_data(symbol, period, interval)
    if df.empty:
        st.error("Brak danych dla podanych parametrów.")
    else:
        st.caption(f"Danych: {len(df)} wierszy | Od: {df['Date'].min().date()}  Do: {df['Date'].max().date()}")
        html_chart = plot_with_tooltips(df, title=f"{symbol} – {period}, {interval}")
        st.components.v1.html(html_chart, height=620, scrolling=False)
else:
    st.info("Ustaw parametry po lewej i kliknij przycisk, aby narysować wykres.")
