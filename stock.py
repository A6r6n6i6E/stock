import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
import matplotlib.dates as mdates

# =====================
# Funkcja pobierająca dane
# =====================
def load_data(symbol, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

# =====================
# Funkcja rysująca wykres z tooltipami
# =====================
def plot_with_tooltips(df):
    # Konwersja typów
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')

    # Usuwamy wiersze z brakami danych
    df = df.dropna(subset=["Date", "Close"])

    # Zamiana dat na format liczbowy matplotliba
    dates_num = mdates.date2num(df["Date"].tolist())  # liczby zamiast datetime
    closes = df["Close"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(dates_num, closes, c='blue', s=20)

    # Tooltipy (pokazujemy datę jako string w formacie RRRR-MM-DD)
    labels = [
        f"<b>{d.strftime('%Y-%m-%d')}</b><br>Cena: {c:.2f}"
        for d, c in zip(df["Date"], closes)
    ]

    tooltip = plugins.PointHTMLTooltip(scatter, labels, voffset=10, hoffset=10)
    plugins.connect(fig, tooltip)

    ax.set_title("Wykres z interaktywnymi tooltipami", fontsize=14)
    ax.set_xlabel("Data")
    ax.set_ylabel("Cena zamknięcia")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    return fig

# =====================
# Aplikacja Streamlit
# =====================
st.title("📈 Wykres giełdowy z interaktywnymi tooltipami (mpld3)")

symbol = st.text_input("Podaj symbol giełdowy:", "AAPL")
period = st.selectbox("Okres:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=2)
interval = st.selectbox("Interwał:", ["1d", "1wk", "1mo"], index=0)

if st.button("Pobierz dane i narysuj wykres"):
    df = load_data(symbol, period, interval)

    if df.empty:
        st.error("Brak danych dla podanego symbolu.")
    else:
        fig = plot_with_tooltips(df)
        html_str = mpld3.fig_to_html(fig)
        st.components.v1.html(html_str, height=600, scrolling=True)
