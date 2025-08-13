import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins

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
    # Upewniamy się, że Date jest datetime
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    # Upewniamy się, że Close jest float
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(df["Date"], df["Close"], c='blue', s=20)

    # Tworzymy tooltipy
    labels = []
    for i, row in df.iterrows():
        date_str = row["Date"].strftime('%Y-%m-%d') if not pd.isnull(row["Date"]) else "Brak daty"
        close_val = f"{row['Close']:.2f}" if not pd.isnull(row["Close"]) else "Brak danych"
        labels.append(f"<b>{date_str}</b><br>Cena: {close_val}")

    tooltip = plugins.PointHTMLTooltip(scatter, labels, voffset=10, hoffset=10)
    plugins.connect(fig, tooltip)

    ax.set_title("Wykres z interaktywnymi tooltipami", fontsize=14)
    ax.set_xlabel("Data")
    ax.set_ylabel("Cena zamknięcia")
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
        # Wyświetlenie w Streamlit
        html_str = mpld3.fig_to_html(fig)
        st.components.v1.html(html_str, height=600, scrolling=True)
