import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
import yfinance as yf
import pandas_ta as ta


# Funkcja do pobierania danych giełdowych
@st.cache_data
def load_data(symbol="AAPL", period="1mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df


# Funkcja rysująca wykres z tooltipami
def plot_with_tooltips(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Wykres ceny zamknięcia
    line, = ax.plot(df["Date"], df["Close"], marker="o", linestyle="-", color="b", label="Close")

    # Przygotowanie etykiet tooltipów
    labels = []
    for i, row in df.iterrows():
        labels.append(f"<b>{row['Date'].strftime('%Y-%m-%d')}</b><br>Cena: {row['Close']:.2f}")

    # Dodanie tooltipów do punktów
    tooltip = plugins.PointHTMLTooltip(line, labels, voffset=10, hoffset=10)
    plugins.connect(fig, tooltip)

    ax.set_title("Wykres z interaktywnymi tooltipami", fontsize=14)
    ax.set_xlabel("Data")
    ax.set_ylabel("Cena zamknięcia [USD]")
    ax.legend()

    # Zwrócenie HTML z mpld3
    return mpld3.fig_to_html(fig)


# Główna część aplikacji Streamlit
st.set_page_config(page_title="Interaktywne Tooltipy", layout="wide")

st.title("📈 Wykres z interaktywnymi tooltipami (matplotlib + mpld3)")

# Wybór spółki
symbol = st.text_input("Podaj symbol spółki:", value="AAPL")

# Pobranie danych
df = load_data(symbol)

# Generowanie wykresu i wstawienie do Streamlit
html_chart = plot_with_tooltips(df)
st.components.v1.html(html_chart, height=600, scrolling=False)
