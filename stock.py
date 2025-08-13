# app.py
# ============================================
# WIG20 AI Trader — mobilna aplikacja inwestycyjna (Streamlit)
# ============================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ------------------------------
# KONFIGURACJA STRONY + STYL
# ------------------------------
st.set_page_config(
    page_title="WIG20 AI Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MOBILE_CSS = """
<style>
/* większe klikacze na mobile */
@media (max-width: 820px) {
  .stPlotlyChart { height: 68vh !important; }
  .stButton button, .stDownloadButton button {
    font-size: 18px !important; padding: 14px 22px !important; border-radius: 12px !important;
  }
  .stSelectbox div[data-baseweb="select"] { font-size: 18px !important; }
  .block-container { padding-top: 0.6rem; }
}
/* status pills */
.pill { display:inline-block; padding:6px 12px; border-radius:999px; font-weight:600; }
.pill-buy { background:#10b98122; color:#059669; border:1px solid #10b98155;}
.pill-sell{ background:#ef444422; color:#b91c1c; border:1px solid #ef444455;}
.pill-neutral{ background:#6b728022; color:#374151; border:1px solid #6b728055;}
/* nagłówek sekcji */
h3 span.kicker { font-size:0.85rem; font-weight:600; opacity:0.7; margin-left:.25rem;}
/* tabele kompaktowe */
table td, table th { padding: 6px 8px !important; }
</style>
"""
st.markdown(MOBILE_CSS, unsafe_allow_html=True)

# ------------------------------
# SŁOWNIK WIG20
# ------------------------------
wig20_dict = {
    "ALIOR": "ALR.WA", "ALLEGRO": "ALE.WA", "BUDIMEX": "BDX.WA", "CCC": "CCC.WA",
    "CDPROJEKT": "CDR.WA", "DINOPL": "DNP.WA", "KETY": "KTY.WA", "KGHM": "KGH.WA",
    "KRUK": "KRK.WA", "LPP": "LPP.WA", "MBANK": "MBK.WA", "ORANGEPL": "OPL.WA",
    "PEKAO": "PEO.WA", "PEPCO": "PCO.WA", "PGE": "PGE.WA", "PKNORLEN": "PKN.WA",
    "PKOBP": "PKO.WA", "PZU": "PZU.WA", "SANPL": "SAN.WA", "ZABKA": "ZAB.WA"
}

# ------------------------------
# SIDEBAR (ustawienia)
# ------------------------------
with st.sidebar:
    st.title("⚙️ Ustawienia")
    ticker = st.selectbox("Spółka WIG20", list(wig20_dict.keys()), index=15)  # PKN domyślnie
    symbol = wig20_dict[ticker]
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Zakres", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    with col2:
        interval = st.selectbox("Interwał", ["1d", "1h"], index=0)
    num_rows = st.slider("Ile ostatnich świec", 50, 600, 200, step=10)
    st.caption("Wskazówki: RSI(14), SMA(50/200), MACD(12,26,9) • Sygnały: SMA×RSI×MACD")

# ------------------------------
# FUNKCJE POMOCNICZE
# ------------------------------
def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ujednolica nazwy kolumn (UPPER + bez spacji)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col if i]).upper().replace(" ", "_") for col in df.columns]
    else:
        df.columns = [str(c).upper().replace(" ", "_") for c in df.columns]
    return df

@st.cache_data(ttl=3600)
def load_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = _std_cols(df.reset_index())
    # Gwarantujemy CLOSE:
    if "CLOSE" not in df.columns:
        # yfinance zawsze zwraca Close, ale dla pewności:
        close_candidates = [c for c in df.columns if "CLOSE" in c]
        if not close_candidates:
            return pd.DataFrame()
        df["CLOSE"] = df[close_candidates[0]]
    # Indykatory na jawnie wskazanym CLOSE -> stałe nazwy kolumn!
    df.ta.sma(close=df["CLOSE"], length=50, append=True)     # SMA_50
    df.ta.sma(close=df["CLOSE"], length=200, append=True)    # SMA_200
    df.ta.rsi(close=df["CLOSE"], length=14, append=True)     # RSI_14
    df.ta.macd(close=df["CLOSE"], append=True)               # MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    return df

def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Zwraca DataFrame z kolumną 'SIGNAL' (BUY/SELL/NEUTRAL) i pomocniczymi sygnałami."""
    req = {"SMA_50", "SMA_200", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9", "CLOSE"}
    if not req.issubset(set(df.columns)):
        return pd.DataFrame()

    # Crossover SMA50/200
    sma_cross_up = (df["SMA_50"] > df["SMA_200"]) & (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))
    sma_cross_dn = (df["SMA_50"] < df["SMA_200"]) & (df["SMA_50"].shift(1) >= df["SMA_200"].shift(1))

    # Filtry RSI i MACD
    rsi_buy = df["RSI_14"] < 30
    rsi_sell = df["RSI_14"] > 70
    macd_buy = df["MACD_12_26_9"] > df["MACDs_12_26_9"]
    macd_sell = df["MACD_12_26_9"] < df["MACDs_12_26_9"]

    final_buy = (sma_cross_up & rsi_buy & macd_buy)
    final_sell = (sma_cross_dn & rsi_sell & macd_sell)

    out = df.copy()
    out["BUY"] = final_buy.astype(int)
    out["SELL"] = final_sell.astype(int)
    out["SIGNAL"] = np.where(final_buy, "BUY", np.where(final_sell, "SELL", "NEUTRAL"))
    return out

def make_candles(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["DATE"] if "DATE" in df.columns else df.index,
        open=df["OPEN"], high=df["HIGH"], low=df["LOW"], close=df["CLOSE"],
        name="Świece",
        increasing_line_color="green",
        decreasing_line_color="red",
    ))
    if "SMA_50" in df and "SMA_200" in df:
        fig.add_trace(go.Scatter(x=df["DATE"], y=df["SMA_50"], mode="lines", name="SMA 50",
                                 line=dict(width=2)))
        fig.add_trace(go.Scatter(x=df["DATE"], y=df["SMA_200"], mode="lines", name="SMA 200",
                                 line=dict(width=2)))
    fig.update_layout(
        title=title, xaxis_title="Data", yaxis_title="Cena",
        xaxis_rangeslider_visible=False, height=600, margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig

def add_markers(fig: go.Figure, df_sig: pd.DataFrame):
    buy_pts = df_sig[df_sig["BUY"] == 1]
    sell_pts = df_sig[df_sig["SELL"] == 1]
    if not buy_pts.empty:
        fig.add_trace(go.Scatter(
            x=buy_pts["DATE"], y=buy_pts["CLOSE"],
            mode="markers", name="Kupno",
            marker=dict(symbol="triangle-up", size=14)
        ))
    if not sell_pts.empty:
        fig.add_trace(go.Scatter(
            x=sell_pts["DATE"], y=sell_pts["CLOSE"],
            mode="markers", name="Sprzedaż",
            marker=dict(symbol="triangle-down", size=14)
        ))

def last_signal_badge(df_sig: pd.DataFrame):
    last_row = df_sig.iloc[-1]
    lab = last_row["SIGNAL"]
    price = last_row["CLOSE"]
    date = pd.to_datetime(last_row["DATE"]).date() if "DATE" in df_sig else df_sig.index[-1].date()
    if lab == "BUY":
        st.markdown(f'<span class="pill pill-buy">🟢 KUPNO</span> &nbsp; {date} @ {price:.2f} zł', unsafe_allow_html=True)
    elif lab == "SELL":
        st.markdown(f'<span class="pill pill-sell">🔴 SPRZEDAŻ</span> &nbsp; {date} @ {price:.2f} zł', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="pill pill-neutral">⚪ NEUTRAL</span> &nbsp; {date} @ {price:.2f} zł', unsafe_allow_html=True)

def quick_backtest(df_sig: pd.DataFrame) -> dict:
    """Bardzo prosty backtest: BUY -> long do SELL; SELL -> cash. Brak kosztów/poślizgu."""
    if df_sig.empty:
        return {"trades": 0, "return": 0.0, "buy_hold": 0.0}
    prices = df_sig["CLOSE"].values
    signals = df_sig["SIGNAL"].values
    pos = 0  # 1 long, 0 cash
    entry_price = None
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    trades = 0

    for i in range(1, len(prices)):
        if signals[i-1] == "BUY" and pos == 0:
            pos = 1
            entry_price = prices[i]
            trades += 1
        elif signals[i-1] == "SELL" and pos == 1:
            # zamknięcie long
            ret = prices[i] / entry_price
            equity *= ret
            pos = 0
            entry_price = None
        # equity trail (jeśli otwarta pozycja, mark-to-market)
        if pos == 1 and entry_price:
            mtm = (prices[i] / entry_price)
            eq_inst = equity * mtm
        else:
            eq_inst = equity
        peak = max(peak, eq_inst)
        max_dd = min(max_dd, (eq_inst / peak) - 1.0)

    # domknięcie pozycji na końcu:
    if pos == 1 and entry_price:
        equity *= prices[-1] / entry_price

    bh = prices[-1] / prices[0]  # buy&hold
    return {
        "trades": trades,
        "return": equity - 1.0,
        "buy_hold": bh - 1.0,
        "max_drawdown": max_dd
    }

def ai_forecast(df: pd.DataFrame) -> dict:
    """RandomForest: prognoza, pewność, trafność testowa, ważność cech."""
    need = ["SMA_50", "SMA_200", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9", "CLOSE"]
    if not set(need).issubset(df.columns):
        return {}
    features = df[["SMA_50", "SMA_200", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9"]].copy()
    # cel: wzrost jutro?
    target = (df["CLOSE"].shift(-1) > df["CLOSE"]).astype(int)
    data = pd.concat([features, target.rename("TARGET")], axis=1).dropna()
    if len(data) < 80:
        return {}

    X = data.drop(columns=["TARGET"])
    y = data["TARGET"]

    # nie mieszamy — zachowujemy kolejność czasową
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    acc = float((model.predict(X_test) == y_test).mean())
    last_x = X.iloc[[-1]]
    proba = model.predict_proba(last_x)[0][1]
    pred = int(proba >= 0.5)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {"pred": pred, "proba": proba, "acc": acc, "importances": importances}

# ------------------------------
# DANE + OBRÓBKA
# ------------------------------
df_raw = load_ohlc(wig20_dict[ticker], period, interval)
if df_raw.empty:
    st.error("❌ Brak danych (ticker/zakres/interwał). Spróbuj innej konfiguracji.")
    st.stop()

df = df_raw.tail(max(num_rows, 60)).copy()
df_sig = build_signals(df)
if df_sig.empty:
    st.error("❌ Nie udało się obliczyć wskaźników/sygnałów.")
    st.stop()

# ------------------------------
# NAGŁÓWEK
# ------------------------------
left, right = st.columns([0.7, 0.3])
with left:
    st.subheader(f"📊 {ticker}  ({wig20_dict[ticker]})")
    last_signal_badge(df_sig)
with right:
    st.write("")
    st.caption("Informacyjnie — to nie jest porada inwestycyjna.")

# ------------------------------
# ZAKŁADKI
# ------------------------------
tab_chart, tab_signals, tab_ai, tab_screener = st.tabs(["📈 Wykres", "🔔 Sygnały", "🤖 AI", "🧭 Skaner WIG20"])

# --- Wykres ---
with tab_chart:
    show_markers = st.toggle("Pokaż markery sygnałów na wykresie", value=True)
    fig = make_candles(df, f"Notowania {ticker}")
    if show_markers:
        add_markers(fig, df_sig)
    st.plotly_chart(fig, use_container_width=True)

    # RSI wykres
    if "RSI_14" in df:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df["DATE"], y=df["RSI_14"], mode="lines", name="RSI 14"))
        fig_rsi.add_hline(y=30, line_dash="dash")
        fig_rsi.add_hline(y=70, line_dash="dash")
        fig_rsi.update_layout(title="RSI (14)", height=260, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig_rsi, use_container_width=True)

    # szybki backtest
    bt = quick_backtest(df_sig)
    colb1, colb2, colb3, colb4 = st.columns(4)
    colb1.metric("Transakcji", bt["trades"])
    colb2.metric("Stopa zwrotu strategii", f"{bt['return']*100:0.1f}%")
    colb3.metric("Buy&Hold (ten sam okres)", f"{bt['buy_hold']*100:0.1f}%")
    colb4.metric("Max Drawdown", f"{bt['max_drawdown']*100:0.1f}%")

# --- Sygnały ---
with tab_signals:
    st.markdown("### Ostatnie sygnały")
    sig_rows = df_sig[df_sig["SIGNAL"].isin(["BUY", "SELL"])].copy()
    sig_rows = sig_rows[["DATE", "CLOSE", "SIGNAL"]].tail(20)
    if sig_rows.empty:
        st.info("Brak świeżych sygnałów w wybranym oknie.")
    else:
        # kolorowe znaczniki
        sig_rows["STATUS"] = sig_rows["SIGNAL"].map({
            "BUY": "🟢 BUY",
            "SELL": "🔴 SELL",
            "NEUTRAL": "⚪ NEUTRAL"
        })
        sig_rows = sig_rows[["DATE", "CLOSE", "STATUS"]].rename(columns={"DATE":"Data", "CLOSE":"Kurs", "STATUS":"Sygnał"})
        st.dataframe(sig_rows, use_container_width=True, hide_index=True)

# --- AI ---
with tab_ai:
    st.markdown("### Prognoza AI na jutro")
    res = ai_forecast(df)
    if not res:
        st.warning("Za mało danych do wiarygodnej prognozy AI. Zwiększ zakres lub zmień interwał.")
    else:
        pred_txt = "📈 **Wzrost**" if res["pred"] == 1 else "📉 **Spadek**"
        colA, colB, colC = st.columns(3)
        colA.metric("Prognoza", pred_txt)
        colB.metric("Pewność modelu", f"{res['proba']*100:0.1f}%")
        colC.metric("Trafność (test)", f"{res['acc']*100:0.1f}%")

        st.markdown("#### Ważność cech")
        imp = res["importances"].reset_index()
        imp.columns = ["Cecha", "Ważność"]
        fig_imp = go.Figure()
        fig_imp.add_bar(x=imp["Cecha"], y=imp["Ważność"])
        fig_imp.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

        st.caption("Model: RandomForestClassifier (bez strojenia hiperparametrów). Cel: wzrost ceny kolejnego dnia.")

# --- Skaner WIG20 ---
with tab_screener:
    st.markdown("### Skaner sygnałów — WIG20")
    scan_period = st.selectbox("Zakres dla skanera", ["3mo", "6mo", "1y"], index=0, key="scan_period")
    results = []
    for name, sym in wig20_dict.items():
        try:
            d0 = load_ohlc(sym, scan_period, "1d").tail(120)
            if d0.empty:
                results.append((name, sym, "NO DATA", np.nan, np.nan))
                continue
            ds = build_signals(d0)
            if ds.empty:
                results.append((name, sym, "NO INDICATORS", np.nan, np.nan))
                continue
            last = ds.iloc[-1]
            results.append((name, sym, last["SIGNAL"], float(last["CLOSE"]), float(last["RSI_14"])))
        except Exception:
            results.append((name, sym, "ERROR", np.nan, np.nan))

    scan_df = pd.DataFrame(results, columns=["Spółka", "Ticker", "Sygnał", "Kurs", "RSI_14"])
    # sortuj BUY > NEUTRAL > SELL i po RSI
    cat = pd.CategoricalDtype(categories=["BUY", "NEUTRAL", "SELL", "NO DATA", "NO INDICATORS", "ERROR"], ordered=True)
    scan_df["Sygnał"] = scan_df["Sygnał"].astype(cat)
    scan_df = scan_df.sort_values(["Sygnał", "RSI_14"], ascending=[True, True])

    def color_signal(val: str) -> str:
        if val == "BUY": return "🟢 BUY"
        if val == "SELL": return "🔴 SELL"
        if val == "NEUTRAL": return "⚪ NEUTRAL"
        return f"⚠️ {val}"

    scan_df["Sygnał"] = scan_df["Sygnał"].astype(str).map(color_signal)
    st.dataframe(scan_df, use_container_width=True, hide_index=True)

# ------------------------------
# STOPKA / DISCLAIMER
# ------------------------------
st.markdown("---")
st.caption("⚠️ To narzędzie ma charakter wyłącznie informacyjny i edukacyjny. "
           "Nie stanowi rekomendacji inwestycyjnej. Inwestowanie na rynkach finansowych wiąże się z ryzykiem.")
