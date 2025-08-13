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
    name='Linia trendu', line=dict(dash='dash')
))

# Sygnały SMA cross z filtrem RSI (czytelne wejście/wyjście)
signals_sma = pd.Series(0, index=df.index)
if has_data(df, 'SMA_50') and has_data(df, 'SMA_200'):
    raw = find_crossovers(df['SMA_50'], df['SMA_200'])
    # Filtry: kupno tylko gdy RSI<70 i close > SMA200; sprzedaż gdy RSI>30 i close < SMA200
    rsi = df["RSI_14"]
    cond_buy = (raw == 1) & (rsi < 70) & (df[close_col] > df["SMA_200"])
    cond_sell = (raw == -1) & (rsi > 30) & (df[close_col] < df["SMA_200"])
    signals_sma[cond_buy] = 1
    signals_sma[cond_sell] = -1

    buy_points = df.loc[signals_sma == 1]
    sell_points = df.loc[signals_sma == -1]

    fig_price.add_trace(go.Scatter(
        x=buy_points.index, y=buy_points[close_col], mode='markers',
        name='Kupno (SMA filtrowane)', marker=dict(size=12, symbol='triangle-up', color='green')
    ))
    fig_price.add_trace(go.Scatter(
        x=sell_points.index, y=sell_points[close_col], mode='markers',
        name='Sprzedaż (SMA filtrowane)', marker=dict(size=12, symbol='triangle-down', color='red')
    ))

# Trendlines (opcjonalnie – na końcu, lekkie)
close_prices = df[close_col].values
local_max_idx = argrelextrema(close_prices, np.greater, order=3)[0]
local_min_idx = argrelextrema(close_prices, np.less, order=3)[0]
max_lines = detect_trendlines(local_max_idx, close_prices, tolerance=np.std(close_prices)*0.05)
min_lines = detect_trendlines(local_min_idx, close_prices, tolerance=np.std(close_prices)*0.05)

for (i1, i3, coeffs) in max_lines + min_lines:
    x_line = np.arange(i1, i3 + 1)
    y_line = np.poly1d(coeffs)(x_line)
    fig_price.add_trace(go.Scatter(
        x=df.index[x_line], y=y_line, mode='lines',
        name='Trendline', line=dict(dash='dot', width=1)
    ))

fig_price.update_layout(
    title=f"{ticker} ({symbol}) – wykres z sygnałami",
    xaxis_title="Data",
    yaxis_title="Cena",
    xaxis_rangeslider_visible=False,
    height=520,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

# =========================
# WYKRES RSI
# =========================
fig_rsi = go.Figure()
if has_data(df, 'RSI_14'):
    rsi_data = df['RSI_14'].dropna()
    fig_rsi.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data, mode='lines', name='RSI 14'))
    fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="30 (wyprzedanie)", annotation_position="bottom right")
    fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="70 (wykupienie)", annotation_position="top right")
fig_rsi.update_layout(
    title="RSI (Relative Strength Index)",
    yaxis=dict(range=[0, 100]),
    height=240,
    xaxis_rangeslider_visible=False,
    margin=dict(t=40, b=20)
)

# =========================
# AI / ML – RandomForest
# =========================
AI_METRICS = {}

@st.cache_resource(show_spinner=False)
def train_model(train_df: pd.DataFrame, seed: int = 42):
    feats = [
        'SMA_50', 'SMA_200', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'RET_1', 'RET_5', 'VOL_5', 'VOL_20', 'MA_GAP'
    ]
    d = train_df.dropna(subset=feats + ['TARGET']).copy()
    if len(d) < 200:
        return None, None, None, None
    # Split czasowy 80/20
    split = int(len(d) * 0.8)
    X_train, y_train = d[feats].iloc[:split], d['TARGET'].iloc[:split]
    X_test, y_test = d[feats].iloc[split:], d['TARGET'].iloc[split:]

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    metrics = {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}
    return model, feats, d, metrics

model, features, model_df, metrics = (None, None, None, None)
if use_ai:
    model, features, model_df, metrics = train_model(df)
    if model is None:
        st.info("Za mało danych do trenowania modelu AI (min ~200 rekordów po NaN drop).")
        use_ai = False

# Predykcja na ostatnią świecę
ai_signal_series = pd.Series(0, index=df.index, name="AI_SIGNAL")
if use_ai and features:
    last_row = df[features].tail(1)
    if not last_row.isna().any(axis=1).iloc[0]:
        pred = int(model.predict(last_row)[0])  # -1/0/1
        ai_signal_series.iloc[-1] = pred

# =========================
# TABY
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Wykres", "🧭 Sygnały", "🤖 AI", "📊 Backtest", "⚙️ Ustawienia"])

with tab1:
    st.plotly_chart(fig_price, use_container_width=True)
    st.plotly_chart(fig_rsi, use_container_width=True)

with tab2:
    # Panel sygnałów – czytelny status
    latest = df.iloc[-1]
    rsi_val = latest.get("RSI_14", np.nan)
    trend = "Wzrostowy" if has_data(df, 'SMA_50') and has_data(df, 'SMA_200') and latest["SMA_50"] >= latest["SMA_200"] else "Spadkowy"
    last_sma_sig = "Brak"
    if signals_sma.iloc[-1] == 1:
        last_sma_sig = "KUPNO ✅"
    elif signals_sma.iloc[-1] == -1:
        last_sma_sig = "SPRZEDAŻ ❌"

    st.subheader("Status rynku")
    cA, cB, cC = st.columns(3)
    cA.metric("Trend (SMA50/200)", trend)
    cB.metric("RSI (14)", f"{rsi_val:.1f}" if pd.notna(rsi_val) else "—")
    cC.metric("Sygnał SMA (filtrowany)", last_sma_sig)

    if use_ai:
        label = {1: "KUPNO ✅", 0: "BRAK", -1: "SPRZEDAŻ ❌"}[ai_signal_series.iloc[-1]]
        st.success(f"AI sygnał (następna świeca): **{label}**")
    else:
        st.info("AI wyłączone (przejdź do zakładki 🤖 AI aby włączyć).")

with tab3:
    st.subheader("Model AI – RandomForest (klasyfikacja kierunku)")
    if use_ai and metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        m2.metric("Precision (macro)", f"{metrics['precision']*100:.1f}%")
        m3.metric("Recall (macro)", f"{metrics['recall']*100:.1f}%")
        m4.metric("F1 (macro)", f"{metrics['f1']*100:.1f}%")

        # Predykcje na całej serii (rolling out-of-sample jest bardziej złożony;
        # tutaj pokazujemy 'ex-post' na dłuższym fragmencie testowym dla poglądu)
        d = model_df.copy()
        split = int(len(d) * 0.8)
        d.loc[d.index[split:], "AI_PRED"] = model.predict(d[features].iloc[split:])
        vis = d.tail(min(400, len(d)))
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=vis.index, y=vis["CLOSE"], mode="lines", name="Close"))
        buys = vis[vis["AI_PRED"] == 1]
        sells = vis[vis["AI_PRED"] == -1]
        fig_ai.add_trace(go.Scatter(x=buys.index, y=buys["CLOSE"], mode='markers',
                                    name='AI: Kupno', marker=dict(symbol='triangle-up', size=10)))
        fig_ai.add_trace(go.Scatter(x=sells.index, y=sells["CLOSE"], mode='markers',
                                    name='AI: Sprzedaż', marker=dict(symbol='triangle-down', size=10)))
        fig_ai.update_layout(title="Predykcje AI (część testowa)", height=420, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_ai, use_container_width=True)
    else:
        st.info("Model AI nie jest aktywny lub zbyt mało danych.")

with tab4:
    st.subheader("Backtest – prosta strategia")
    st.caption("Strategia A: SMA50/200 z filtrem RSI i SMA200 (jak na wykresie). Strategia B: AI (predykcja kolejnej świecy).")
    initial_capital = 10000.0
    fee = 0.0005  # prowizja 0.05%

    # --- Strategia A: SMA filtrowane ---
    pos = 0
    cash = initial_capital
    shares = 0.0
    equity_series_A = []
    last_price = None

    for i, row in df.iterrows():
        price = row[close_col]
        sig = signals_sma.loc[i]
        # Wyjście / Wejście (long/flat)
        if sig == -1 and pos == 1:
            cash = cash + shares * price * (1 - fee)
            shares = 0.0
            pos = 0
        elif sig == 1 and pos == 0:
            shares = (cash * (1 - fee)) / price
            cash = 0.0
            pos = 1
        # Wycena
        equity = cash + shares * price if pos == 1 else cash
        equity_series_A.append((i, equity))
        last_price = price

    if not equity_series_A:
        st.warning("Za mało danych do backtestu.")
    else:
        equity_A = pd.DataFrame(equity_series_A, columns=["date", "equity"]).set_index("date")

        # --- Strategia B: AI ---
        equity_series_B = []
        pos = 0; cash = initial_capital; shares = 0.0

        if use_ai and features:
            # Używamy predykcji na całej serii ex-post (dla podglądu),
            # opcjonalnie można wdrożyć walk-forward.
            tmp = df.copy()
            # Prosta zasada: pred=1 -> long, pred=-1 -> flat (zamknięcie), pred=0 -> brak zmian
            # Aby uniknąć lookahead: przesuwamy predykcję o 1 świecę do przodu
            valid = tmp[features].dropna().index
            preds = pd.Series(index=tmp.index, dtype=float)
            # model trenowany na df -> używamy predykcji tylko tam gdzie brak NaN
            preds.loc[valid] = model.predict(tmp.loc[valid, features]) if use_ai else np.nan
            preds = preds.shift(1)  # zastosuj od kolejnej świecy
            tmp["AI_TRADE"] = preds.fillna(0)

            for i, row in tmp.iterrows():
                price = row[close_col]
                sig = row["AI_TRADE"]
                # Wyjście / Wejście (long/flat)
                if sig == -1 and pos == 1:
                    cash = cash + shares * price * (1 - fee)
                    shares = 0.0
                    pos = 0
                elif sig == 1 and pos == 0:
                    shares = (cash * (1 - fee)) / price
                    cash = 0.0
                    pos = 1
                equity = cash + shares * price if pos == 1 else cash
                equity_series_B.append((i, equity))

            equity_B = pd.DataFrame(equity_series_B, columns=["date", "equity"]).set_index("date")
        else:
            equity_B = pd.DataFrame(index=df.index, data={"equity": np.nan})

        # Buy&Hold do porównania
        bh_equity = initial_capital * (df[close_col] / df[close_col].iloc[0])

        # Wykres
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=equity_A.index, y=equity_A["equity"], mode="lines", name="Strategia A: SMA filtrowane"))
        if equity_B["equity"].notna().any():
            fig_bt.add_trace(go.Scatter(x=equity_B.index, y=equity_B["equity"], mode="lines", name="Strategia B: AI"))
        fig_bt.add_trace(go.Scatter(x=df.index, y=bh_equity, mode="lines", name="Buy & Hold"))
        fig_bt.update_layout(title="Krzywe kapitału (symulacja)", height=420, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_bt, use_container_width=True)

        colA, colB, colC = st.columns(3)
        colA.metric("Wynik A (SMA)", f"{(equity_A['equity'].iloc[-1]/initial_capital-1)*100:.1f}%")
        if equity_B["equity"].notna().any():
            colB.metric("Wynik B (AI)", f"{(equity_B['equity'].iloc[-1]/initial_capital-1)*100:.1f}%")
        colC.metric("Buy & Hold", f"{(bh_equity.iloc[-1]/initial_capital-1)*100:.1f}%")

with tab5:
    st.subheader("Ustawienia i pomoc")
    st.markdown("""
- **Sygnały SMA** – to przecięcia SMA50/SMA200 dodatkowo filtrowane przez RSI oraz pozycję ceny względem SMA200, aby zredukować fałszywe strzały.
- **AI (RandomForest)** – model klasyfikuje kierunek zmiany kolejnej świecy na podstawie pakietu cech (SMA/RSI/MACD/stopy zwrotu/zmienność).
- **Backtest** – to prosta symulacja long/flat z prowizją 0.05%. Dla AI predykcje są przesunięte o 1 świecę, aby uniknąć look-ahead w sygnałach.
- **Wskazówka mobilna** – przewijaj poziomo legendę, a interfejs działa najlepiej w trybie pionowym.
""")
    st.code("pip install streamlit yfinance pandas numpy plotly pandas-ta scikit-learn", language="bash")
