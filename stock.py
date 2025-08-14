import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import pandas_ta as ta
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja strony z lepszym layoutem mobilnym
st.set_page_config(
    page_title="📈 WIG20 Trading Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS dla lepszego wyglądu na mobile
st.markdown("""
<style>
    .stSelectbox > div > div > select {
        font-size: 16px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .buy-signal {
        color: #00C851;
        font-weight: bold;
        font-size: 18px;
    }
    .sell-signal {
        color: #FF4444;
        font-weight: bold;
        font-size: 18px;
    }
    .hold-signal {
        color: #FF8800;
        font-weight: bold;
        font-size: 18px;
    }
    .signal-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .buy-box { background-color: #d4edda; border: 2px solid #00C851; }
    .sell-box { background-color: #f8d7da; border: 2px solid #FF4444; }
    .hold-box { background-color: #fff3cd; border: 2px solid #FF8800; }
</style>
""", unsafe_allow_html=True)

# Słownik spółek WIG20
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

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col if i]).upper() for col in df.columns]
    else:
        df.columns = [str(col).upper() for col in df.columns]
    return df

def find_ohlc_columns(df):
    """Znajdź kolumny OHLC niezależnie od formatowania"""
    cols = {}
    
    # Możliwe nazwy kolumn
    possible_names = {
        'OPEN': ['OPEN', 'Open', 'open'],
        'HIGH': ['HIGH', 'High', 'high'], 
        'LOW': ['LOW', 'Low', 'low'],
        'CLOSE': ['CLOSE', 'Close', 'close'],
        'VOLUME': ['VOLUME', 'Volume', 'volume']
    }
    
    for key, possible in possible_names.items():
        found = False
        for col_name in df.columns:
            for poss in possible:
                if poss in col_name:
                    cols[key] = col_name
                    found = True
                    break
            if found:
                break
        if not found:
            cols[key] = None
    
    return cols

@st.cache_data(ttl=300)  # Cache na 5 minut
def load_data(symbol, period, interval):
    try:
        # Pobierz dane z yfinance
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return df, "Brak danych"
        
        # Debug: pokaż oryginalne kolumny
        st.write("🔍 **Debug - Oryginalne kolumny:**", df.columns.tolist())
        
        df = flatten_columns(df)
        
        # Debug: pokaż kolumny po przetworzeniu
        st.write("🔍 **Debug - Kolumny po przetworzeniu:**", df.columns.tolist())
        
        # Znajdź kolumny OHLC
        ohlc_cols = find_ohlc_columns(df)
        st.write("🔍 **Debug - Znalezione kolumny OHLC:**", ohlc_cols)
        
        # Sprawdź czy mamy podstawowe dane
        required_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        missing_cols = [col for col in required_cols if ohlc_cols.get(col) is None]
        
        if missing_cols:
            return df, f"Brak wymaganych kolumn: {missing_cols}"
        
        # Użyj znalezionych nazw kolumn
        close_col = ohlc_cols['CLOSE']
        high_col = ohlc_cols['HIGH']
        low_col = ohlc_cols['LOW']
        open_col = ohlc_cols['OPEN']
        volume_col = ohlc_cols.get('VOLUME')
        
        # Wskaźniki techniczne - używaj właściwych nazw kolumn
        try:
            # Podstawowe średnie
            df.ta.sma(close=close_col, length=20, append=True)
            df.ta.sma(close=close_col, length=50, append=True)
            df.ta.sma(close=close_col, length=200, append=True)
            df.ta.ema(close=close_col, length=12, append=True)
            df.ta.ema(close=close_col, length=26, append=True)
            
            # Bollinger Bands
            df.ta.bbands(close=close_col, length=20, std=2, append=True)
            
            # RSI
            df.ta.rsi(close=close_col, length=14, append=True)
            
            # MACD
            df.ta.macd(close=close_col, append=True)
            
            # Inne wskaźniki
            df.ta.stoch(high=high_col, low=low_col, close=close_col, append=True)
            df.ta.adx(high=high_col, low=low_col, close=close_col, append=True)
            df.ta.cci(high=high_col, low=low_col, close=close_col, append=True)
            df.ta.willr(high=high_col, low=low_col, close=close_col, append=True)
            
        except Exception as indicator_error:
            st.warning(f"Ostrzeżenie przy obliczaniu wskaźników: {indicator_error}")
        
        # Dodatkowe cechy dla ML
        try:
            df['PRICE_CHANGE'] = df[close_col].pct_change()
            
            if volume_col and volume_col in df.columns:
                df['VOLUME_MA'] = df[volume_col].rolling(20).mean()
                df['VOLUME_RATIO'] = df[volume_col] / df['VOLUME_MA']
            
            df['HIGH_LOW_PCT'] = (df[high_col] - df[low_col]) / df[close_col]
            df['CLOSE_OPEN_PCT'] = (df[close_col] - df[open_col]) / df[open_col]
            
        except Exception as feature_error:
            st.warning(f"Ostrzeżenie przy tworzeniu cech: {feature_error}")
        
        return df, None
        
    except Exception as e:
        return pd.DataFrame(), f"Błąd podczas pobierania danych: {e}"

def create_features(df):
    """Tworzenie cech dla modelu ML"""
    features = []
    
    # Lista wskaźników do wykorzystania
    indicators = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14', 'MACD_12_26_9', 
                 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ADX_14', 'CCI_14_0.015', 
                 'WILLR_14', 'VOLUME_RATIO', 'HIGH_LOW_PCT', 'CLOSE_OPEN_PCT']
    
    for indicator in indicators:
        if indicator in df.columns:
            features.append(indicator)
            # Dodaj trend wskaźnika
            df[f'{indicator}_TREND'] = np.where(df[indicator] > df[indicator].shift(1), 1, -1)
            features.append(f'{indicator}_TREND')
    
    # Relative position indicators
    close_col = None
    for col in df.columns:
        if 'CLOSE' in col.upper():
            close_col = col
            break
    
    if close_col:
        for period in [5, 10, 20]:
            df[f'CLOSE_VS_MA{period}'] = df[close_col] / df[close_col].rolling(period).mean() - 1
            features.append(f'CLOSE_VS_MA{period}')
    
    return features

def create_target(df, forward_days=5, threshold=0.02):
    """Tworzenie zmiennej docelowej - czy cena wzrośnie o threshold% w ciągu forward_days"""
    close_col = None
    for col in df.columns:
        if 'CLOSE' in col.upper():
            close_col = col
            break
    
    if close_col is None:
        return np.array([])
    
    future_return = df[close_col].shift(-forward_days) / df[close_col] - 1
    target = np.where(future_return > threshold, 2,  # Strong Buy
                     np.where(future_return < -threshold, 0, 1))  # Sell, Hold, Buy
    return target

def train_ml_model(df):
    """Trenowanie modelu ML"""
    try:
        features = create_features(df)
        target = create_target(df)
        
        if len(target) == 0:
            return None, None, [], 0
        
        # Usuń wiersze z brakującymi danymi
        feature_data = df[features].fillna(method='ffill').fillna(method='bfill')
        valid_idx = ~np.isnan(target)
        
        X = feature_data[valid_idx]
        y = target[valid_idx]
        
        if len(X) < 50:  # Za mało danych
            return None, None, [], 0
        
        # Podział danych
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Skalowanie
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Trenowanie modelu
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train_scaled, y_train)
        
        # Predykcja
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, scaler, features, accuracy
        
    except Exception as e:
        st.error(f"Błąd podczas trenowania modelu: {e}")
        return None, None, [], 0

def get_ml_signal(df, model, scaler, features):
    """Pobieranie sygnału z modelu ML"""
    try:
        if model is None or len(df) == 0:
            return "HOLD", 0.33
        
        latest_data = df[features].iloc[-1:].fillna(method='ffill').fillna(0)
        latest_scaled = scaler.transform(latest_data)
        
        prediction = model.predict(latest_scaled)[0]
        probabilities = model.predict_proba(latest_scaled)[0]
        
        signals = {0: "SELL", 1: "HOLD", 2: "BUY"}
        confidence = max(probabilities)
        
        return signals[prediction], confidence
        
    except Exception as e:
        return "HOLD", 0.33

def calculate_traditional_signals(df):
    """Obliczanie tradycyjnych sygnałów technicznych"""
    signals = {"RSI": "HOLD", "MACD": "HOLD", "SMA": "HOLD", "Bollinger": "HOLD"}
    
    if 'RSI_14' in df.columns and not df['RSI_14'].isna().all():
        latest_rsi = df['RSI_14'].iloc[-1]
        if latest_rsi < 30:
            signals["RSI"] = "BUY"
        elif latest_rsi > 70:
            signals["RSI"] = "SELL"
    
    if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
        latest_macd = df['MACD_12_26_9'].iloc[-1]
        latest_signal = df['MACDs_12_26_9'].iloc[-1]
        if latest_macd > latest_signal:
            signals["MACD"] = "BUY"
        else:
            signals["MACD"] = "SELL"
    
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
            signals["SMA"] = "BUY"
        else:
            signals["SMA"] = "SELL"
    
    # Znajdź kolumny Bollinger Bands i Close
    close_col = None
    bb_lower_col = None
    bb_upper_col = None
    
    for col in df.columns:
        if 'CLOSE' in col.upper():
            close_col = col
        elif 'BBL' in col:
            bb_lower_col = col
        elif 'BBU' in col:
            bb_upper_col = col
    
    if close_col and bb_lower_col and bb_upper_col:
        close = df[close_col].iloc[-1]
        bb_lower = df[bb_lower_col].iloc[-1]
        bb_upper = df[bb_upper_col].iloc[-1]
        
        if close <= bb_lower:
            signals["Bollinger"] = "BUY"
        elif close >= bb_upper:
            signals["Bollinger"] = "SELL"
    
    return signals

def main():
    st.title("📈 WIG20 Trading Assistant z AI")
    st.markdown("*Analiza techniczna z wykorzystaniem Machine Learning*")
    
    # Sidebar - kompaktowy dla mobile
    with st.sidebar:
        st.header("⚙️ Ustawienia")
        ticker = st.selectbox("🏢 Spółka WIG20", list(wig20_dict.keys()))
        symbol = wig20_dict[ticker]
        
        period = st.selectbox("📅 Zakres", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        interval = st.selectbox("⏱️ Interwał", ["1d", "1h"], index=0)
        num_sessions = st.slider("📊 Liczba sesji", 50, 500, 200)
        
        use_ml = st.checkbox("🤖 Włącz AI/ML", value=True)
    
    # Główna część aplikacji
    col1, col2 = st.columns([2, 1])
    
    with st.spinner(f"Ładowanie danych dla {ticker}..."):
        result = load_data(symbol, period, interval)
        if isinstance(result, tuple):
            df, error_msg = result
        else:
            df = result
            error_msg = None
    
    if error_msg:
        st.error(f"❌ {error_msg}")
        return
        
    if df.empty:
        st.error("❌ Otrzymano puste dane")
        return
    
    df = df.tail(num_sessions)
    
    # Znajdź kolumny OHLC ponownie po obcięciu danych
    ohlc_cols = find_ohlc_columns(df)
    st.write("🔍 **Debug - Finalne kolumny OHLC:**", ohlc_cols)
    
    # Sprawdź czy mamy wszystkie wymagane kolumny
    required_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    missing_cols = [k for k in required_cols if ohlc_cols.get(k) is None]
    if missing_cols:
        st.error(f"❌ Brak wymaganych kolumn: {missing_cols}")
        st.info("Dostępne kolumny: " + ", ".join(df.columns.tolist()))
        return
    
    # Trenowanie modelu ML
    ml_model, scaler, features, accuracy = None, None, [], 0
    if use_ml:
        with st.spinner("Trenowanie modelu AI..."):
            ml_model, scaler, features, accuracy = train_ml_model(df)
    
    # Sygnały
    traditional_signals = calculate_traditional_signals(df)
    ml_signal, ml_confidence = get_ml_signal(df, ml_model, scaler, features) if use_ml else ("HOLD", 0.33)
    
    # Główny sygnał (kombinacja)
    buy_votes = sum(1 for signal in traditional_signals.values() if signal == "BUY")
    sell_votes = sum(1 for signal in traditional_signals.values() if signal == "SELL")
    
    if use_ml and ml_signal == "BUY" and ml_confidence > 0.6:
        buy_votes += 2
    elif use_ml and ml_signal == "SELL" and ml_confidence > 0.6:
        sell_votes += 2
    
    if buy_votes > sell_votes and buy_votes >= 2:
        main_signal = "BUY"
        signal_class = "buy-box"
        signal_emoji = "🟢"
    elif sell_votes > buy_votes and sell_votes >= 2:
        main_signal = "SELL"
        signal_class = "sell-box"
        signal_emoji = "🔴"
    else:
        main_signal = "HOLD"
        signal_class = "hold-box"
        signal_emoji = "🟡"
    
    # Wyświetlanie głównego sygnału
    close_col = ohlc_cols['CLOSE']
    current_price = df[close_col].iloc[-1]
    
    st.markdown(f"""
    <div class="signal-box {signal_class}">
        <h2>{signal_emoji} GŁÓWNY SYGNAŁ: {main_signal}</h2>
        <p>Aktualna cena: <strong>{current_price:.2f} PLN</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Szczegóły sygnałów
    with col2:
        st.subheader("🔍 Analiza sygnałów")
        
        # Tradycyjne sygnały
        for indicator, signal in traditional_signals.items():
            color = "#00C851" if signal == "BUY" else "#FF4444" if signal == "SELL" else "#FF8800"
            st.markdown(f"**{indicator}**: <span style='color: {color}'>{signal}</span>", unsafe_allow_html=True)
        
        # Sygnał ML
        if use_ml and ml_model is not None:
            st.markdown("---")
            st.subheader("🤖 Sygnał AI")
            color = "#00C851" if ml_signal == "BUY" else "#FF4444" if ml_signal == "SELL" else "#FF8800"
            st.markdown(f"**Predykcja**: <span style='color: {color}'>{ml_signal}</span>", unsafe_allow_html=True)
            st.progress(ml_confidence)
            st.caption(f"Pewność: {ml_confidence:.1%}")
            if accuracy > 0:
                st.caption(f"Dokładność modelu: {accuracy:.1%}")
    
    # Wykres główny
    with col1:
        fig_price = go.Figure()
        
        # Świece
        fig_price.add_trace(go.Candlestick(
            x=df.index,
            open=df[ohlc_cols['OPEN']],
            high=df[ohlc_cols['HIGH']],
            low=df[ohlc_cols['LOW']],
            close=df[ohlc_cols['CLOSE']],
            increasing_line_color='#00C851',
            decreasing_line_color='#FF4444',
            name='Cena'
        ))
        
        # Średnie kroczące
        if 'SMA_20' in df.columns:
            fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], 
                                         mode='lines', name='SMA 20', 
                                         line=dict(color='orange', width=1)))
        
        if 'SMA_50' in df.columns:
            fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], 
                                         mode='lines', name='SMA 50', 
                                         line=dict(color='blue', width=2)))
        
        if 'SMA_200' in df.columns:
            fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], 
                                         mode='lines', name='SMA 200', 
                                         line=dict(color='purple', width=2)))
        
        # Bollinger Bands
        bb_upper_col = None
        bb_lower_col = None
        for col in df.columns:
            if 'BBU' in col:
                bb_upper_col = col
            elif 'BBL' in col:
                bb_lower_col = col
        
        if bb_upper_col and bb_lower_col:
            fig_price.add_trace(go.Scatter(x=df.index, y=df[bb_upper_col], 
                                         mode='lines', name='BB Upper', 
                                         line=dict(color='gray', dash='dash')))
            fig_price.add_trace(go.Scatter(x=df.index, y=df[bb_lower_col], 
                                         mode='lines', name='BB Lower', 
                                         line=dict(color='gray', dash='dash')))
        
        fig_price.update_layout(
            title=f"{ticker} ({symbol}) - Analiza techniczna",
            xaxis_title="Data",
            yaxis_title="Cena (PLN)",
            height=500,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        st.plotly_chart(fig_price, use_container_width=True)
    
    # Wskaźniki oscylatorów
    col3, col4 = st.columns(2)
    
    with col3:
        if 'RSI_14' in df.columns:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], 
                                       mode='lines', name='RSI 14', 
                                       line=dict(color='purple')))
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.update_layout(title="RSI", height=300, 
                                yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col4:
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], 
                                        mode='lines', name='MACD'))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], 
                                        mode='lines', name='Signal'))
            fig_macd.update_layout(title="MACD", height=300)
            st.plotly_chart(fig_macd, use_container_width=True)
    
    # Podsumowanie i rekomendacje
    st.subheader("📋 Podsumowanie i rekomendacje")
    
    # Oblicz zmianę ceny
    price_change = 0
    if len(df) > 1:
        price_change = ((current_price - df[close_col].iloc[-2]) / df[close_col].iloc[-2]) * 100
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("Aktualna cena", f"{current_price:.2f} PLN", 
                 f"{price_change:+.2f}%")
    
    with col6:
        volume_col = ohlc_cols.get('VOLUME')
        if volume_col and volume_col in df.columns:
            volume = df[volume_col].iloc[-1]
            st.metric("Wolumen", f"{volume:,.0f}")
        else:
            st.metric("Wolumen", "N/A")
    
    with col7:
        if 'RSI_14' in df.columns:
            rsi_value = df['RSI_14'].iloc[-1]
            st.metric("RSI", f"{rsi_value:.1f}")
    
    # Instrukcje handlowe
    if main_signal == "BUY":
        st.success("""
        🟢 **SYGNAŁ KUPNA**
        - Rozważ otwarcie pozycji długiej
        - Ustaw stop-loss 2-3% poniżej aktualnej ceny
        - Cele: kolejne poziomy oporu
        """)
    elif main_signal == "SELL":
        st.error("""
        🔴 **SYGNAŁ SPRZEDAŻY**
        - Rozważ zamknięcie pozycji długiej lub otwarcie krótkiej
        - Ustaw stop-loss 2-3% powyżej aktualnej ceny
        - Cele: kolejne poziomy wsparcia
        """)
    else:
        st.warning("""
        🟡 **SYGNAŁ WSTRZYMANIA**
        - Brak wyraźnego trendu
        - Poczekaj na lepsze sygnały
        - Monitoruj kluczowe poziomy wsparcia/oporu
        """)
    
    # Disclaimer
    st.markdown("---")
    st.caption("""
    ⚠️ **OSTRZEŻENIE**: Ta aplikacja służy wyłącznie celom edukacyjnym i informacyjnym. 
    Nie stanowi porady inwestycyjnej. Handel na giełdzie wiąże się z ryzykiem straty kapitału.
    Zawsze przeprowadź własną analizę przed podjęciem decyzji inwestycyjnych.
    """)

if __name__ == "__main__":
    main()
