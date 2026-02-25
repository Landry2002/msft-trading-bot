#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dashboard Streamlit pour le trading MSFT
Visualisation en temps réel des prédictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
import joblib
import os

# Configuration de la page
st.set_page_config(
    page_title="MSFT Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 20px;
        padding: 20px;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .signal-hausse {
        background: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        border: 2px solid #c3e6cb;
    }
    .signal-baisse {
        background: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        border: 2px solid #f5c6cb;
    }
    .info-text {
        color: #666;
        font-size: 0.9rem;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# CHARGEMENT DU MODÈLE
# ============================================
@st.cache_resource
def load_model():
    """Charge le dernier modèle disponible"""
    model_files = glob.glob("models/msft_rf_*.pkl")
    if not model_files:
        return None, None, None
    
    latest_model = max(model_files)
    model = joblib.load(latest_model)
    
    # Charger le scaler
    date_part = os.path.basename(latest_model).replace('msft_rf_', '').replace('.pkl', '')
    scaler_file = f"models/scaler_{date_part}.pkl"
    
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
    else:
        scaler = None
    
    # Charger les features
    features_file = f"models/features_{date_part}.txt"
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            features = [line.strip() for line in f.readlines()]
    else:
        features = None
    
    return model, scaler, features

# ============================================
# FONCTIONS UTILITAIRES
# ============================================
def get_current_data():
    """Récupère les données actuelles de MSFT"""
    try:
        msft = yf.Ticker("MSFT")
        hist = msft.history(period="5d")
        return hist
    except Exception as e:
        st.error(f"Erreur de récupération des données: {e}")
        return None

def calculate_indicators_for_prediction(df):
    """Calcule les indicateurs pour la prédiction"""
    df = df.copy()
    
    # Moyennes mobiles
    for period in [5, 10, 20, 50]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Returns
    df['Returns'] = df['Close'].pct_change() * 100
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    return df

def get_prediction(model, scaler, features):
    """Obtient une prédiction"""
    try:
        # Récupérer les données récentes
        data = get_current_data()
        if data is None or len(data) < 50:
            return None
        
        # Calculer les indicateurs
        df = calculate_indicators_for_prediction(data)
        
        # Dernier jour
        latest = df.iloc[-1:].copy()
        
        # Features disponibles
        available_features = [f for f in features if f in latest.columns]
        
        if not available_features:
            return None
        
        # Préparer les données
        X = latest[available_features].fillna(0)
        X_scaled = scaler.transform(X)
        
        # Prédire
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]
        
        return {
            'signal': 'HAUSSE' if pred == 1 else 'BAISSE',
            'price': float(latest['Close'].iloc[-1]),
            'confidence': float(max(prob)),
            'prob_hausse': float(prob[1]),
            'prob_baisse': float(prob[0]),
            'date': latest.index[0],
            'rsi': float(latest['RSI'].iloc[-1]) if 'RSI' in latest.columns else None,
            'macd': float(latest['MACD'].iloc[-1]) if 'MACD' in latest.columns else None
        }
    except Exception as e:
        st.error(f"Erreur de prédiction: {e}")
        return None

# ============================================
# CHARGEMENT INITIAL
# ============================================
model, scaler, features = load_model()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/microsoft.png", width=80)
    st.title("MSFT Trading")
    st.markdown("---")
    
    st.subheader("📊 Configuration")
    auto_refresh = st.checkbox("Actualisation automatique", value=True)
    refresh_interval = st.slider("Intervalle (secondes)", 10, 60, 30)
    
    st.markdown("---")
    st.subheader("ℹ️ Informations")
    st.markdown(f"**Modèle:** Random Forest")
    st.markdown(f"**Features:** {len(features) if features else 0}")
    st.markdown(f"**Dernière mise à jour:** {datetime.now().strftime('%H:%M:%S')}")
    
    if st.button("🔄 Actualiser maintenant"):
        st.rerun()

# ============================================
# MAIN CONTENT
# ============================================
st.markdown('<div class="main-header">📈 Microsoft (MSFT) Trading Dashboard</div>', unsafe_allow_html=True)

# Colonnes pour les métriques principales
col1, col2, col3, col4 = st.columns(4)

# Obtenir la prédiction
prediction = get_prediction(model, scaler, features)

if prediction:
    # Métrique 1: Prix actuel
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="💰 Prix Actuel",
            value=f"${prediction['price']:.2f}",
            delta=f"{((prediction['price']/prediction['price']-1)*100):+.2f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Métrique 2: Signal
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if prediction['signal'] == 'HAUSSE':
            st.markdown(f'<div class="signal-hausse">📈 {prediction["signal"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="signal-baisse">📉 {prediction["signal"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Métrique 3: Confiance
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="🎯 Confiance",
            value=f"{prediction['confidence']*100:.1f}%",
            delta=f"{prediction['prob_hausse']*100:.1f}% Hausse"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Métrique 4: RSI
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if prediction['rsi']:
            rsi_value = prediction['rsi']
            rsi_color = "normal"
            if rsi_value > 70:
                rsi_color = "inverse"
            elif rsi_value < 30:
                rsi_color = "off"
            st.metric(
                label="📊 RSI",
                value=f"{rsi_value:.1f}",
                delta="Suracheté" if rsi_value > 70 else "Survendu" if rsi_value < 30 else "Neutre",
                delta_color=rsi_color
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Graphique des prix
    st.markdown("---")
    st.subheader("📈 Évolution du prix")
    
    # Récupérer plus de données pour le graphique
    hist = yf.Ticker("MSFT").history(period="3mo")
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Prix MSFT", "Volume", "RSI")
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name="Prix",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name="Volume",
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    # RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=rsi,
            name="RSI",
            line=dict(color='purple')
        ),
        row=3, col=1
    )
    
    # Lignes RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Probabilités
    st.markdown("---")
    st.subheader("📊 Probabilités")
    
    prob_col1, prob_col2, prob_col3 = st.columns([1, 2, 1])
    
    with prob_col2:
        prob_data = pd.DataFrame({
            'Direction': ['Hausse', 'Baisse'],
            'Probabilité': [prediction['prob_hausse']*100, prediction['prob_baisse']*100]
        })
        
        st.bar_chart(prob_data.set_index('Direction'))
    
    # Détails techniques
    with st.expander("🔍 Détails techniques"):
        st.json({
            'timestamp': datetime.now().isoformat(),
            'price': prediction['price'],
            'signal': prediction['signal'],
            'confidence': prediction['confidence'],
            'prob_hausse': prediction['prob_hausse'],
            'prob_baisse': prediction['prob_baisse'],
            'rsi': prediction['rsi'],
            'macd': prediction['macd']
        })
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

else:
    st.error("❌ Impossible d'obtenir une prédiction. Vérifie que le modèle est bien chargé.")
    
    # Afficher les fichiers disponibles
    st.subheader("📁 Fichiers dans models/")
    files = os.listdir("models/") if os.path.exists("models/") else []
    for f in files:
        st.write(f"- {f}")