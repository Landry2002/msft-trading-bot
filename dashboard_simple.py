#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dashboard Streamlit pour le trading MSFT - Version corrigée
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="MSFT Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Style CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 20px;
    }
    .signal-hausse {
        background-color: #28a745;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .signal-baisse {
        background-color: #dc3545;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📈 Microsoft (MSFT) - Trading Dashboard</div>', unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_url = st.text_input("URL de l'API", "http://127.0.0.1:5000")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_rate = st.slider("Intervalle (secondes)", 5, 60, 30)
    
    if st.button("🔄 Actualiser maintenant"):
        st.rerun()
    
    st.markdown("---")
    st.subheader("📊 Statut")
    
    # Vérifier le statut de l'API
    try:
        response = requests.get(f"{api_url}/status", timeout=2)
        if response.status_code == 200:
            status = response.json()
            st.success(f"✅ API connectée")
            st.info(f"Modèle: {'✅' if status['model_loaded'] else '❌'}")
            st.info(f"Features: {status['features_count']}")
        else:
            st.error("❌ API non disponible")
    except:
        st.error("❌ API non disponible")
        st.info("Lance d'abord: `python api_trading.py`")

# ============================================
# MAIN CONTENT
# ============================================

# Récupérer la prédiction
try:
    response = requests.get(f"{api_url}/predict", timeout=5)
    if response.status_code == 200:
        pred = response.json()
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 Prix Actuel",
                value=f"${pred['price']}",
                delta=None
            )
        
        with col2:
            if pred['signal'] == 'HAUSSE':
                st.markdown('<div class="signal-hausse">📈 HAUSSE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="signal-baisse">📉 BAISSE</div>', unsafe_allow_html=True)
        
        with col3:
            st.metric(
                label="🎯 Confiance",
                value=f"{pred['confidence']}%",
                delta=f"{pred['prob_hausse']}% Hausse"
            )
        
        with col4:
            st.metric(
                label="📅 Date",
                value=pred['date'],
                delta=None
            )
        
        # Graphique
        st.markdown("---")
        st.subheader("📈 Historique des prix")
        
        # Récupérer les données historiques
        msft = yf.Ticker("MSFT")
        hist = msft.history(period="3mo")
        
        # Créer le graphique
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Prix", "Volume")
        )
        
        # Prix
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name="Prix"
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
        
        fig.update_layout(
            height=600,
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Détails
        with st.expander("📋 Détails de la prédiction"):
            st.json(pred)
        
    else:
        st.error(f"❌ Erreur API: {response.status_code}")
        st.json(response.json() if response.text else {})
        
except Exception as e:
    st.warning("⚠️ En attente des données...")
    st.info("Vérifie que l'API tourne: `python api_trading.py`")
    st.error(f"Erreur: {e}")

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()