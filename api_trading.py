#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API de trading pour MSFT - Version corrigée avec toutes les features
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import glob
import os
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)  # Important pour Streamlit

# ============================================
# CHARGEMENT DU MODÈLE
# ============================================
print("📂 Chargement du modèle...")

model = None
scaler = None
features = None

try:
    # Chercher le dernier modèle
    model_files = glob.glob("models/msft_rf_*.pkl")
    if model_files:
        latest_model = max(model_files)
        model = joblib.load(latest_model)
        
        # Extraire la date
        model_basename = os.path.basename(latest_model)
        date_part = model_basename.replace('msft_rf_', '').replace('.pkl', '')
        
        # Charger le scaler
        scaler_file = f"models/scaler_{date_part}.pkl"
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            print(f"✅ Scaler chargé: scaler_{date_part}.pkl")
        else:
            print("❌ Scaler non trouvé")
        
        # Charger les features
        features_file = f"models/features_{date_part}.txt"
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                features = [line.strip() for line in f.readlines()]
            print(f"✅ Features chargées: {len(features)}")
            print(f"   Liste: {features}")
        else:
            print("❌ Fichier features non trouvé")
        
        print(f"✅ Modèle chargé: {model_basename}")
    else:
        print("❌ Aucun modèle trouvé")
        
except Exception as e:
    print(f"❌ Erreur: {e}")
    traceback.print_exc()

# ============================================
# FONCTIONS UTILITAIRES
# ============================================
def calculate_indicators(df):
    """Calcule tous les indicateurs techniques nécessaires au modèle"""
    try:
        df = df.copy()
        print("📊 Calcul des indicateurs techniques...")
        
        # 1. Moyennes mobiles simples (SMA)
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            print(f"   ✅ SMA_{period} calculée")
        
        # 2. RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        print(f"   ✅ RSI calculé")
        
        # 3. MACD complet (avec histogramme)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        print(f"   ✅ MACD calculé (avec histogramme)")
        
        # 4. Returns (rendements)
        df['Returns'] = df['Close'].pct_change() * 100
        print(f"   ✅ Returns calculés")
        
        # 5. Volatilité
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        print(f"   ✅ Volatilité calculée")
        
        # 6. Prix (déjà présents)
        print(f"   ✅ Prix: Open, High, Low, Close")
        
        # 7. Volume
        print(f"   ✅ Volume")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur calculate_indicators: {e}")
        traceback.print_exc()
        return df

def get_prediction():
    """Obtient une prédiction pour MSFT"""
    try:
        if model is None or scaler is None or features is None:
            return None, "Modèle non chargé"
        
        print("\n🔄 Récupération des données Yahoo Finance...")
        # Récupérer les données
        msft = yf.Ticker("MSFT")
        hist = msft.history(period="3mo")  # 3 mois pour avoir assez de données
        
        if hist.empty:
            return None, "Pas de données"
        
        print(f"✅ {len(hist)} jours de données récupérés")
        
        # Calculer les indicateurs
        df = calculate_indicators(hist)
        
        # Prendre le dernier jour
        latest = df.iloc[-1:].copy()
        
        # Features disponibles
        available_features = []
        missing_features = []
        
        for f in features:
            if f in latest.columns:
                available_features.append(f)
            else:
                missing_features.append(f)
        
        if missing_features:
            print(f"⚠️ Features manquantes: {missing_features}")
        
        if not available_features:
            return None, f"Aucune feature disponible. Features dispo: {list(latest.columns)}"
        
        print(f"✅ Features disponibles: {len(available_features)}/{len(features)}")
        
        # Préparer les données
        X = latest[available_features].fillna(0)
        
        # Normaliser
        X_scaled = scaler.transform(X)
        
        # Prédire
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]
        
        # Prix actuel
        current_price = float(latest['Close'].iloc[-1])
        
        result = {
            'ticker': 'MSFT',
            'price': round(current_price, 2),
            'signal': 'HAUSSE' if pred == 1 else 'BAISSE',
            'confidence': round(float(max(prob)) * 100, 1),
            'prob_hausse': round(float(prob[1]) * 100, 1),
            'prob_baisse': round(float(prob[0]) * 100, 1),
            'features_used': len(available_features),
            'features_total': len(features),
            'date': latest.index[0].strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n📊 PRÉDICTION:")
        print(f"   Prix: ${result['price']}")
        print(f"   Signal: {result['signal']}")
        print(f"   Confiance: {result['confidence']}%")
        
        return result, None
        
    except Exception as e:
        print(f"❌ Erreur get_prediction: {e}")
        traceback.print_exc()
        return None, str(e)

# ============================================
# ROUTES
# ============================================
@app.route('/')
def home():
    return jsonify({
        'name': 'MSFT Trading API',
        'status': 'online',
        'model_loaded': model is not None,
        'features_count': len(features) if features else 0,
        'endpoints': [
            '/status',
            '/price',
            '/predict',
            '/model-info'
        ]
    })

@app.route('/status')
def status():
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_count': len(features) if features else 0,
        'features_list': features if features else [],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/price')
def price():
    try:
        msft = yf.Ticker("MSFT")
        hist = msft.history(period="1d")
        current_price = float(hist['Close'].iloc[-1])
        
        return jsonify({
            'ticker': 'MSFT',
            'price': round(current_price, 2),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict')
def predict():
    result, error = get_prediction()
    
    if error:
        return jsonify({
            'error': error,
            'timestamp': datetime.now().isoformat()
        }), 500
    
    return jsonify(result)

@app.route('/model-info')
def model_info():
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    return jsonify({
        'type': type(model).__name__,
        'features': features if features else [],
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'model_file': os.path.basename(max(glob.glob("models/msft_rf_*.pkl"))) if glob.glob("models/msft_rf_*.pkl") else None
    })

@app.route('/debug/features')
def debug_features():
    """Route de debug pour voir les features disponibles"""
    try:
        msft = yf.Ticker("MSFT")
        hist = msft.history(period="2mo")
        df = calculate_indicators(hist)
        
        return jsonify({
            'columns': list(df.columns),
            'features_requises': features if features else [],
            'features_disponibles': [f for f in features if f in df.columns] if features else [],
            'features_manquantes': [f for f in features if f not in df.columns] if features else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 API DE TRADING MSFT")
    print("="*50)
    print("\n📍 Points d'accès:")
    print("   http://127.0.0.1:5000/")
    print("   http://127.0.0.1:5000/status")
    print("   http://127.0.0.1:5000/price")
    print("   http://127.0.0.1:5000/predict")
    print("   http://127.0.0.1:5000/model-info")
    print("   http://127.0.0.1:5000/debug/features")
    print("\n⚠️  Ctrl+C pour arrêter\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)