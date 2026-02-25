#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtest de la stratégie de trading - Version avec gestion des timezones
"""

import pandas as pd
import numpy as np
import glob
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import os

print("="*50)
print("BACKTEST DE LA STRATÉGIE")
print("="*50)

# ============================================
# 1. CHARGEMENT DU MODÈLE
# ============================================
model_files = glob.glob("models/msft_rf_*.pkl")
if not model_files:
    print("❌ Aucun modèle trouvé")
    exit()

latest_model = max(model_files)
model = joblib.load(latest_model)
print(f"✅ Modèle chargé: {os.path.basename(latest_model)}")

# Extraire la date du modèle
model_basename = os.path.basename(latest_model)
date_part = model_basename.replace('msft_rf_', '').replace('.pkl', '')
print(f"📅 Date du modèle: {date_part}")

# ============================================
# 2. CHARGEMENT DU SCALER
# ============================================
scaler_file = f"models/scaler_{date_part}.pkl"
if not os.path.exists(scaler_file):
    print(f"❌ Scaler non trouvé: {scaler_file}")
    scaler_files = glob.glob("models/scaler_*.pkl")
    if scaler_files:
        scaler_file = max(scaler_files)
        print(f"✅ Utilisation du scaler: {os.path.basename(scaler_file)}")
    else:
        print("❌ Aucun scaler trouvé")
        exit()

scaler = joblib.load(scaler_file)
print(f"✅ Scaler chargé: {os.path.basename(scaler_file)}")

# ============================================
# 3. CHARGEMENT DES DONNÉES
# ============================================
data_files = glob.glob("data/MSFT_indicators_*.csv")
if not data_files:
    print("❌ Aucune donnée trouvée")
    exit()

latest_data = max(data_files)
df = pd.read_csv(latest_data, index_col=0, parse_dates=True)

# Solution pour les timezones mixtes - on convertit tout en UTC puis on enlève le timezone
try:
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
except:
    # Si ça échoue, on essaie une approche plus simple
    df.index = pd.to_datetime(df.index).tz_localize(None)

print(f"📂 Données: {os.path.basename(latest_data)}")
print(f"   {len(df)} jours")
print(f"   Période: {df.index[0].strftime('%Y-%m-%d')} à {df.index[-1].strftime('%Y-%m-%d')}")

# ============================================
# 4. CHARGEMENT DES FEATURES
# ============================================
features_file = f"models/features_{date_part}.txt"
if os.path.exists(features_file):
    with open(features_file, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    print(f"✅ Features chargées: {len(features)}")
else:
    exclude = ['Target', 'Dividends', 'Stock Splits']
    features = [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]
    print(f"⚠️ Utilisation de toutes les colonnes numériques: {len(features)}")

# ============================================
# 5. PRÉPARATION DES DONNÉES
# ============================================
available_features = [f for f in features if f in df.columns]
X = df[available_features].dropna()
df_clean = df.loc[X.index].copy()
print(f"✅ {len(df_clean)} jours après nettoyage")

# ============================================
# 6. PRÉDICTIONS
# ============================================
X_scaled = scaler.transform(X)
df_clean['Prediction'] = model.predict(X_scaled)
df_clean['Prob'] = model.predict_proba(X_scaled)[:, 1]

print(f"\n📊 STATISTIQUES DES PRÉDICTIONS:")
print(f"   HAUSSE: {(df_clean['Prediction'] == 1).sum()} ({((df_clean['Prediction'] == 1).sum()/len(df_clean)*100):.1f}%)")
print(f"   BAISSE: {(df_clean['Prediction'] == 0).sum()} ({((df_clean['Prediction'] == 0).sum()/len(df_clean)*100):.1f}%)")

# ============================================
# 7. SIMULATION DES TRADES
# ============================================
capital = 10000
position = 0
trades = []
equity = [capital]
equity_dates = [df_clean.index[0]]
equity_idx = [0]

print(f"\n💰 SIMULATION DES TRADES:")
print(f"{'-'*50}")

for i in range(len(df_clean)-1):
    current_date = df_clean.index[i]
    current_price = df_clean['Close'].iloc[i]
    
    # Convertir en string pour l'affichage
    date_str = current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)
    
    if df_clean['Prediction'].iloc[i] == 1 and position == 0:
        # Achat
        shares = capital / current_price
        position = shares
        capital = 0
        trades.append(('ACHAT', i, current_date, current_price, df_clean['Prob'].iloc[i]))
        print(f"   📈 ACHAT {date_str} @ ${current_price:.2f} (conf: {df_clean['Prob'].iloc[i]:.1%})")
        
    elif df_clean['Prediction'].iloc[i] == 0 and position > 0:
        # Vente
        capital = position * current_price
        buy_price = trades[-1][3]
        pnl = (current_price / buy_price - 1) * 100
        position = 0
        trades.append(('VENTE', i, current_date, current_price, pnl))
        print(f"   📉 VENTE {date_str} @ ${current_price:.2f} (PnL: {pnl:+.2f}%)")
    
    # Valeur du portefeuille
    if position > 0:
        equity.append(position * current_price)
    else:
        equity.append(capital)
    equity_dates.append(current_date)
    equity_idx.append(i+1)

# Ajouter la dernière valeur
if position > 0:
    final_price = df_clean['Close'].iloc[-1]
    capital = position * final_price
    buy_price = trades[-1][3]
    pnl = (final_price / buy_price - 1) * 100
    last_date = df_clean.index[-1]
    date_str = last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else str(last_date)
    print(f"   🔚 CLÔTURE {date_str} @ ${final_price:.2f} (PnL: {pnl:+.2f}%)")

equity.append(capital)
equity_dates.append(df_clean.index[-1])
equity_idx.append(len(df_clean))

# ============================================
# 8. CALCUL DES RÉSULTATS
# ============================================
final_capital = capital
total_return = (final_capital - 10000) / 10000 * 100
buy_hold_return = (df_clean['Close'].iloc[-1] / df_clean['Close'].iloc[0] - 1) * 100

print(f"\n{'='*50}")
print("💰 RÉSULTATS FINAUX")
print(f"{'='*50}")
print(f"Capital initial: $10,000.00")
print(f"Capital final: ${final_capital:,.2f}")
print(f"Return total: {total_return:+.2f}%")
print(f"Buy & Hold: {buy_hold_return:+.2f}%")
print(f"Performance vs B&H: {total_return - buy_hold_return:+.2f}%")

buy_trades = [t for t in trades if t[0] == 'ACHAT']
sell_trades = [t for t in trades if t[0] == 'VENTE']
print(f"Nombre de trades: {len(buy_trades)}")

if len(sell_trades) > 0:
    pnls = [t[4] for t in sell_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    win_rate = len(wins) / len(sell_trades) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    print(f"Win rate: {win_rate:.1f}%")
    if wins:
        print(f"Gain moyen: {avg_win:+.2f}%")
    if losses:
        print(f"Perte moyenne: {avg_loss:.2f}%")
    if avg_loss != 0:
        profit_factor = abs(avg_win / avg_loss)
        print(f"Profit Factor: {profit_factor:.2f}")

# ============================================
# 9. GRAPHIQUE - VERSION SIMPLIFIÉE
# ============================================
print("\n📊 Génération du graphique...")

# Créer une figure avec 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

# === Graphique 1 : Courbe d'équité ===
# Notre stratégie
ax1.plot(equity_idx, equity, label='Notre Stratégie', color='blue', linewidth=2)

# Buy & Hold
buy_hold = 10000 * df_clean['Close'] / df_clean['Close'].iloc[0]
ax1.plot(range(len(df_clean)), buy_hold, label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)

# Marquer les trades
for trade in trades:
    if trade[0] == 'ACHAT':
        ax1.scatter(trade[1], equity[trade[1]], color='green', marker='^', s=100, zorder=5, edgecolor='black', linewidth=2)
    elif trade[0] == 'VENTE':
        ax1.scatter(trade[1], equity[trade[1]], color='red', marker='v', s=100, zorder=5, edgecolor='black', linewidth=2)

ax1.set_title('Backtest - Stratégie vs Buy & Hold', fontsize=14, fontweight='bold')
ax1.set_ylabel('Capital ($)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# === Graphique 2 : Prix avec signaux ===
# Prix
ax2.plot(range(len(df_clean)), df_clean['Close'], label='Prix MSFT', color='black', alpha=0.7, linewidth=1)

# Signaux d'achat/vente
for trade in trades:
    if trade[0] == 'ACHAT':
        ax2.scatter(trade[1], trade[3], color='green', marker='^', s=80, zorder=5, edgecolor='black', linewidth=2)
    elif trade[0] == 'VENTE':
        ax2.scatter(trade[1], trade[3], color='red', marker='v', s=80, zorder=5, edgecolor='black', linewidth=2)

ax2.set_xlabel('Jours')
ax2.set_ylabel('Prix ($)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Ajouter des labels pour les dates importantes
n_dates = 8
step = len(df_clean) // n_dates
for i in range(0, len(df_clean), step):
    try:
        date_str = df_clean.index[i].strftime('%Y-%m')
    except:
        date_str = str(df_clean.index[i])[:7]
    ax2.text(i, ax2.get_ylim()[0] - 5, date_str, ha='center', va='top', fontsize=8, rotation=45)

plt.tight_layout()

# Sauvegarder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
graph_file = f'graphs/backtest_{timestamp}.png'
plt.savefig(graph_file, dpi=100, bbox_inches='tight')
print(f"📊 Graphique sauvegardé: {graph_file}")

# Afficher
plt.show()

print("\n✅ Backtest terminé avec succès!")