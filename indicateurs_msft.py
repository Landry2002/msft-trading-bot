#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calcul des indicateurs techniques pour MSFT
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

print("="*50)
print("CALCUL DES INDICATEURS TECHNIQUES")
print("="*50)

# Charger le dernier fichier
files = glob.glob("data/MSFT_brut_*.csv")
if not files:
    print("❌ Aucune donnée trouvée. Exécute d'abord collecte_msft.py")
    exit()

latest = max(files)
df = pd.read_csv(latest, index_col=0, parse_dates=True)
print(f"📂 Données chargées: {latest}")
print(f"   {len(df)} jours")

# 1. Moyennes mobiles
for period in [5, 10, 20, 50, 200]:
    df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()

# 2. RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# 3. MACD
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# 4. Rendements
df['Returns'] = df['Close'].pct_change() * 100
df['Volatility'] = df['Returns'].rolling(window=20).std()

# 5. Target (ce qu'on veut prédire)
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Supprimer les NaN
df_clean = df.dropna()
print(f"✅ {len(df_clean)} jours après nettoyage")

# Sauvegarder
date = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"data/MSFT_indicators_{date}.csv"
df_clean.to_csv(filename)
print(f"💾 Indicateurs sauvegardés: {filename}")
print(f"   {len(df_clean.columns)} colonnes")