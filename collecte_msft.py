#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collecte des données Microsoft (MSFT)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os

print("="*50)
print("COLLECTE DES DONNÉES MICROSOFT")
print("="*50)

# Télécharger les données (2 ans d'historique)
print("📥 Téléchargement des données MSFT...")
msft = yf.Ticker("MSFT")
df = msft.history(period="2y")

print(f"✅ {len(df)} jours de données récupérés")
print(f"   Période: {df.index[0].date()} à {df.index[-1].date()}")

# Sauvegarder
date = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"data/MSFT_brut_{date}.csv"
df.to_csv(filename)
print(f"💾 Données sauvegardées: {filename}")

# Aperçu
print("\n📊 Aperçu des données:")
print(df.tail())