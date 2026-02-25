#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modèle Random Forest pour prédire MSFT
"""

import pandas as pd
import numpy as np
import glob
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

print("="*50)
print("ENTRAÎNEMENT DU MODÈLE RANDOM FOREST")
print("="*50)

# Charger les données
files = glob.glob("data/MSFT_indicators_*.csv")
if not files:
    print("❌ Aucune donnée trouvée")
    exit()

latest = max(files)
df = pd.read_csv(latest, index_col=0, parse_dates=True)
print(f"📂 Données chargées: {latest}")
print(f"   {len(df)} jours")

# Préparer les features
exclude = ['Target', 'Dividends', 'Stock Splits']
features = [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]

X = df[features].copy()
y = df['Target'].copy()

# Split temporel
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"\n📊 Split:")
print(f"   Train: {len(X_train)} jours")
print(f"   Test: {len(X_test)} jours")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("\n🔄 Entraînement...")
model.fit(X_train_scaled, y_train)

# Évaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📈 Accuracy: {accuracy:.2%}")
print("\n📊 Rapport de classification:")
print(classification_report(y_test, y_pred, target_names=['BAISSE', 'HAUSSE']))

# Sauvegarde
date = datetime.now().strftime("%Y%m%d_%H%M%S")
joblib.dump(model, f"models/msft_rf_{date}.pkl")
joblib.dump(scaler, f"models/scaler_{date}.pkl")

# Sauvegarder la liste des features
with open(f"models/features_{date}.txt", 'w') as f:
    for feat in features:
        f.write(f"{feat}\n")

print(f"\n💾 Modèle sauvegardé: models/msft_rf_{date}.pkl")
print(f"💾 Scaler sauvegardé: models/scaler_{date}.pkl")