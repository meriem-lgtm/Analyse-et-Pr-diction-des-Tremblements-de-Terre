# 🌍 Earthquake ML — Guide de démarrage

## Fichiers
| Fichier | Description |
|---------|-------------|
| `earthquake_model.py` | Pipeline ML complet (entraînement, évaluation, sauvegarde) |
| `dashboard.py` | Dashboard Streamlit interactif |
| `database.csv` | Dataset USGS (à placer dans le même dossier) |

## Installation
```bash
pip install pandas numpy scikit-learn xgboost lightgbm streamlit plotly joblib
```

## Étape 1 — Entraîner le modèle
```bash
python earthquake_model.py
```
✅ Accuracy attendue : **~90 %** (XGBoost / LogisticRegression)

## Étape 2 — Lancer le dashboard
```bash
streamlit run dashboard.py
```
Le dashboard s'ouvre sur http://localhost:8501

## Résultats obtenus
| Modèle | Accuracy | F1 |
|--------|----------|----|
| LogisticRegression | **90.2 %** | 0.855 |
| LightGBM | 90.1 % | 0.856 |
| XGBoost | 89.8 % | 0.857 |
| RandomForest | 89.6 % | 0.856 |

## Architecture des classes de magnitude
| Classe | Magnitude |
|--------|-----------|
| Faible | < 5.0 |
| Moyen | 5.0 – 6.5 |
| Fort | ≥ 6.5 |

## Pages du Dashboard
1. **📊 Exploration** — Histogrammes, évolution temporelle, heatmaps
2. **🗺️ Carte mondiale** — Carte interactive filtrée par magnitude & année
3. **🤖 Modèles ML** — Comparaison des 4 modèles, matrice de confusion, feature importance
4. **🔍 Prédiction** — Prédiction interactive avec probabilités
5. **📈 Régression** — Réel vs prédit, distribution des résidus
