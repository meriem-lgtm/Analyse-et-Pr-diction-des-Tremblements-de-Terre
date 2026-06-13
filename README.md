# 🌍 Earthquake ML Pipeline

Pipeline de machine learning pour la **classification** et la **prédiction** de la magnitude des séismes, basé sur des données sismiques historiques.

---

## 📁 Structure du projet

```
.
├── database.csv              # Dataset source (23 412 séismes)
├── earthquake_model.py       # Script principal du pipeline ML
├── dashboard.py              # Dashboard de visualisation
└── models/                   # Modèles sauvegardés (généré à l'exécution)
    ├── best_classifier.pkl
    ├── best_regressor.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── type_encoder.pkl
    └── features.pkl
```

---

## 📊 Dataset

**Source :** `database.csv`  
**Taille :** ~23 400 enregistrements de séismes mondiaux

**Colonnes utilisées :**

| Colonne | Description |
|---|---|
| `Date`, `Time` | Horodatage de l'événement |
| `Latitude`, `Longitude` | Localisation géographique |
| `Type` | Type d'événement (Earthquake, Nuclear Explosion, etc.) |
| `Depth` | Profondeur du foyer (km) |
| `Magnitude` | Magnitude du séisme |

---

## ⚙️ Pipeline ML

### 1. Nettoyage des données
- Suppression des doublons
- Conversion des dates en `datetime` UTC
- Filtrage : profondeur ∈ [0, 800 km], magnitude ∈ [0, 10]

### 2. Feature Engineering

| Feature | Description |
|---|---|
| `Year`, `Month`, `Day`, `Hour` | Composantes temporelles |
| `Region_enc` | Région géographique encodée (6 zones) |
| `lat_bin`, `lon_bin` | Bins de latitude/longitude (18 × 36) |
| `Type_enc` | Type d'événement encodé |
| `distance_center` | Distance à l'origine (0°, 0°) |
| `depth_ratio` | Profondeur / (Magnitude + 1) |
| `is_deep` | Séisme profond (> 300 km) — binaire |

### 3. Classification de la magnitude

| Classe | Plage de magnitude |
|---|---|
| **Faible** | < 5.0 |
| **Moyen** | 5.0 – 6.5 |
| **Fort** | ≥ 6.5 |

Le déséquilibre de classes est corrigé via `compute_sample_weight("balanced")`.

### 4. Modèles entraînés

**Classification :**

| Modèle | Particularités |
|---|---|
| Logistic Regression | Baseline, `class_weight="balanced"` |
| Random Forest | 300 arbres, `class_weight="balanced"` |
| XGBoost | 600 estimateurs, `lr=0.03`, `max_depth=7` |
| LightGBM | 500 estimateurs, `lr=0.05`, `class_weight="balanced"` |

Le meilleur modèle est sélectionné selon le **F1-score pondéré**.

**Régression (prédiction de la magnitude exacte) :**

XGBRegressor — 400 estimateurs, `lr=0.05`, `max_depth=6`

---

## 🚀 Installation & Utilisation

### Prérequis

```bash
pip install numpy pandas scikit-learn xgboost lightgbm joblib
```

### Lancer le pipeline

```bash
python earthquake_model.py
```

Le script affiche dans la console :
- La distribution des classes
- Les performances de chaque modèle (Accuracy, F1, Recall)
- La matrice de confusion et le rapport de classification du meilleur modèle
- Les métriques de régression (MAE, RMSE, R²)

Les modèles sont ensuite sauvegardés dans le dossier `models/`.

### Lancer le dashboard

```bash
python dashboard.py
```

---

## 📈 Métriques évaluées

**Classification :**
- Accuracy
- F1-score pondéré
- Recall pondéré
- Matrice de confusion

**Régression :**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (coefficient de détermination)

---

## 🔧 Paramètres clés

| Paramètre | Valeur |
|---|---|
| `RANDOM_STATE` | 42 |
| `test_size` | 20 % |
| Split classification | Stratifié par classe |
| Scaler | StandardScaler (Logistic Regression uniquement) |

---

## 📦 Modèles sauvegardés

Après exécution, le dossier `models/` contient :

- `best_classifier.pkl` — Meilleur classificateur (sélection automatique)
- `best_regressor.pkl` — XGBRegressor pour la prédiction de magnitude
- `scaler.pkl` — StandardScaler (pour la Logistic Regression)
- `label_encoder.pkl` — Encodeur des classes de magnitude
- `type_encoder.pkl` — Encodeur du type d'événement
- `features.pkl` — Liste ordonnée des features (pour l'inférence)

Pour charger et utiliser un modèle en inférence :

```python
import joblib
import numpy as np

model = joblib.load("models/best_classifier.pkl")
le_y  = joblib.load("models/label_encoder.pkl")
features = joblib.load("models/features.pkl")

# Exemple (valeurs à adapter)
X_new = np.array([[35.6, 139.7, 50, 2024, 3, 11, 14,
                   0, 1, 10, 20, 150.0, 10.0, 0]])
pred_class = le_y.inverse_transform(model.predict(X_new))
print("Classe prédite :", pred_class[0])
```
