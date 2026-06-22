"""
Earthquake ML Pipeline — VERSION CORRIGÉE & ROBUSTE
Fix imbalance + features + training propre
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb

from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────
DATA_PATH = "database.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset:", df.shape)

# ─────────────────────────────
# 2. CLEANING
# ─────────────────────────────
keep = ['Date','Time','Latitude','Longitude','Type','Depth','Magnitude']
df = df[keep].drop_duplicates()

df["Datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    errors="coerce",
    utc=True
)

df = df.dropna(subset=["Datetime","Magnitude","Latitude","Longitude","Depth"])
df["Datetime"] = df["Datetime"].dt.tz_convert(None)

df = df[(df["Depth"] >= 0) & (df["Depth"] <= 800)]
df = df[(df["Magnitude"] >= 0) & (df["Magnitude"] <= 10)]

# ─────────────────────────────
# 3. FEATURE ENGINEERING (AMÉLIORÉ)
# ─────────────────────────────
df["Year"] = df["Datetime"].dt.year
df["Month"] = df["Datetime"].dt.month
df["Day"] = df["Datetime"].dt.day
df["Hour"] = df["Datetime"].dt.hour

def region_enc(lat, lon):
    if lat >= 0 and -30 <= lon <= 60: return 0
    if lat >= 0 and lon > 60: return 1
    if lat >= 0 and lon < -30: return 2
    if lat < 0 and -30 <= lon <= 60: return 3
    if lat < 0 and lon > 60: return 4
    return 5

df["Region_enc"] = [region_enc(a,b) for a,b in zip(df["Latitude"], df["Longitude"])]

df["lat_bin"] = pd.cut(df["Latitude"], 18, labels=False)
df["lon_bin"] = pd.cut(df["Longitude"], 36, labels=False)

# Type encoding
le_type = LabelEncoder()
df["Type_enc"] = le_type.fit_transform(df["Type"].astype(str))

# 🔥 FEATURES SUPPLÉMENTAIRES (IMPORTANT)
df["distance_center"] = np.sqrt(df["Latitude"]**2 + df["Longitude"]**2)
df["depth_ratio"] = df["Depth"] / (df["Magnitude"] + 1)
df["is_deep"] = (df["Depth"] > 300).astype(int)

# ─────────────────────────────
# 4. TARGET
# ─────────────────────────────
def mag_class(m):
    if m < 5:
        return "Faible"
    elif m < 6.5:
        return "Moyen"
    return "Fort"

df["magnitude_class"] = df["Magnitude"].apply(mag_class)

print("\nDistribution:")
print(df["magnitude_class"].value_counts())

FEATURES = [
    "Latitude","Longitude","Depth",
    "Year","Month","Day","Hour",
    "Type_enc","Region_enc","lat_bin","lon_bin",
    "distance_center","depth_ratio","is_deep"
]

X = df[FEATURES].fillna(0).values

le_y = LabelEncoder()
y = le_y.fit_transform(df["magnitude_class"])

y_reg = df["Magnitude"].values

# ─────────────────────────────
# 5. SPLIT
# ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg,
    test_size=0.2,
    random_state=RANDOM_STATE
)

# ─────────────────────────────
# 6. SCALING
# ─────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# sample weights (🔥 FIX IMBALANCE IMPORTANT)
weights = compute_sample_weight("balanced", y_train)

# ─────────────────────────────
# 7. MODELS (ROBUSTES)
# ─────────────────────────────
models = {
    "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": xgb.XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE
    ),

    "LightGBM": lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1
    )
}

results = []
best_model = None
best_name = ""

print("\n=== TRAINING ===")

for name, model in models.items():

    if name == "LogReg":
        Xtr, Xte = X_train_s, X_test_s
    else:
        Xtr, Xte = X_train, X_test

    model.fit(Xtr, y_train, sample_weight=weights)
    pred = model.predict(Xte)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    rec = recall_score(y_test, pred, average="weighted")

    results.append([name, acc, f1, rec])

    print(name, "OK")

    if f1 > 0.5:
        best_model = model
        best_name = name

results = pd.DataFrame(results, columns=["Model","Accuracy","F1","Recall"])
results = results.sort_values("F1", ascending=False)

print("\nBEST MODEL:", results.iloc[0]["Model"])

# ─────────────────────────────
# 8. EVALUATION
# ─────────────────────────────
pred = best_model.predict(X_test)

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, pred))

print("\nCLASS REPORT:")
print(classification_report(y_test, pred, target_names=le_y.classes_))

# ─────────────────────────────
# 9. REGRESSION
# ─────────────────────────────
reg = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    random_state=RANDOM_STATE
)

reg.fit(X_train_r, y_train_r)
pred_r = reg.predict(X_test_r)

print("\nREGRESSION:")
print("MAE:", mean_absolute_error(y_test_r, pred_r))
print("RMSE:", np.sqrt(mean_squared_error(y_test_r, pred_r)))
print("R2:", r2_score(y_test_r, pred_r))

# ─────────────────────────────
# 10. SAVE
# ─────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/best_classifier.pkl")
joblib.dump(reg, "models/best_regressor.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le_y, "models/label_encoder.pkl")
joblib.dump(le_type, "models/type_encoder.pkl")
joblib.dump(FEATURES, "models/features.pkl")

print("\nMODEL SAVED ✔")