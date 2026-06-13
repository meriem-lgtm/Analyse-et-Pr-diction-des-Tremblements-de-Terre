"""
Dashboard Streamlit — Analyse & Prédiction des Séismes
Lancer : streamlit run dashboard.py
"""

import os, warnings
warnings.filterwarnings('ignore')
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, r2_score
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌍 Earthquake ML Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
  .block-container { padding-top: 1.5rem; }
  h1 { color: #e63946; }
  h2 { color: #457b9d; border-bottom: 2px solid #457b9d; padding-bottom: .3rem; }
</style>
""", unsafe_allow_html=True)

RANDOM_STATE = 42

# ── Helpers ───────────────────────────────────────────────────────────────────
def region_enc(lat, lon):
    if lat >= 0 and -30 <= lon <= 60: return 0
    if lat >= 0 and lon > 60:         return 1
    if lat >= 0 and lon < -30:        return 2
    if lat < 0  and -30 <= lon <= 60: return 3
    if lat < 0  and lon > 60:         return 4
    return 5

REGION_LABELS = {
    0: "Europe/Afrique", 1: "Asie", 2: "Amérique Nord",
    3: "Afrique Sud",    4: "Océanie", 5: "Amérique Sud"
}

def mag_class(m):
    if m < 5:   return 'Faible'
    if m < 6.5: return 'Moyen'
    return 'Fort'

FEATURES = [
    'Latitude',
    'Longitude',
    'Depth',
    'Year',
    'Month',
    'Day',
    'Hour',
    'Type_enc',
    'Region_enc',
    'lat_bin',
    'lon_bin',
    'distance_center',
    'depth_ratio',
    'is_deep'
]

# ── Chargement & préparation des données ──────────────────────────────────────
@st.cache_data(show_spinner="Chargement du dataset…")
def load_data(path: str):
    df = pd.read_csv(path)
    keep = ['Date', 'Time', 'Latitude', 'Longitude', 'Type', 'Depth', 'Magnitude']
    df = df[keep].copy().drop_duplicates()

    df['Datetime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str),
        errors='coerce', utc=True
    )
    mask = df['Datetime'].isna()
    df.loc[mask, 'Datetime'] = pd.to_datetime(df.loc[mask, 'Date'], errors='coerce', utc=True)
    df = df.dropna(subset=['Datetime', 'Magnitude', 'Latitude', 'Longitude', 'Depth'])
    df['Datetime'] = df['Datetime'].dt.tz_convert(None)

    df = df[(df['Depth'] >= 0) & (df['Depth'] <= 800)]
    df = df[(df['Magnitude'] >= 0) & (df['Magnitude'] <= 10)]

    df['Year']  = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    df['Day']   = df['Datetime'].dt.day
    df['Hour']  = df['Datetime'].dt.hour

    df['Region_enc'] = [region_enc(la, lo) for la, lo in zip(df['Latitude'], df['Longitude'])]
    df['Region']     = df['Region_enc'].map(REGION_LABELS)
    df['lat_bin']    = pd.cut(df['Latitude'],  bins=18, labels=False)
    df['lon_bin']    = pd.cut(df['Longitude'], bins=36, labels=False)

    le_type = LabelEncoder()
    df['Type_enc'] = le_type.fit_transform(df['Type'].astype(str))
    df['distance_center'] = np.sqrt(
    df['Latitude']**2 +
    df['Longitude']**2
)

    df['depth_ratio'] = (
    df['Depth'] /
    (df['Magnitude'] + 1)
)

    df['is_deep'] = (
    df['Depth'] > 300
).astype(int)
    df['magnitude_class'] = df['Magnitude'].apply(mag_class)
    return df, le_type

@st.cache_resource(show_spinner="Entraînement du modèle XGBoost…")
def train_model(df):
    le_y = LabelEncoder()
    X    = df[FEATURES].fillna(0).values
    y    = le_y.fit_transform(df['magnitude_class'].values)
    y_r  = df['Magnitude'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X, y_r, test_size=0.2, random_state=RANDOM_STATE
    )
    sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_tr
)
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    # ── Tous les classifieurs ─────────────────────────────────────────────────
    classifiers = {

    'LogisticRegression': (
        LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        True
    ),

    'RandomForest': (
        RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        False
    ),

    'XGBoost': (
        xgb.XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        False
    ),

    'LightGBM': (
        lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ),
        False
    )
}

    cls_results, cls_fitted = [], {}
    for name, (model, scaled) in classifiers.items():
        Xtr = X_tr_s if scaled else X_tr
        Xte = X_te_s if scaled else X_te
        model.fit(Xtr, y_tr)
        pred = model.predict(Xte)
        cls_results.append({
            'Modèle': name,
            'Accuracy':  round(accuracy_score(y_te, pred), 4),
            'Precision': round(precision_score(y_te, pred, average='weighted', zero_division=0), 4),
            'Recall':    round(recall_score(y_te, pred, average='weighted', zero_division=0), 4),
            'F1':        round(f1_score(y_te, pred, average='weighted', zero_division=0), 4),
        })
        cls_fitted[name] = (model, scaled)

    cls_df = pd.DataFrame(cls_results).sort_values('Accuracy', ascending=False).reset_index(drop=True)

    best_name = cls_df.iloc[0]['Modèle']
    best_model, best_scaled = cls_fitted[best_name]
    pred_best = best_model.predict(X_te_s if best_scaled else X_te)
    cm = confusion_matrix(y_te, pred_best)

    # ── Régresseur XGBoost ───────────────────────────────────────────────────
    reg = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                           random_state=RANDOM_STATE, n_jobs=-1)
    reg.fit(X_tr_r, y_tr_r)
    pred_r = reg.predict(X_te_r)

    reg_metrics = {
        'MAE':  round(mean_absolute_error(y_te_r, pred_r), 4),
        'RMSE': round(float(np.sqrt(np.mean((y_te_r - pred_r)**2))), 4),
        'R2':   round(r2_score(y_te_r, pred_r), 4),
    }

    return {
        'cls_df':      cls_df,
        'cls_fitted':  cls_fitted,
        'best_name':   best_name,
        'best_model':  best_model,
        'best_scaled': best_scaled,
        'cm':          cm,
        'le_y':        le_y,
        'scaler':      scaler,
        'X_te':        X_te,
        'X_te_s':      X_te_s,
        'y_te':        y_te,
        'reg':         reg,
        'reg_metrics': reg_metrics,
        'pred_reg':    pred_r,
        'y_te_r':      y_te_r,
        'X_te_r':      X_te_r,
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Seismograph_output.svg/320px-Seismograph_output.svg.png",
    width="stretch"
)
st.sidebar.title("⚙️ Configuration")

data_path = st.sidebar.text_input("Chemin du CSV", value="database.csv")
if not os.path.exists(data_path):
    st.sidebar.error(f"Fichier introuvable : {data_path}")
    st.stop()

df, le_type = load_data(data_path)
results     = train_model(df)

page = st.sidebar.radio(
    "Navigation",
    ["📊 Exploration", "🗺️ Carte mondiale", "🤖 Modèles ML", "🔍 Prédiction", "📈 Régression"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Séismes** : {len(df):,}")
st.sidebar.markdown(f"**Période** : {df['Year'].min()} – {df['Year'].max()}")
st.sidebar.markdown(f"**Best model** : `{results['best_name']}`")
best_acc = results['cls_df'].iloc[0]['Accuracy']
st.sidebar.markdown(f"**Accuracy** : `{best_acc*100:.1f} %`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Exploration
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Exploration":
    st.title("🌍 Analyse Exploratoire des Séismes")
    st.caption("Dataset USGS Earthquake Database (Kaggle) — 1965-2016")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total séismes", f"{len(df):,}")
    c2.metric("Magnitude max", f"{df['Magnitude'].max():.1f}")
    c3.metric("Profondeur max", f"{df['Depth'].max():.0f} km")
    c4.metric("Types uniques", df['Type'].nunique())

    st.markdown("---")

    # Distribution des magnitudes
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x='Magnitude', nbins=60, color='magnitude_class',
            color_discrete_map={'Faible': '#2196F3', 'Moyen': '#FF9800', 'Fort': '#F44336'},
            title="Distribution des magnitudes par classe",
            labels={'magnitude_class': 'Classe'}
        )
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        depth_bins = pd.cut(df['Depth'], bins=[0,70,300,800], labels=['Superficiel','Intermédiaire','Profond'])
        fig = px.pie(
            values=depth_bins.value_counts().values,
            names=depth_bins.value_counts().index,
            title="Répartition par profondeur",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)

    # Évolution temporelle
    yearly = df.groupby('Year').agg(
        count=('Magnitude', 'count'),
        mag_mean=('Magnitude', 'mean')
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['count'], name="Nombre", marker_color='#457b9d'), secondary_y=False)
    fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['mag_mean'], name="Mag. moy.", line=dict(color='#e63946', width=2)), secondary_y=True)
    fig.update_layout(title="Évolution annuelle des séismes (1965–2016)", xaxis_title="Année")
    fig.update_yaxes(title_text="Nombre de séismes", secondary_y=False)
    fig.update_yaxes(title_text="Magnitude moyenne", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Par région
    col3, col4 = st.columns(2)
    with col3:
        reg_counts = df['Region'].value_counts().reset_index()
        reg_counts.columns = ['Region', 'count']
        fig = px.bar(reg_counts, x='Region', y='count', color='Region',
                     title="Séismes par région", color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(df, x='magnitude_class', y='Depth',
                     color='magnitude_class',
                     color_discrete_map={'Faible': '#2196F3', 'Moyen': '#FF9800', 'Fort': '#F44336'},
                     title="Profondeur vs Classe de magnitude",
                     category_orders={'magnitude_class': ['Faible', 'Moyen', 'Fort']})
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap mois x heure
    pivot = df.pivot_table(values='Magnitude', index='Month', columns='Hour', aggfunc='count', fill_value=0)
    fig = px.imshow(pivot, color_continuous_scale='YlOrRd',
                    title="Fréquence des séismes par Mois × Heure (UTC)",
                    labels=dict(color="Nombre", x="Heure", y="Mois"))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Carte mondiale
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Carte mondiale":
    st.title("🗺️ Carte mondiale des séismes")

    col1, col2, col3 = st.columns(3)
    mag_min  = col1.slider("Magnitude minimale", 0.0, 9.5, 5.0, 0.1)
    year_rng = col2.slider("Période", int(df['Year'].min()), int(df['Year'].max()),
                           (int(df['Year'].min()), int(df['Year'].max())))
    n_pts    = col3.slider("Points max (carte)", 1000, 10000, 5000, 500)

    subset = df[
        (df['Magnitude'] >= mag_min) &
        (df['Year'] >= year_rng[0]) &
        (df['Year'] <= year_rng[1])
    ].sample(min(n_pts, len(df[(df['Magnitude']>=mag_min)])), random_state=42)

    st.caption(f"{len(subset):,} événements affichés")

    fig = px.scatter_geo(
        subset, lat='Latitude', lon='Longitude',
        color='Magnitude', size='Magnitude',
        hover_name='Type',
        hover_data={'Year': True, 'Depth': True, 'magnitude_class': True, 'Latitude': False, 'Longitude': False},
        projection='natural earth',
        color_continuous_scale='Inferno',
        title=f"Séismes ≥ {mag_min} ({year_rng[0]}–{year_rng[1]})"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Densité
    fig2 = px.density_mapbox(
        subset, lat='Latitude', lon='Longitude', z='Magnitude',
        radius=6, center=dict(lat=0, lon=0), zoom=0,
        mapbox_style='open-street-map',
        color_continuous_scale='YlOrRd',
        title="Densité géographique"
    )
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Modèles ML
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Modèles ML":
    st.title("🤖 Comparaison des modèles ML")

    # KPIs du meilleur modèle
    best_row = results['cls_df'].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Meilleur modèle", best_row['Modèle'])
    c2.metric("✅ Accuracy", f"{best_row['Accuracy']*100:.1f} %")
    c3.metric("📐 F1 (weighted)", f"{best_row['F1']:.4f}")
    c4.metric("🎯 Precision", f"{best_row['Precision']:.4f}")

    st.markdown("---")

    # Tableau comparatif
    st.subheader("Tableau comparatif")
    styled = results['cls_df'].style.highlight_max(
        subset=['Accuracy', 'F1'], color='#d4edda'
    ).format({'Accuracy': '{:.2%}', 'Precision': '{:.2%}', 'Recall': '{:.2%}', 'F1': '{:.4f}'})
    st.dataframe(styled, use_container_width=True)

    # Graphique comparatif
    fig = px.bar(
        results['cls_df'].melt(id_vars='Modèle', var_name='Métrique', value_name='Score'),
        x='Modèle', y='Score', color='Métrique', barmode='group',
        title="Comparaison des métriques par modèle",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig, use_container_width=True)

    # Matrice de confusion
    st.subheader(f"Matrice de confusion — {results['best_name']}")
    le_y = results['le_y']
    cm   = results['cm']
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        x=le_y.classes_, y=le_y.classes_,
        color_continuous_scale='Blues',
        labels=dict(x="Prédite", y="Réelle", color="Nb"),
        title=f"Confusion Matrix — {results['best_name']}"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importance
    best_model = results['best_model']
    if hasattr(best_model, 'feature_importances_'):
        st.subheader("Importance des variables")
        imp = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig_imp = px.bar(imp, orientation='h', title="Feature Importance",
                         color=imp.values, color_continuous_scale='Teal',
                         labels={'value': 'Importance', 'index': 'Variable'})
        st.plotly_chart(fig_imp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Prédiction interactive
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Prédiction":
    st.title("🔍 Prédiction interactive")
    st.markdown("Entrez les paramètres d'un séisme potentiel pour prédire sa classe de magnitude.")

    col1, col2, col3 = st.columns(3)
    with col1:
        lat    = st.number_input("Latitude", -90.0, 90.0, 35.0, 0.1)
        lon    = st.number_input("Longitude", -180.0, 180.0, 139.0, 0.1)
        depth  = st.number_input("Profondeur (km)", 0.0, 800.0, 50.0, 1.0)
    with col2:
        year   = st.number_input("Année", 1965, 2030, 2024)
        month  = st.slider("Mois", 1, 12, 6)
        day    = st.slider("Jour", 1, 31, 15)
        hour   = st.slider("Heure (UTC)", 0, 23, 12)
    with col3:
        stype  = st.selectbox("Type", ['Earthquake', 'Nuclear Explosion', 'Explosion', 'Rock Burst'])

    le_y      = results['le_y']
    best_model = results['best_model']
    scaler    = results['scaler']
    le_type_enc = LabelEncoder()
    le_type_enc.classes_ = np.array(['Earthquake', 'Explosion', 'Nuclear Explosion', 'Rock Burst'])

    try:
        type_enc_val = list(le_type_enc.classes_).index(stype)
    except ValueError:
        type_enc_val = 0

    reg_val = region_enc(lat, lon)
    lat_b   = min(17, max(0, int((lat + 90) / 180 * 18)))
    lon_b   = min(35, max(0, int((lon + 180) / 360 * 36)))

    X_pred = np.array([[lat, lon, depth, year, month, day, hour,
                        type_enc_val, reg_val, lat_b, lon_b]])

    if results['best_scaled']:
        X_pred_use = scaler.transform(X_pred)
    else:
        X_pred_use = X_pred

    if st.button("🚀 Prédire", type="primary"):
        pred_cls   = best_model.predict(X_pred_use)[0]
        pred_label = le_y.inverse_transform([pred_cls])[0]

        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(X_pred_use)[0]
        else:
            proba = None

        color_map = {'Faible': '🟢', 'Moyen': '🟡', 'Fort': '🔴'}
        st.markdown(f"## Résultat : {color_map.get(pred_label, '')} **{pred_label}**")

        if proba is not None:
            st.subheader("Probabilités par classe")
            prob_df = pd.DataFrame({
                'Classe': le_y.classes_,
                'Probabilité': proba
            })
            fig = px.bar(prob_df, x='Classe', y='Probabilité',
                         color='Classe',
                         color_discrete_map={'Faible': '#2196F3', 'Moyen': '#FF9800', 'Fort': '#F44336'},
                         range_y=[0, 1], title="Distribution des probabilités")
            fig.add_hline(y=0.5, line_dash='dash', line_color='gray')
            st.plotly_chart(fig, use_container_width=True)

        # Contexte géographique
        st.subheader("Localisation")
        fig_map = px.scatter_geo(
            pd.DataFrame({'lat': [lat], 'lon': [lon], 'label': ['Événement prédit']}),
            lat='lat', lon='lon', text='label',
            projection='natural earth', size_max=20
        )
        fig_map.update_traces(marker=dict(size=15, color='red', symbol='star'))
        st.plotly_chart(fig_map, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Régression
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Régression":
    st.title("📈 Régression — Prédiction de la Magnitude")

    m = results['reg_metrics']
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{m['MAE']:.4f}")
    c2.metric("RMSE", f"{m['RMSE']:.4f}")
    c3.metric("R²", f"{m['R2']:.4f}")

    st.markdown("---")

    # Valeurs réelles vs prédites
    n_show = min(1000, len(results['y_te_r']))
    idx    = np.random.choice(len(results['y_te_r']), n_show, replace=False)
    fig = px.scatter(
        x=results['y_te_r'][idx], y=results['pred_reg'][idx],
        labels={'x': 'Magnitude réelle', 'y': 'Magnitude prédite'},
        title=f"Réel vs Prédit (XGBoost Regressor, n={n_show})",
        opacity=0.5, color_discrete_sequence=['#457b9d']
    )
    fig.add_shape(type='line',
                  x0=results['y_te_r'].min(), y0=results['y_te_r'].min(),
                  x1=results['y_te_r'].max(), y1=results['y_te_r'].max(),
                  line=dict(color='red', dash='dash'))
    st.plotly_chart(fig, use_container_width=True)

    # Distribution des résidus
    residuals = results['y_te_r'][idx] - results['pred_reg'][idx]
    fig_res = px.histogram(residuals, nbins=50, title="Distribution des résidus",
                           labels={'value': 'Résidu', 'count': 'Fréquence'},
                           color_discrete_sequence=['#e63946'])
    fig_res.add_vline(x=0, line_dash='dash', line_color='black')
    st.plotly_chart(fig_res, use_container_width=True)

    # Feature importance régression
    reg = results['reg']
    if hasattr(reg, 'feature_importances_'):
        imp = pd.Series(reg.feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig_imp = px.bar(imp, orientation='h',
                         title="Feature Importance — Régresseur",
                         color=imp.values, color_continuous_scale='Magma',
                         labels={'value': 'Importance', 'index': 'Variable'})
        st.plotly_chart(fig_imp, use_container_width=True)
