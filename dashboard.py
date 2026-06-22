"""
Dashboard de Prédiction Sismique — Édition Professionnelle
=============================================================
Lancer : streamlit run dashboard_prediction.py

Ce dashboard se concentre exclusivement sur la prédiction interactive :
- Saisie des paramètres d'un événement sismique potentiel
- Prédiction de la classe de magnitude (Faible / Moyen / Fort)
- Estimation précise de la magnitude (régression)
- Visualisation des probabilités, de la confiance du modèle et de la localisation
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Prédiction Sismique | Seismic AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

RANDOM_STATE = 42

# ════════════════════════════════════════════════════════════════════════════
# STYLE — DESIGN PROFESSIONNEL
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    :root {
        --bg-primary:   #0b0f19;
        --bg-secondary: #131a2a;
        --bg-card:      #161e30;
        --border-col:   #232c42;
        --accent:       #3b82f6;
        --accent-2:     #06b6d4;
        --danger:       #ef4444;
        --warning:      #f59e0b;
        --success:      #10b981;
        --text-main:    #e5e9f0;
        --text-dim:     #8b97ad;
    }

    .stApp { background: radial-gradient(circle at 15% 0%, #101729 0%, #0b0f19 55%); }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1320 0%, #0b0f19 100%);
        border-right: 1px solid var(--border-col);
    }

    /* Header bandeau */
    .hero {
        background: linear-gradient(120deg, #0f1b33 0%, #15233f 60%, #0f1b33 100%);
        border: 1px solid var(--border-col);
        border-radius: 18px;
        padding: 2.1rem 2.4rem;
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute; top: -40%; right: -10%;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(59,130,246,0.18) 0%, transparent 70%);
    }
    .hero-eyebrow {
        color: var(--accent-2);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .5rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f4f6fb;
        margin: 0 0 .35rem 0;
        letter-spacing: -.02em;
    }
    .hero-sub { color: var(--text-dim); font-size: .98rem; margin: 0; }

    /* Cartes section */
    .panel {
        background: var(--bg-card);
        border: 1px solid var(--border-col);
        border-radius: 16px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1.2rem;
    }
    .panel-title {
        font-size: .92rem;
        font-weight: 700;
        color: var(--text-main);
        text-transform: uppercase;
        letter-spacing: .06em;
        margin-bottom: 1.1rem;
        display: flex;
        align-items: center;
        gap: .5rem;
    }
    .panel-title .bar { width: 4px; height: 16px; background: var(--accent); border-radius: 4px; display:inline-block; }

    /* Résultat */
    .result-card {
        border-radius: 18px;
        padding: 2rem 2.2rem;
        border: 1px solid var(--border-col);
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
    }
    .result-faible { background: linear-gradient(135deg, rgba(16,185,129,.14), rgba(20,30,48,.6)); border-color: rgba(16,185,129,.35); }
    .result-moyen  { background: linear-gradient(135deg, rgba(245,158,11,.16), rgba(20,30,48,.6)); border-color: rgba(245,158,11,.35); }
    .result-fort   { background: linear-gradient(135deg, rgba(239,68,68,.18), rgba(20,30,48,.6)); border-color: rgba(239,68,68,.4); }

    .result-label { font-size: .8rem; text-transform: uppercase; letter-spacing: .12em; color: var(--text-dim); font-weight: 700; margin-bottom: .4rem; }
    .result-value { font-size: 2.6rem; font-weight: 800; color: #f8fafc; margin: 0; line-height: 1; }
    .result-conf  { font-size: .95rem; color: var(--text-dim); margin-top: .6rem; }

    .badge {
        display: inline-flex; align-items: center; gap: .4rem;
        padding: .32rem .85rem; border-radius: 999px;
        font-size: .78rem; font-weight: 700; letter-spacing: .03em;
    }
    .badge-faible { background: rgba(16,185,129,.16); color: #34d399; border: 1px solid rgba(16,185,129,.4); }
    .badge-moyen  { background: rgba(245,158,11,.16); color: #fbbf24; border: 1px solid rgba(245,158,11,.4); }
    .badge-fort   { background: rgba(239,68,68,.16);  color: #f87171; border: 1px solid rgba(239,68,68,.4); }

    /* Metric mini-cards */
    .mini-metric {
        background: var(--bg-secondary);
        border: 1px solid var(--border-col);
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }
    .mini-metric .label { font-size: .72rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: .07em; font-weight: 700; }
    .mini-metric .value { font-size: 1.5rem; font-weight: 800; color: var(--text-main); margin-top: .15rem; font-family: 'JetBrains Mono', monospace; }

    [data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 700; color: var(--text-main); }
    [data-testid="stMetricLabel"] { color: var(--text-dim); font-weight: 600; }

    .stButton > button {
        background: linear-gradient(120deg, var(--accent), var(--accent-2));
        color: white; border: none; border-radius: 10px;
        font-weight: 700; padding: .7rem 1.4rem; letter-spacing: .02em;
        box-shadow: 0 6px 20px rgba(59,130,246,.25);
        transition: all .15s ease;
        width: 100%;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 10px 26px rgba(59,130,246,.35); }

    .footer-note { color: var(--text-dim); font-size: .8rem; text-align:center; padding-top: 1.2rem; }

    hr { border-color: var(--border-col); }
    [data-testid="stExpander"] { background: var(--bg-card); border: 1px solid var(--border-col); border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HELPERS MÉTIER
# ════════════════════════════════════════════════════════════════════════════
def region_enc(lat, lon):
    if lat >= 0 and -30 <= lon <= 60: return 0
    if lat >= 0 and lon > 60:         return 1
    if lat >= 0 and lon < -30:        return 2
    if lat < 0  and -30 <= lon <= 60: return 3
    if lat < 0  and lon > 60:         return 4
    return 5

REGION_LABELS = {
    0: "Europe / Afrique du Nord", 1: "Asie", 2: "Amérique du Nord",
    3: "Afrique Australe",         4: "Océanie", 5: "Amérique du Sud"
}

def mag_class(m):
    if m < 5:   return 'Faible'
    if m < 6.5: return 'Moyen'
    return 'Fort'

FEATURES = [
    'Latitude', 'Longitude', 'Depth', 'Year', 'Month', 'Day', 'Hour',
    'Type_enc', 'Region_enc', 'lat_bin', 'lon_bin',
    'distance_center', 'depth_ratio', 'is_deep'
]

CLASS_META = {
    'Faible': {'css': 'faible', 'icon': '🟢', 'desc': 'Risque limité — dégâts généralement négligeables.'},
    'Moyen':  {'css': 'moyen',  'icon': '🟡', 'desc': 'Risque modéré — vigilance recommandée selon la zone.'},
    'Fort':   {'css': 'fort',   'icon': '🔴', 'desc': 'Risque élevé — impact potentiellement significatif.'},
}

# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT & ENTRAÎNEMENT (mis en cache)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Chargement du jeu de données…")
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

    df['distance_center'] = np.sqrt(df['Latitude']**2 + df['Longitude']**2)
    df['depth_ratio']     = df['Depth'] / (df['Magnitude'] + 1)
    df['is_deep']         = (df['Depth'] > 300).astype(int)
    df['magnitude_class'] = df['Magnitude'].apply(mag_class)
    return df, le_type

@st.cache_resource(show_spinner="Entraînement des modèles…")
def train_models(df):
    le_y = LabelEncoder()
    X   = df[FEATURES].fillna(0).values
    y   = le_y.fit_transform(df['magnitude_class'].values)
    y_r = df['Magnitude'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_tr_r, _, y_tr_r, _ = train_test_split(X, y_r, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    scaler.fit(X_tr)

    clf = lgb.LGBMClassifier(
        n_estimators=1200,
        num_leaves=63,
        max_depth=-1,
        learning_rate=0.03,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        min_child_samples=15,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)

    metrics = {
        'Accuracy':  accuracy_score(y_te, pred),
        'Precision': precision_score(y_te, pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_te, pred, average='weighted', zero_division=0),
        'F1':        f1_score(y_te, pred, average='weighted', zero_division=0),
    }

    reg = lgb.LGBMRegressor(
        n_estimators=900,
        num_leaves=63,
        learning_rate=0.03,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        min_child_samples=15,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    reg.fit(X_tr_r, y_tr_r)

    return {'clf': clf, 'reg': reg, 'le_y': le_y, 'scaler': scaler, 'metrics': metrics}

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌐 Seismic AI")
    st.caption("Module de prédiction sismique")
    st.markdown("---")
    data_path = st.text_input("📁 Chemin du dataset (CSV)", value="database.csv")

    if not os.path.exists(data_path):
        st.error(f"Fichier introuvable : `{data_path}`")
        st.stop()

    df, le_type = load_data(data_path)
    models = train_models(df)

    st.markdown("---")
    st.markdown("**📊 État du modèle**")
    st.markdown(f"<div class='mini-metric'><div class='label'>Accuracy</div><div class='value'>{models['metrics']['Accuracy']*100:.1f}%</div></div>", unsafe_allow_html=True)
    st.write("")
    st.markdown(f"<div class='mini-metric'><div class='label'>F1-score</div><div class='value'>{models['metrics']['F1']:.3f}</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"Entraîné sur {len(df):,} séismes\n\n{df['Year'].min()} – {df['Year'].max()}")
    st.caption("Modèle : LightGBM Classifier + Regressor")

# ════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">SEISMIC AI · MODULE DE PRÉDICTION</div>
    <p class="hero-title">🔍 Prédiction d'Événement Sismique</p>
    <p class="hero-sub">Renseignez les paramètres géophysiques d'un événement pour estimer sa classe de magnitude
    et sa valeur précise, à partir d'un modèle LightGBM entraîné sur les données USGS (1965–2016).</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# FORMULAIRE DE SAISIE
# ════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1.05, 1.55], gap="large")

with left:
    st.markdown('<div class="panel"><div class="panel-title"><span class="bar"></span>Paramètres de l\'événement</div>', unsafe_allow_html=True)

    st.markdown("**🌍 Localisation**")
    lat = st.slider("Latitude", -90.0, 90.0, 35.0, 0.1)
    lon = st.slider("Longitude", -180.0, 180.0, 139.0, 0.1)
    depth = st.slider("Profondeur (km)", 0.0, 800.0, 50.0, 1.0)

    st.markdown("**🕒 Date & heure**")
    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Année", 1965, 2035, 2026)
        day  = st.number_input("Jour", 1, 31, 15)
    with c2:
        month = st.number_input("Mois", 1, 12, 6)
        hour  = st.number_input("Heure (UTC)", 0, 23, 12)

    st.markdown("**⚙️ Type d'événement**")
    stype = st.selectbox("Type", ['Earthquake', 'Nuclear Explosion', 'Explosion', 'Rock Burst'], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀  Lancer la prédiction", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    region_name = REGION_LABELS.get(region_enc(lat, lon), "—")
    st.markdown(f"""
    <div class="panel" style="padding-top:1.1rem; padding-bottom:1.1rem;">
        <div class="panel-title" style="margin-bottom:.6rem;"><span class="bar"></span>Contexte géographique</div>
        <span style="color:var(--text-dim); font-size:.9rem;">Région détectée : </span>
        <span class="badge badge-moyen">{region_name}</span>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PRÉPARATION DES FEATURES & PRÉDICTION
# ════════════════════════════════════════════════════════════════════════════
le_type_enc = LabelEncoder()
le_type_enc.classes_ = np.array(['Earthquake', 'Explosion', 'Nuclear Explosion', 'Rock Burst'])
try:
    type_enc_val = list(le_type_enc.classes_).index(stype)
except ValueError:
    type_enc_val = 0

reg_val = region_enc(lat, lon)
lat_b = min(17, max(0, int((lat + 90) / 180 * 18)))
lon_b = min(35, max(0, int((lon + 180) / 360 * 36)))
distance_center = float(np.sqrt(lat**2 + lon**2))

with right:
    if not predict_btn and 'last_pred' not in st.session_state:
        st.markdown("""
        <div class="panel" style="text-align:center; padding: 4rem 2rem;">
            <div style="font-size:2.6rem; margin-bottom:.6rem;">🛰️</div>
            <div style="color:var(--text-main); font-weight:700; font-size:1.15rem; margin-bottom:.4rem;">
                En attente de paramètres
            </div>
            <div style="color:var(--text-dim); font-size:.92rem;">
                Renseignez les champs à gauche puis cliquez sur <b>« Lancer la prédiction »</b>
                pour obtenir une estimation de magnitude.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if predict_btn:
        depth_ratio_est = depth / 6.0  # estimation neutre avant connaissance de la magnitude
        is_deep = 1 if depth > 300 else 0

        X_pred = np.array([[lat, lon, depth, year, month, day, hour,
                             type_enc_val, reg_val, lat_b, lon_b,
                             distance_center, depth_ratio_est, is_deep]])

        clf, reg, le_y, scaler = models['clf'], models['reg'], models['le_y'], models['scaler']

        pred_cls   = clf.predict(X_pred)[0]
        pred_label = le_y.inverse_transform([pred_cls])[0]
        proba      = clf.predict_proba(X_pred)[0]
        confidence = float(np.max(proba))

        pred_mag = float(reg.predict(X_pred)[0])

        st.session_state['last_pred'] = {
            'label': pred_label, 'proba': proba, 'classes': le_y.classes_,
            'confidence': confidence, 'mag': pred_mag,
            'lat': lat, 'lon': lon, 'depth': depth, 'stype': stype
        }

    if 'last_pred' in st.session_state:
        res = st.session_state['last_pred']
        meta = CLASS_META[res['label']]

        st.markdown(f"""
        <div class="result-card result-{meta['css']}">
            <div class="result-label">Classe de magnitude prédite</div>
            <p class="result-value">{meta['icon']} {res['label']}</p>
            <div class="result-conf">{meta['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Magnitude estimée", f"{res['mag']:.2f}")
        m2.metric("Confiance du modèle", f"{res['confidence']*100:.1f} %")
        m3.metric("Profondeur saisie", f"{res['depth']:.0f} km")

        st.markdown("<br>", unsafe_allow_html=True)

        # Probabilités par classe
        st.markdown('<div class="panel"><div class="panel-title"><span class="bar"></span>Distribution des probabilités</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({'Classe': res['classes'], 'Probabilité': res['proba']})
        color_map = {'Faible': '#10b981', 'Moyen': '#f59e0b', 'Fort': '#ef4444'}
        fig = px.bar(
            prob_df, x='Probabilité', y='Classe', orientation='h',
            color='Classe', color_discrete_map=color_map, text='Probabilité'
        )
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(
            showlegend=False, height=240,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e5e9f0', xaxis=dict(range=[0,1], gridcolor='#232c42', tickformat='.0%'),
            yaxis=dict(gridcolor='#232c42'), margin=dict(l=0, r=20, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Jauge de confiance
        st.markdown('<div class="panel"><div class="panel-title"><span class="bar"></span>Niveau de confiance</div>', unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=res['confidence']*100,
            number={'suffix': '%', 'font': {'color': '#e5e9f0', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8b97ad'},
                'bar': {'color': '#3b82f6'},
                'bgcolor': '#161e30',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 50], 'color': '#1f2940'},
                    {'range': [50, 80], 'color': '#243150'},
                    {'range': [80, 100], 'color': '#2c3c63'},
                ],
            }
        ))
        gauge.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10),
                            paper_bgcolor='rgba(0,0,0,0)', font_color='#e5e9f0')
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Carte de localisation
        st.markdown('<div class="panel"><div class="panel-title"><span class="bar"></span>Localisation de l\'événement</div>', unsafe_allow_html=True)
        fig_map = px.scatter_geo(
            pd.DataFrame({'lat': [res['lat']], 'lon': [res['lon']], 'label': [f"{res['stype']} · {res['mag']:.2f}"]}),
            lat='lat', lon='lon', text='label', projection='natural earth'
        )
        fig_map.update_traces(marker=dict(size=16, color=color_map[res['label']], symbol='circle',
                                          line=dict(width=2, color='white')))
        fig_map.update_geos(
            bgcolor='rgba(0,0,0,0)', landcolor='#1c2538', oceancolor='#0b0f19',
            showocean=True, lakecolor='#0b0f19', showcountries=True, countrycolor='#2a344c'
        )
        fig_map.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                              paper_bgcolor='rgba(0,0,0,0)', font_color='#e5e9f0')
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer-note">Seismic AI · Modèle LightGBM entraîné sur USGS Earthquake Database (1965–2016) · Usage indicatif uniquement, ne remplace pas une alerte officielle.</div>', unsafe_allow_html=True)