# -*- coding: utf-8 -*-
"""
Dashboard interactif — HR Turnover Predictor
Étape 8 : Interface Streamlit pour les responsables RH

Lancement : streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Turnover Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH    = "HRDataset_v14.csv"
RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────
# CHARGEMENT & PRÉTRAITEMENT (CACHÉ)
# ──────────────────────────────────────────────────────────────

@st.cache_data
def load_and_prepare():
    """Charge, nettoie et encode le dataset. Retourne X, y et le df original."""
    df_raw = pd.read_csv(DATA_PATH)

    # Nettoyage des espaces dans les chaînes
    for col in df_raw.select_dtypes(include="object").columns:
        df_raw[col] = df_raw[col].str.strip()

    df = df_raw.copy()

    # Dates
    df["DOB"]      = pd.to_datetime(df["DOB"], dayfirst=False, errors="coerce")
    df["DateofHire"] = pd.to_datetime(df["DateofHire"], dayfirst=False, errors="coerce")
    df["LastPerformanceReview_Date"] = pd.to_datetime(
        df["LastPerformanceReview_Date"], dayfirst=False, errors="coerce"
    )

    ref = pd.Timestamp("2019-03-01")
    df["Age"]               = (ref - df["DOB"]).dt.days // 365
    df["Tenure"]            = (ref - df["DateofHire"]).dt.days // 365
    df["DaysSinceLastReview"] = (ref - df["LastPerformanceReview_Date"]).dt.days
    df["ManagerID"]         = df["ManagerID"].fillna(df["ManagerID"].median())

    # Feature engineering
    df["AbsenteeismRate"]      = df["Absences"] / df["Tenure"].clip(lower=1)
    df["RiskScore_Engagement"] = df["DaysLateLast30"] * 2 + df["Absences"]

    # Suppression PII + leaky columns
    drop_cols = [
        "Employee_Name", "EmpID", "Zip", "ManagerName",
        "DOB", "DateofHire", "LastPerformanceReview_Date",
        "TermReason", "EmploymentStatus", "EmpStatusID", "DateofTermination",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    y = df["Termd"].copy()
    X = df.drop(columns=["Termd"])

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    return X, y, df_raw


@st.cache_resource
def train_model(X_hash: str):
    """Entraîne le modèle RandomForest et SHAP explainer. X_hash sert de clé de cache."""
    X, y, _ = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),  columns=X.columns, index=X_test.index)

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_sc, y_train)

    probs  = model.predict_proba(X_test_sc)[:, 1]
    preds  = model.predict(X_test_sc)
    auc    = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds, target_names=["Actif", "Démission"], output_dict=True)

    # Explainer SHAP
    explainer   = shap.TreeExplainer(model, X_train_sc)
    shap_values = explainer.shap_values(X_test_sc)

    # Normalisation API SHAP
    if isinstance(shap_values, list):
        sv_class1 = shap_values[1]
    elif shap_values.ndim == 3:
        sv_class1 = shap_values[:, :, 1]
    else:
        sv_class1 = shap_values

    expected_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train_sc,
        "X_test": X_test_sc,
        "y_test": y_test,
        "probs": probs,
        "auc": auc,
        "report": report,
        "explainer": explainer,
        "sv_class1": sv_class1,
        "expected_value": expected_value,
        "feature_names": list(X.columns),
    }


# ──────────────────────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────────────────────

def risk_badge(prob: float) -> str:
    if prob >= 0.70:
        return "🔴 **RISQUE ÉLEVÉ**"
    elif prob >= 0.40:
        return "🟠 **RISQUE MODÉRÉ**"
    else:
        return "🟢 **RISQUE FAIBLE**"

def risk_color(prob: float) -> str:
    if prob >= 0.70:
        return "#F44336"
    elif prob >= 0.40:
        return "#FF9800"
    return "#4CAF50"


def build_feature_vector(inputs: dict, feature_names: list, scaler: StandardScaler,
                          X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un vecteur de features conforme au modèle à partir des inputs du formulaire.
    Les colonnes non renseignées prennent la valeur moyenne originale (non normalisée).
    """
    # Base : moyennes originales (non normalisées) via scaler.mean_
    row = pd.Series(scaler.mean_, index=feature_names)

    # ── Variables numériques directes ──────────────────────────
    numeric_map = {
        "Salary":                inputs.get("Salary", 62000),
        "Age":                   inputs.get("Age", 35),
        "Tenure":                inputs.get("Tenure", 5),
        "EngagementSurvey":      inputs.get("EngagementSurvey", 4.0),
        "EmpSatisfaction":       inputs.get("EmpSatisfaction", 3),
        "Absences":              inputs.get("Absences", 10),
        "DaysLateLast30":        inputs.get("DaysLateLast30", 0),
        "SpecialProjectsCount":  inputs.get("SpecialProjectsCount", 0),
        "GenderID":              1 if inputs.get("Sex", "M") == "M" else 0,
        "MarriedID":             int(inputs.get("Married", False)),
        "FromDiversityJobFairID": int(inputs.get("DiversityHire", False)),
    }

    # Recalcul des features dérivées
    tenure_safe = max(inputs.get("Tenure", 1), 1)
    numeric_map["AbsenteeismRate"]      = inputs.get("Absences", 10) / tenure_safe
    numeric_map["RiskScore_Engagement"] = inputs.get("DaysLateLast30", 0) * 2 + inputs.get("Absences", 10)
    numeric_map["DaysSinceLastReview"]  = inputs.get("DaysSinceLastReview", 365)

    for col, val in numeric_map.items():
        if col in row.index:
            row[col] = val

    # ── Variables one-hot : tout mettre à 0 puis activer la bonne ──
    categorical_prefixes = [
        "Department", "Position", "State", "Sex",
        "MaritalDesc", "CitizenDesc", "HispanicLatino",
        "RaceDesc", "RecruitmentSource", "PerformanceScore"
    ]
    for prefix in categorical_prefixes:
        ohe_cols = [c for c in feature_names if c.startswith(f"{prefix}_")]
        for c in ohe_cols:
            row[c] = 0

    def set_ohe(prefix, value):
        col = f"{prefix}_{value}"
        if col in row.index:
            row[col] = 1

    set_ohe("Department",        inputs.get("Department", "Production"))
    set_ohe("PerformanceScore",  inputs.get("PerformanceScore", "Fully Meets"))
    set_ohe("RecruitmentSource", inputs.get("RecruitmentSource", "Indeed"))
    set_ohe("Sex",               inputs.get("Sex", "M"))

    # Normalisation — toutes les valeurs sont en échelle originale
    df_row = row.to_frame().T
    arr    = scaler.transform(df_row)
    return pd.DataFrame(arr, columns=feature_names)


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

def render_sidebar(results: dict, df_raw: pd.DataFrame):
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/people.png", width=60
    )
    st.sidebar.title("HR Turnover Predictor")
    st.sidebar.markdown("---")

    total = len(df_raw)
    termd = df_raw["Termd"].sum()
    st.sidebar.metric("Employés analysés", total)
    st.sidebar.metric("Taux de départs", f"{termd/total*100:.1f}%")
    st.sidebar.metric("AUC-ROC du modèle", f"{results['auc']:.3f}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Modèle** : Random Forest  \n"
        "**Explicabilité** : SHAP  \n"
        "**Cible** : `Termd` (0 = Actif | 1 = Départ)"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Projet Hackathon IA RH · Données anonymisées · "
        "Conformité RGPD appliquée"
    )


# ──────────────────────────────────────────────────────────────
# ONGLET 1 — VUE GLOBALE
# ──────────────────────────────────────────────────────────────

def tab_overview(df_raw: pd.DataFrame, results: dict):
    st.header("📊 Vue Globale des données RH")

    col1, col2, col3, col4 = st.columns(4)
    df_clean = df_raw.copy()
    for c in df_clean.select_dtypes("object"):
        df_clean[c] = df_clean[c].str.strip()

    total     = len(df_clean)
    departed  = df_clean["Termd"].sum()
    avg_sat   = df_clean["EmpSatisfaction"].mean()
    avg_abs   = df_clean["Absences"].mean()

    col1.metric("Total employés",    total)
    col2.metric("Départs totaux",    departed, delta=f"{departed/total*100:.1f}%")
    col3.metric("Satisfaction moy.", f"{avg_sat:.2f}/5")
    col4.metric("Absences moyennes", f"{avg_abs:.1f} j/an")

    st.markdown("---")
    c1, c2 = st.columns(2)

    # Taux de départ par département
    with c1:
        st.subheader("Départs par département")
        dept_stats = df_clean.groupby("Department")["Termd"].agg(["sum", "count"])
        dept_stats["rate"] = dept_stats["sum"] / dept_stats["count"] * 100
        dept_stats = dept_stats.sort_values("rate", ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#F44336" if r > 40 else "#FF9800" if r > 20 else "#4CAF50"
                  for r in dept_stats["rate"]]
        dept_stats["rate"].plot(kind="barh", ax=ax, color=colors)
        ax.set_xlabel("Taux de départ (%)")
        ax.set_title("Taux de départ par département")
        for i, v in enumerate(dept_stats["rate"]):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Distribution satisfaction selon statut
    with c2:
        st.subheader("Satisfaction vs. Statut de départ")
        fig, ax = plt.subplots(figsize=(7, 4))
        df_clean["Termd_label"] = df_clean["Termd"].map({0: "Actif", 1: "Démissionné"})
        sns.violinplot(
            data=df_clean, x="Termd_label", y="EmpSatisfaction",
            palette={"Actif": "#4CAF50", "Démissionné": "#F44336"}, ax=ax
        )
        ax.set_title("Distribution de la satisfaction")
        ax.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    c3, c4 = st.columns(2)

    # Absences vs. Départ
    with c3:
        st.subheader("Absentéisme vs. Départ")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=df_clean, x="Termd_label", y="Absences",
            palette={"Actif": "#2196F3", "Démissionné": "#F44336"}, ax=ax
        )
        ax.set_title("Jours d'absence par statut")
        ax.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Top 10 raisons de départ
    with c4:
        st.subheader("Top raisons de démission")
        reasons = (
            df_clean[df_clean["Termd"] == 1]["TermReason"]
            .value_counts()
            .head(8)
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        reasons[::-1].plot(kind="barh", ax=ax, color="#E53935")
        ax.set_title("Raisons de départ (top 8)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Performances du modèle
    st.markdown("---")
    st.subheader("Performance du modèle Random Forest")
    report = results["report"]
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("AUC-ROC",    f"{results['auc']:.3f}")
    col_b.metric("Précision",  f"{report['Démission']['precision']:.3f}")
    col_c.metric("Rappel",     f"{report['Démission']['recall']:.3f}")
    col_d.metric("F1-Score",   f"{report['Démission']['f1-score']:.3f}")


# ──────────────────────────────────────────────────────────────
# ONGLET 2 — PRÉDICTION
# ──────────────────────────────────────────────────────────────

def tab_prediction(results: dict):
    st.header("🎯 Prédiction du risque de départ")
    st.info(
        "Renseignez le profil de l'employé ci-dessous. "
        "Le modèle calculera la probabilité de démission et identifiera les facteurs de risque."
    )

    with st.form("employee_form"):
        st.subheader("Informations générales")
        c1, c2, c3 = st.columns(3)
        with c1:
            dept = st.selectbox(
                "Département",
                ["Production", "IT/IS", "Sales", "Software Engineering",
                 "Admin Offices", "Executive Office"],
            )
            sex = st.selectbox("Genre", ["M", "F"])
            married = st.checkbox("Marié(e)")

        with c2:
            age    = st.slider("Âge", 18, 70, 35)
            tenure = st.slider("Ancienneté (années)", 0, 25, 5)
            salary = st.number_input("Salaire annuel ($)", 40000, 300000, 62000, step=1000)

        with c3:
            perf_score = st.selectbox(
                "Score de performance",
                ["Fully Meets", "Exceeds", "Needs Improvement", "PIP"]
            )
            recruit_src = st.selectbox(
                "Source de recrutement",
                ["Indeed", "LinkedIn", "Google Search", "Employee Referral",
                 "Diversity Job Fair", "CareerBuilder", "Website", "Other"]
            )
            diversity_hire = st.checkbox("Recruté via salon diversité")

        st.subheader("Engagement & Présence")
        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            engagement = st.slider("Score d'engagement", 1.0, 5.0, 4.0, step=0.1)
        with r2:
            satisfaction = st.slider("Satisfaction (1-5)", 1, 5, 3)
        with r3:
            absences = st.slider("Absences (jours/an)", 0, 20, 10)
        with r4:
            days_late = st.slider("Retards (30 derniers j.)", 0, 6, 0)
        with r5:
            special_proj = st.slider("Projets spéciaux", 0, 8, 0)

        submitted = st.form_submit_button("Prédire le risque de départ")

    if submitted:
        inputs = {
            "Department":        dept,
            "Sex":               sex,
            "Married":           married,
            "DiversityHire":     diversity_hire,
            "Age":               age,
            "Tenure":            tenure,
            "Salary":            salary,
            "PerformanceScore":  perf_score,
            "RecruitmentSource": recruit_src,
            "EngagementSurvey":  engagement,
            "EmpSatisfaction":   satisfaction,
            "Absences":          absences,
            "DaysLateLast30":    days_late,
            "SpecialProjectsCount": special_proj,
            "DaysSinceLastReview":  365,
        }

        X_row = build_feature_vector(
            inputs,
            results["feature_names"],
            results["scaler"],
            results["X_train"],
        )

        prob = results["model"].predict_proba(X_row)[0, 1]
        pred = int(prob >= 0.5)

        st.markdown("---")
        st.subheader("Résultat de la prédiction")

        col_res, col_gauge = st.columns([2, 1])
        with col_res:
            st.markdown(f"### {risk_badge(prob)}")
            st.markdown(f"**Probabilité de départ : {prob*100:.1f}%**")
            st.progress(float(prob))
            if prob >= 0.70:
                st.error(
                    "⚠️ Cet employé présente un risque élevé de quitter l'entreprise. "
                    "Une action RH préventive est recommandée."
                )
            elif prob >= 0.40:
                st.warning(
                    "Risque modéré détecté. Un entretien de suivi est conseillé."
                )
            else:
                st.success("Le profil de cet employé présente peu de signes de départ imminent.")

        with col_gauge:
            fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={"aspect": "equal"})
            theta = np.linspace(0, np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), "lightgray", lw=10)
            ax.plot(
                np.cos(theta[:int(prob * 200)]),
                np.sin(theta[:int(prob * 200)]),
                color=risk_color(prob), lw=10
            )
            ax.text(0, -0.2, f"{prob*100:.0f}%", ha="center", va="center",
                    fontsize=22, fontweight="bold", color=risk_color(prob))
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.5, 1.2)
            ax.axis("off")
            ax.set_title("Risque", fontsize=12)
            st.pyplot(fig)
            plt.close()

        # SHAP waterfall individuel
        st.subheader("Explication SHAP — Pourquoi ce score ?")
        try:
            shap_row = results["explainer"].shap_values(X_row)
            if isinstance(shap_row, list):
                sv_row = shap_row[1][0]
            elif shap_row.ndim == 3:
                sv_row = shap_row[0, :, 1]
            else:
                sv_row = shap_row[0]

            exp = shap.Explanation(
                values       = sv_row,
                base_values  = results["expected_value"],
                data         = X_row.values[0],
                feature_names= results["feature_names"],
            )
            fig_w, ax_w = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(exp, max_display=15, show=False)
            plt.title("Facteurs influençant la prédiction")
            plt.tight_layout()
            st.pyplot(fig_w)
            plt.close()

            # Tableau des 10 facteurs les plus importants pour ce profil
            shap_df = pd.DataFrame({
                "Variable": results["feature_names"],
                "SHAP":     sv_row,
            }).sort_values("SHAP", key=abs, ascending=False).head(10)
            shap_df["Direction"] = shap_df["SHAP"].apply(
                lambda v: "↑ Augmente le risque" if v > 0 else "↓ Réduit le risque"
            )

            st.markdown("**Top 10 facteurs explicatifs pour ce profil :**")
            st.dataframe(
                shap_df[["Variable", "SHAP", "Direction"]].style.format({"SHAP": "{:.4f}"}),
                width='stretch'
            )
        except Exception as e:
            st.warning(f"Explication SHAP non disponible pour cette prédiction : {e}")


# ──────────────────────────────────────────────────────────────
# ONGLET 3 — EXPLICATIONS IA
# ──────────────────────────────────────────────────────────────

def tab_xai(results: dict):
    st.header("🧠 Intelligence Artificielle Explicable")

    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** mesure la contribution de chaque variable
    à la prédiction du modèle, en se basant sur la théorie des jeux coopératifs.

    - Une valeur SHAP **positive** → augmente la probabilité de démission
    - Une valeur SHAP **négative** → la réduit
    - La magnitude indique l'importance de la variable
    """)

    st.subheader("Impact global des variables (Beeswarm Plot)")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        results["sv_class1"], results["X_test"],
        feature_names=results["feature_names"], show=False
    )
    plt.title("SHAP — Impact sur la prédiction de démission")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    st.subheader("Importance moyenne des variables (SHAP)")
    mean_abs = np.abs(results["sv_class1"]).mean(axis=0)
    fi_df = pd.DataFrame({
        "Variable": results["feature_names"],
        "SHAP moyen": mean_abs,
    }).sort_values("SHAP moyen", ascending=True).tail(20)

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    fi_df.set_index("Variable")["SHAP moyen"].plot(kind="barh", ax=ax2, color="#1976D2")
    ax2.set_title("Top 20 variables — Importance moyenne SHAP")
    ax2.set_xlabel("|SHAP| moyen")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Feature importance Random Forest classique
    st.subheader("Importance des variables — Random Forest (Gini)")
    fi_rf = pd.Series(
        results["model"].feature_importances_,
        index=results["feature_names"]
    ).sort_values(ascending=True).tail(20)

    fig3, ax3 = plt.subplots(figsize=(10, 7))
    fi_rf.plot(kind="barh", ax=ax3, color="#43A047")
    ax3.set_title("Top 20 variables — Random Forest Feature Importance")
    ax3.set_xlabel("Importance (Gini)")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Explication du modèle
    st.markdown("---")
    st.subheader("Interprétation des résultats")
    top5 = sorted(
        zip(results["feature_names"], mean_abs),
        key=lambda x: x[1], reverse=True
    )[:5]
    st.markdown("**Les 5 principaux facteurs de démission identifiés :**")
    for i, (feat, val) in enumerate(top5, 1):
        st.markdown(f"{i}. **{feat}** — contribution SHAP moyenne : `{val:.4f}`")


# ──────────────────────────────────────────────────────────────
# ONGLET 4 — SÉCURITÉ & RGPD
# ──────────────────────────────────────────────────────────────

def tab_security():
    st.header("🔐 Sécurité & Conformité RGPD")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mesures appliquées")
        st.success("""
✅ **Pseudonymisation des données**
Noms, identifiants (EmpID), dates de naissance et codes postaux supprimés.

✅ **Suppression de la fuite de données**
TermReason, EmploymentStatus et EmpStatusID exclus du modèle (variables post-événement).

✅ **Droit à l'explication (RGPD Art. 22)**
Chaque prédiction est accompagnée d'une explication SHAP individuelle.

✅ **Split stratifié train/test**
Évaluation honnête du modèle — aucune donnée de test vue à l'entraînement.

✅ **Équilibrage des classes**
`class_weight="balanced"` pour éviter le biais vers la classe majoritaire.

✅ **Audit SHAP**
Vérification que le modèle n'utilise pas de variables discriminantes comme facteurs principaux.
        """)

    with col2:
        st.subheader("Recommandations supplémentaires")
        st.warning("""
🔐 **Chiffrement des données**
Chiffrement AES-256 au repos, TLS/HTTPS en transit.

🔐 **Contrôle d'accès (RBAC)**
Accès au dashboard restreint aux responsables RH accrédités.

🔐 **Journal d'audit**
Traçabilité de chaque prédiction individuelle (qui, quand, quel résultat).

🔐 **Tests adversariaux**
Valider la robustesse du modèle face aux manipulations intentionnelles des entrées.

🔐 **Analyse des biais**
Auditer les taux FP/FN par genre, origine ethnique pour détecter les discriminations.

🔐 **Durée de conservation**
Politique de suppression des données conforme au RGPD (droit à l'oubli).

🔐 **Revue humaine obligatoire**
Toute décision impactante (fin de contrat, formation) doit être validée par un humain.
        """)

    st.markdown("---")
    st.subheader("Variables exclues pour la conformité")
    excluded_data = {
        "Variable": [
            "Employee_Name", "EmpID", "DOB", "Zip", "ManagerName",
            "TermReason", "EmploymentStatus", "EmpStatusID", "DateofTermination"
        ],
        "Raison d'exclusion": [
            "Identifiant personnel (PII)",
            "Identifiant unique (PII)",
            "Date de naissance → remplacée par Age (RGPD)",
            "Code postal trop spécifique (PII)",
            "Identifiant personnel (PII)",
            "Variable post-événement (fuite de données)",
            "Encode directement la variable cible (fuite)",
            "Code de statut d'emploi (fuite)",
            "Date de licenciement (fuite + PII)",
        ],
        "Action": [
            "Supprimée", "Supprimée", "Transformée → Age",
            "Supprimée", "Supprimée",
            "Supprimée", "Supprimée", "Supprimée", "Supprimée"
        ]
    }
    st.dataframe(pd.DataFrame(excluded_data), width='stretch')

    st.markdown("---")
    st.subheader("Risques liés aux systèmes d'IA RH")
    risks = [
        ("Biais algorithmique", "Le modèle peut reproduire des biais historiques si les données d'entraînement les contiennent.", "Élevé"),
        ("Manipulation d'entrée", "Un utilisateur malveillant peut modifier les données pour tromper le modèle.", "Modéré"),
        ("Extraction du modèle", "Requêtes répétées pour reconstituer le modèle (model stealing).", "Modéré"),
        ("Surconfiance", "Les RH pourraient ignorer le contexte humain au profit du score.", "Élevé"),
        ("Fuite de modèle", "Les explications SHAP peuvent révéler indirectement des informations sensibles.", "Faible"),
    ]
    risk_df = pd.DataFrame(risks, columns=["Risque", "Description", "Niveau"])
    def _color_niveau(v):
        if v == "Élevé":
            return "background-color: #ffcccc"
        elif v == "Modéré":
            return "background-color: #ffe0b2"
        return "background-color: #c8e6c9"

    st.dataframe(
        risk_df.style.map(_color_niveau, subset=["Niveau"]),
        width='stretch',
    )


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    X, y, df_raw = load_and_prepare()
    results = train_model(str(list(X.columns)))   # hash = noms des colonnes

    render_sidebar(results, df_raw)

    st.title("👥 HR Turnover Predictor — IA Explicable")
    st.markdown(
        "Système de prédiction des démissions d'employés basé sur le Machine Learning "
        "et l'IA Explicable (SHAP). **Toutes les données personnelles ont été anonymisées.**"
    )
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Vue Globale",
        "🎯 Prédiction",
        "🧠 Explications IA",
        "🔐 Sécurité & RGPD",
    ])

    with tab1:
        tab_overview(df_raw, results)

    with tab2:
        tab_prediction(results)

    with tab3:
        tab_xai(results)

    with tab4:
        tab_security()


main()
