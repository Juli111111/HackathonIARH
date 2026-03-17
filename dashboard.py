# -*- coding: utf-8 -*-
"""
HR Turnover Predictor — Dashboard interactif
Hackathon IA RH | Étape 8

Lancement : python -m streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap
import streamlit as st

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────
# CONFIGURATION GLOBALE
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HR Turnover Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette de couleurs cohérente (sans emojis)
CLR_HIGH   = "#C62828"
CLR_MED    = "#E65100"
CLR_LOW    = "#2E7D32"
CLR_PRIMARY = "#1565C0"
CLR_SECONDARY = "#37474F"

DATA_PATH    = "HRDataset_v14.csv"
RANDOM_STATE = 42
SELF_EVAL_PATH = "employee_self_evaluations.csv"  # Fichier de persistance


# ──────────────────────────────────────────────────────────────
# CHARGEMENT & PRÉTRAITEMENT (CACHE)
# ──────────────────────────────────────────────────────────────

@st.cache_data
def load_and_prepare():
    """Charge, nettoie et encode le dataset. Retourne X, y, df_raw."""
    df_raw = pd.read_csv(DATA_PATH)
    for col in df_raw.select_dtypes(include="object").columns:
        df_raw[col] = df_raw[col].str.strip()

    df = df_raw.copy()
    df["DOB"]      = pd.to_datetime(df["DOB"], dayfirst=False, errors="coerce")
    df["DateofHire"] = pd.to_datetime(df["DateofHire"], dayfirst=False, errors="coerce")
    df["LastPerformanceReview_Date"] = pd.to_datetime(
        df["LastPerformanceReview_Date"], dayfirst=False, errors="coerce"
    )

    ref = pd.Timestamp("2019-03-01")
    df["Age"]               = (ref - df["DOB"]).dt.days // 365
    df["Tenure"]            = (ref - df["DateofHire"]).dt.days // 365
    df["ManagerID"]         = df["ManagerID"].fillna(df["ManagerID"].median())
    df["AbsenteeismRate"]   = df["Absences"] / df["Tenure"].clip(lower=1)
    df["RiskScore_Engagement"] = df["DaysLateLast30"] * 2 + df["Absences"]

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
    X[X.select_dtypes(include="bool").columns] = (
        X.select_dtypes(include="bool").astype(int)
    )
    return X, y, df_raw


# ──────────────────────────────────────────────────────────────
# ENTRAÎNEMENT + CALIBRATION + FAIRNESS (CACHE)
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def train_model(X_hash: str):
    """
    Entraîne Random Forest, calibre les probabilités (sigmoid),
    calcule SHAP et métriques de biais.
    """
    X, y, df_raw = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    # ── Modèle de base (pour SHAP) ────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train_sc, y_train)

    # ── Calibration des probabilités (sigmoid) ────────────────
    # CalibratedClassifierCV(cv='prefit') utilise le modèle déjà entraîné
    # et calibre sur un jeu de validation dédié pour éviter la fuite.
    X_tr2, X_cal, y_tr2, y_cal = train_test_split(
        X_train_sc, y_train, test_size=0.25, random_state=RANDOM_STATE
    )
    rf_for_calib = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_for_calib.fit(X_tr2, y_tr2)
    calib_model = CalibratedClassifierCV(rf_for_calib, cv="prefit", method="sigmoid")
    calib_model.fit(X_cal, y_cal)

    probs_raw   = rf.predict_proba(X_test_sc)[:, 1]
    probs_calib = calib_model.predict_proba(X_test_sc)[:, 1]
    preds       = (probs_calib >= 0.5).astype(int)
    auc         = roc_auc_score(y_test, probs_calib)
    report      = classification_report(
        y_test, preds, target_names=["Actif", "Démission"], output_dict=True
    )

    # ── SHAP (sur le RF de base) ──────────────────────────────
    explainer   = shap.TreeExplainer(rf, X_train_sc)
    shap_values = explainer.shap_values(X_test_sc, check_additivity=False)
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

    # ── Métriques de biais (fairness) ─────────────────────────
    df_clean = df_raw.copy()
    for c in df_clean.select_dtypes("object"):
        df_clean[c] = df_clean[c].str.strip()

    def _group_metrics(y_true, y_pred, groups, col_name):
        rows = []
        for g in sorted(groups.unique()):
            mask = (groups == g).values
            if mask.sum() < 5:
                continue
            yt, yp = y_true.values[mask], y_pred[mask]
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            if cm.shape != (2, 2):
                continue
            tn, fp, fn, tp = cm.ravel()
            n = len(yt)
            rows.append({
                col_name:    g,
                "N":         n,
                "Accuracy":  round((tp + tn) / n, 3),
                "Precision": round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0,
                "Rappel":    round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0,
                "Tx. FP":    round(fp / (fp + tn), 3) if (fp + tn) > 0 else 0,
                "Tx. FN":    round(fn / (fn + tp), 3) if (fn + tp) > 0 else 0,
            })
        return pd.DataFrame(rows)

    idx = y_test.index
    fairness_gender = _group_metrics(
        y_test, preds, df_clean.loc[idx, "Sex"], "Genre"
    )
    fairness_dept = _group_metrics(
        y_test, preds, df_clean.loc[idx, "Department"], "Département"
    )

    # Courbe de calibration
    prob_true_raw,   prob_pred_raw   = calibration_curve(y_test, probs_raw,   n_bins=8)
    prob_true_calib, prob_pred_calib = calibration_curve(y_test, probs_calib, n_bins=8)

    return {
        "model":             rf,
        "calib_model":       calib_model,
        "scaler":            scaler,
        "X_train":           X_train_sc,
        "X_test":            X_test_sc,
        "y_test":            y_test,
        "probs":             probs_calib,
        "probs_raw":         probs_raw,
        "preds":             preds,
        "auc":               auc,
        "report":            report,
        "explainer":         explainer,
        "sv_class1":         sv_class1,
        "expected_value":    expected_value,
        "feature_names":     list(X.columns),
        "fairness_gender":   fairness_gender,
        "fairness_dept":     fairness_dept,
        "calib_curve_raw":   (prob_pred_raw,   prob_true_raw),
        "calib_curve_calib": (prob_pred_calib, prob_true_calib),
    }


# ──────────────────────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────────────────────

def risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "RISQUE ELEVE"
    elif prob >= 0.40:
        return "RISQUE MODERE"
    return "RISQUE FAIBLE"

def risk_color(prob: float) -> str:
    if prob >= 0.70:
        return CLR_HIGH
    elif prob >= 0.40:
        return CLR_MED
    return CLR_LOW


def build_feature_vector(
    inputs: dict, feature_names: list, scaler: StandardScaler, _unused=None
) -> pd.DataFrame:
    """Construit le vecteur de features normalisé à partir des inputs formulaire."""
    row = pd.Series(0.0, index=feature_names)

    # --- Mapping PerformanceScore texte → PerfScoreID numérique ---
    perf_map = {"Exceeds": 4, "Fully Meets": 3, "Needs Improvement": 2, "PIP": 1}
    perf_text = inputs.get("PerformanceScore", "Fully Meets")
    perf_id = perf_map.get(perf_text, 3)

    # --- Mapping MaritalDesc → MaritalStatusID ---
    marital_map = {"Single": 0, "Married": 1, "Divorced": 2, "Separated": 3, "Widowed": 4}
    married = inputs.get("Married", False)
    marital_status_id = 1 if married else 0

    # --- Mapping Department → DeptID ---
    dept_map = {
        "Admin Offices": 1, "Executive Office": 2, "IT/IS": 3,
        "Production": 5, "Sales": 4, "Software Engineering": 6,
    }
    dept_text = inputs.get("Department", "Production")
    dept_id = dept_map.get(dept_text, 5)

    tenure = inputs.get("Tenure", 5)
    tenure_safe = max(tenure, 1)
    absences = inputs.get("Absences", 10)
    days_late = inputs.get("DaysLateLast30", 0)

    # --- Variables numériques ---
    numeric_map = {
        "Salary":                inputs.get("Salary", 62000),
        "Age":                   inputs.get("Age", 35),
        "Tenure":                tenure,
        "EngagementSurvey":      inputs.get("EngagementSurvey", 4.0),
        "EmpSatisfaction":       inputs.get("EmpSatisfaction", 3),
        "Absences":              absences,
        "DaysLateLast30":        days_late,
        "SpecialProjectsCount":  inputs.get("SpecialProjectsCount", 0),
        "GenderID":              1 if inputs.get("Sex", "M") == "M" else 0,
        "MarriedID":             int(married),
        "FromDiversityJobFairID": int(inputs.get("DiversityHire", False)),
        "PerfScoreID":           perf_id,
        "MaritalStatusID":       marital_status_id,
        "DeptID":                dept_id,
        "PositionID":            19,  # valeur médiane
        "ManagerID":             14,  # valeur médiane
        "AbsenteeismRate":       absences / tenure_safe,
        "RiskScore_Engagement":  days_late * 2 + absences,
    }
    for col, val in numeric_map.items():
        if col in row.index:
            row[col] = val

    # --- Variables catégorielles (one-hot) : tout à 0 d'abord ---
    ohe_prefixes = [
        "Department", "Position", "State", "Sex", "MaritalDesc",
        "CitizenDesc", "HispanicLatino", "RaceDesc",
        "RecruitmentSource", "PerformanceScore",
    ]

    # --- Activer les bonnes colonnes OHE ---
    ohe_mappings = {
        "Department":        inputs.get("Department", "Production"),
        "PerformanceScore":  perf_text,
        "RecruitmentSource": inputs.get("RecruitmentSource", "Indeed"),
        "Sex":               inputs.get("Sex", "M"),
    }
    for prefix, value in ohe_mappings.items():
        col = f"{prefix}_{value}"
        if col in row.index:
            row[col] = 1

    # MaritalDesc
    marital_text = "Married" if married else "Single"
    col = f"MaritalDesc_{marital_text}"
    if col in row.index:
        row[col] = 1

    # CitizenDesc — défaut US Citizen
    if "CitizenDesc_US Citizen" in row.index:
        row["CitizenDesc_US Citizen"] = 1

    # HispanicLatino — défaut no
    if "HispanicLatino_no" in row.index:
        row["HispanicLatino_no"] = 1

    # RaceDesc — défaut White (valeur la plus fréquente)
    if "RaceDesc_White" in row.index:
        row["RaceDesc_White"] = 1

    return pd.DataFrame(scaler.transform(row.to_frame().T), columns=feature_names)


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

def render_sidebar(results: dict, df_raw: pd.DataFrame):
    st.sidebar.title("HR Turnover Predictor")
    st.sidebar.caption("Hackathon IA Ressources Humaines")
    st.sidebar.markdown("---")

    total = len(df_raw)
    termd = df_raw["Termd"].sum()
    st.sidebar.metric("Employes analyses",    total)
    st.sidebar.metric("Taux de departs",      f"{termd / total * 100:.1f} %")
    st.sidebar.metric("AUC-ROC (calibre)",    f"{results['auc']:.3f}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Modele** : Random Forest  \n"
        "**Calibration** : Sigmoid  \n"
        "**Explicabilite** : SHAP  \n"
        "**Cible** : `Termd`"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Donnees anonymisees — Conformite RGPD"
    )


# ──────────────────────────────────────────────────────────────
# ONGLET 1 — VUE GLOBALE
# ──────────────────────────────────────────────────────────────

def tab_overview(df_raw: pd.DataFrame, results: dict):
    st.header("Vue Globale des Donnees RH")

    df = df_raw.copy()
    for c in df.select_dtypes("object"):
        df[c] = df[c].str.strip()
    df["Statut"] = df["Termd"].map({0: "Actif", 1: "Depart confirme"})

    total    = len(df)
    departed = int(df["Termd"].sum())
    avg_sat  = df["EmpSatisfaction"].mean()
    avg_abs  = df["Absences"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total employes",     total)
    c2.metric("Departs enregistres", departed, delta=f"{departed / total * 100:.1f} %")
    c3.metric("Satisfaction moyenne", f"{avg_sat:.2f} / 5")
    c4.metric("Absences moyennes",    f"{avg_abs:.1f} j/an")

    report = results["report"]
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("AUC-ROC",   f"{results['auc']:.3f}")
    c6.metric("Precision", f"{report['Démission']['precision']:.3f}")
    c7.metric("Rappel",    f"{report['Démission']['recall']:.3f}")
    c8.metric("F1-Score",  f"{report['Démission']['f1-score']:.3f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Taux de depart par departement")
        dept = df.groupby("Department")["Termd"].agg(["sum", "count"])
        dept["rate"] = dept["sum"] / dept["count"] * 100
        dept = dept.sort_values("rate")
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [CLR_HIGH if r > 40 else CLR_MED if r > 20 else CLR_LOW
                  for r in dept["rate"]]
        dept["rate"].plot(kind="barh", ax=ax, color=colors)
        for i, v in enumerate(dept["rate"]):
            ax.text(v + 0.3, i, f"{v:.1f} %", va="center", fontsize=9)
        ax.set_xlabel("Taux de depart (%)")
        ax.set_title("Taux de depart par departement")
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.subheader("Distribution de la satisfaction")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.violinplot(
            data=df, x="Statut", y="EmpSatisfaction",
            palette={"Actif": CLR_LOW, "Depart confirme": CLR_HIGH}, ax=ax
        )
        ax.set_title("Satisfaction selon le statut de depart")
        ax.set_xlabel("")
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.subheader("Absenteisme selon le statut")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=df, x="Statut", y="Absences",
            palette={"Actif": CLR_PRIMARY, "Depart confirme": CLR_HIGH}, ax=ax
        )
        ax.set_title("Jours d'absence par statut")
        ax.set_xlabel("")
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r2:
        st.subheader("Principales raisons de depart")
        reasons = (
            df[df["Termd"] == 1]["TermReason"].value_counts().head(8)
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        reasons[::-1].plot(kind="barh", ax=ax, color=CLR_HIGH)
        ax.set_title("Raisons de depart (top 8)")
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Courbes ROC + calibration
    st.markdown("---")
    st.subheader("Evaluation du modele")
    col_roc, col_cal = st.columns(2)

    with col_roc:
        fpr, tpr, _ = roc_curve(results["y_test"], results["probs"])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color=CLR_PRIMARY, lw=2,
                label=f"Random Forest calibre (AUC = {results['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("Taux faux positifs")
        ax.set_ylabel("Taux vrais positifs")
        ax.set_title("Courbe ROC")
        ax.legend()
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_cal:
        x_raw, y_raw     = results["calib_curve_raw"]
        x_cal, y_cal_pts = results["calib_curve_calib"]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibration parfaite")
        ax.plot(x_raw, y_raw, "o-", color=CLR_MED, lw=2, label="Avant calibration")
        ax.plot(x_cal, y_cal_pts, "s-", color=CLR_LOW, lw=2, label="Apres calibration")
        ax.set_xlabel("Probabilite predite")
        ax.set_ylabel("Frequence reelle")
        ax.set_title("Courbe de calibration des probabilites")
        ax.legend()
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ──────────────────────────────────────────────────────────────
# ONGLET 2 — PREDICTION + SIMULATEUR WHAT-IF
# ──────────────────────────────────────────────────────────────

def tab_prediction(results: dict):
    st.header("Prediction du Risque de Depart")
    st.markdown(
        "Renseignez le profil de l'employe. "
        "Le modele calcule une probabilite calibree de demission "
        "et identifie les facteurs determinants."
    )

    # ── Formulaire de saisie ─────────────────────────────────
    with st.form("employee_form"):
        st.subheader("Profil de l'employe")
        c1, c2, c3 = st.columns(3)
        with c1:
            dept = st.selectbox(
                "Departement",
                ["Production", "IT/IS", "Sales", "Software Engineering",
                 "Admin Offices", "Executive Office"],
            )
            sex     = st.selectbox("Genre", ["M", "F"])
            married = st.checkbox("Marie(e)")
        with c2:
            age    = st.slider("Age", 18, 70, 35)
            tenure = st.slider("Anciennete (annees)", 0, 25, 5)
            salary = st.number_input("Salaire annuel ($)", 40000, 300000, 62000, step=1000)
        with c3:
            perf_score = st.selectbox(
                "Score de performance",
                ["Fully Meets", "Exceeds", "Needs Improvement", "PIP"],
            )
            recruit_src = st.selectbox(
                "Source de recrutement",
                ["Indeed", "LinkedIn", "Google Search", "Employee Referral",
                 "Diversity Job Fair", "CareerBuilder", "Website", "Other"],
            )
            diversity_hire = st.checkbox("Recrutement via salon diversite")

        st.subheader("Engagement et presence")
        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            engagement = st.slider("Score d'engagement", 1.0, 5.0, 4.0, step=0.1)
        with r2:
            satisfaction = st.slider("Satisfaction (1-5)", 1, 5, 3)
        with r3:
            absences = st.slider("Absences (j/an)", 0, 20, 10)
        with r4:
            days_late = st.slider("Retards (30 derniers j.)", 0, 6, 0)
        with r5:
            special_proj = st.slider("Projets speciaux", 0, 8, 0)

        submitted = st.form_submit_button("Analyser le profil")

    if submitted:
        inputs = {
            "Department":           dept,
            "Sex":                  sex,
            "Married":              married,
            "DiversityHire":        diversity_hire,
            "Age":                  age,
            "Tenure":               tenure,
            "Salary":               salary,
            "PerformanceScore":     perf_score,
            "RecruitmentSource":    recruit_src,
            "EngagementSurvey":     engagement,
            "EmpSatisfaction":      satisfaction,
            "Absences":             absences,
            "DaysLateLast30":       days_late,
            "SpecialProjectsCount": special_proj,
        }
        X_row = build_feature_vector(
            inputs, results["feature_names"], results["scaler"]
        )
        prob = float(results["calib_model"].predict_proba(X_row)[0, 1])

        st.session_state["last_inputs"] = inputs
        st.session_state["last_prob"]   = prob

    # ── Affichage du resultat ────────────────────────────────
    if "last_prob" not in st.session_state:
        return

    prob   = st.session_state["last_prob"]
    inputs = st.session_state["last_inputs"]
    X_row  = build_feature_vector(
        inputs, results["feature_names"], results["scaler"]
    )

    st.markdown("---")
    st.subheader("Resultat de l'analyse")

    col_res, col_gauge = st.columns([3, 1])
    with col_res:
        st.markdown(
            f"<h3 style='color:{risk_color(prob)};'>{risk_label(prob)}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Probabilite calibree de depart : {prob * 100:.1f} %**")
        st.progress(float(prob))

        if prob >= 0.70:
            st.error(
                "Ce profil presente des signaux forts de risque de depart. "
                "Une action RH preventive est recommandee dans les meilleurs delais."
            )
        elif prob >= 0.40:
            st.warning(
                "Niveau de risque modere. Un entretien de suivi est conseille."
            )
        else:
            st.success(
                "Le profil ne presente pas de signaux de depart imminent."
            )

    with col_gauge:
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={"aspect": "equal"})
        theta = np.linspace(0, np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), color="#E0E0E0", lw=12)
        ax.plot(
            np.cos(theta[:int(prob * 300)]),
            np.sin(theta[:int(prob * 300)]),
            color=risk_color(prob), lw=12
        )
        ax.text(0, -0.15, f"{prob * 100:.0f}%", ha="center", va="center",
                fontsize=24, fontweight="bold", color=risk_color(prob))
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.4, 1.3)
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

    # ── Explication SHAP individuelle ────────────────────────
    st.subheader("Facteurs determinants — Explication SHAP")
    try:
        shap_row = results["explainer"].shap_values(X_row, check_additivity=False)
        if isinstance(shap_row, list):
            sv_row = shap_row[1][0]
        elif shap_row.ndim == 3:
            sv_row = shap_row[0, :, 1]
        else:
            sv_row = shap_row[0]

        exp = shap.Explanation(
            values=sv_row,
            base_values=results["expected_value"],
            data=X_row.values[0],
            feature_names=results["feature_names"],
        )
        fig_w, _ = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(exp, max_display=15, show=False)
        plt.title("Contribution des variables a la prediction")
        plt.tight_layout()
        st.pyplot(fig_w)
        plt.close()

        shap_df = (
            pd.DataFrame({"Variable": results["feature_names"], "SHAP": sv_row})
            .sort_values("SHAP", key=abs, ascending=False)
            .head(10)
        )
        shap_df["Direction"] = shap_df["SHAP"].apply(
            lambda v: "Augmente le risque" if v > 0 else "Reduit le risque"
        )
        st.dataframe(
            shap_df[["Variable", "SHAP", "Direction"]].style.format({"SHAP": "{:.4f}"}),
            width="stretch",
        )
    except Exception as e:
        st.warning(f"Explication SHAP indisponible : {e}")

    # ── Simulateur What-If ───────────────────────────────────
    st.markdown("---")
    st.subheader("Simulateur d'actions correctives")
    st.markdown(
        "Modifiez les parametres ci-dessous pour estimer l'impact "
        "d'interventions RH sur la probabilite de depart."
    )

    base_salary      = inputs["Salary"]
    base_engagement  = float(inputs["EngagementSurvey"])
    base_absences    = int(inputs["Absences"])
    base_satisfaction = int(inputs["EmpSatisfaction"])

    ws1, ws2, ws3, ws4 = st.columns(4)
    with ws1:
        delta_salary = st.slider(
            "Ajustement salarial ($)", -20000, 40000, 0, step=1000,
            key="wi_salary",
            help="Impact d'une revalorisation ou reduction de salaire"
        )
    with ws2:
        sim_engagement = st.slider(
            "Engagement simule", 1.0, 5.0, base_engagement, step=0.1,
            key="wi_engagement",
            help="Score d'engagement apres programme de suivi"
        )
    with ws3:
        sim_absences = st.slider(
            "Absences simulees (j/an)", 0, 20, base_absences,
            key="wi_absences",
            help="Nombre d'absences apres intervention"
        )
    with ws4:
        sim_satisfaction = st.slider(
            "Satisfaction simulee (1-5)", 1, 5, base_satisfaction,
            key="wi_satisfaction",
            help="Satisfaction apres mesures d'amelioration"
        )

    sim_inputs = {
        **inputs,
        "Salary":          base_salary + delta_salary,
        "EngagementSurvey": sim_engagement,
        "Absences":        sim_absences,
        "EmpSatisfaction": sim_satisfaction,
    }
    sim_row  = build_feature_vector(
        sim_inputs, results["feature_names"], results["scaler"]
    )
    sim_prob = float(results["calib_model"].predict_proba(sim_row)[0, 1])
    delta    = sim_prob - prob

    mc1, mc2, mc3 = st.columns([1, 1, 2])
    with mc1:
        st.metric(
            "Risque initial",
            f"{prob * 100:.1f} %",
        )
    with mc2:
        st.metric(
            "Risque simule",
            f"{sim_prob * 100:.1f} %",
            delta=f"{delta * 100:+.1f} pp",
            delta_color="inverse",
        )
    with mc3:
        labels = ["Profil actuel", "Profil simule"]
        values = [prob, sim_prob]
        colors = [risk_color(prob), risk_color(sim_prob)]
        fig, ax = plt.subplots(figsize=(5, 2))
        bars = ax.barh(labels, values, color=colors, height=0.5)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title("Comparaison des risques")
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val * 100:.1f}%", va="center", fontsize=11, fontweight="bold")
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    if delta < -0.05:
        st.success(
            f"Ces ajustements reduiraient le risque de depart "
            f"de {abs(delta) * 100:.1f} points de pourcentage."
        )
    elif delta > 0.05:
        st.error(
            f"Ces parametres augmenteraient le risque de depart "
            f"de {delta * 100:.1f} points de pourcentage."
        )


# ──────────────────────────────────────────────────────────────
# ONGLET 3 — EXPLICATIONS IA (SHAP)
# ──────────────────────────────────────────────────────────────

def tab_xai(results: dict):
    st.header("Intelligence Artificielle Explicable")

    st.markdown("""
**SHAP (SHapley Additive exPlanations)** quantifie la contribution de chaque variable
a la prediction du modele, en se basant sur la theorie des jeux cooperatifs (Shapley, 1953).

- Une valeur SHAP **positive** augmente la probabilite de demission.
- Une valeur SHAP **negative** la reduit.
- La magnitude reflete l'importance relative de la variable.
    """)

    st.subheader("Importance globale des variables (SHAP)")
    c_bee, c_bar = st.columns(2)

    with c_bee:
        fig, _ = plt.subplots(figsize=(8, 7))
        shap.summary_plot(
            results["sv_class1"], results["X_test"],
            feature_names=results["feature_names"], show=False, max_display=15
        )
        plt.title("Impact des variables — Beeswarm")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c_bar:
        mean_abs = np.abs(results["sv_class1"]).mean(axis=0)
        fi_df = (
            pd.DataFrame({"Variable": results["feature_names"], "SHAP moyen": mean_abs})
            .sort_values("SHAP moyen", ascending=True)
            .tail(15)
        )
        fig, ax = plt.subplots(figsize=(8, 7))
        fi_df.set_index("Variable")["SHAP moyen"].plot(
            kind="barh", ax=ax, color=CLR_PRIMARY
        )
        ax.set_title("Importance moyenne (|SHAP|)")
        ax.set_xlabel("|SHAP| moyen")
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature importance RF (Gini)
    st.subheader("Importance des variables — Critere de Gini (Random Forest)")
    fi_rf = (
        pd.Series(results["model"].feature_importances_, index=results["feature_names"])
        .sort_values(ascending=True)
        .tail(15)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    fi_rf.plot(kind="barh", ax=ax, color=CLR_SECONDARY)
    ax.set_title("Top 15 variables — Importance Gini")
    ax.set_xlabel("Importance")
    sns.despine(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Synthese des 5 principaux facteurs de risque")
    top5 = sorted(
        zip(results["feature_names"], mean_abs), key=lambda x: x[1], reverse=True
    )[:5]
    for i, (feat, val) in enumerate(top5, 1):
        st.markdown(f"**{i}.** `{feat}` — contribution SHAP moyenne : `{val:.4f}`")


# ──────────────────────────────────────────────────────────────
# ONGLET 4 — ANALYSE DES BIAIS (FAIRNESS)
# ──────────────────────────────────────────────────────────────

def tab_fairness(results: dict):
    st.header("Analyse des Biais Algorithmiques")

    st.markdown("""
L'equite algorithmique (fairness) consiste a verifier que le modele ne desavantage
pas certains groupes proteges. Un biais dans les predictions pourrait entraîner
des decisions RH discriminatoires, en violation du **RGPD Art. 22** et de l'**AI Act UE**.

Metriques examinees :
- **Taux de faux positifs (Tx. FP)** : employes actifs incorrectement classes a risque.
- **Taux de faux negatifs (Tx. FN)** : departs non detectes par le modele.
- **Equite** : ces taux doivent etre similaires entre groupes pour eviter la discrimination.
    """)

    # ── Analyse par genre ─────────────────────────────────────
    st.subheader("Metriques par genre")
    fg = results["fairness_gender"].copy()

    if fg.empty:
        st.info("Donnees insuffisantes pour l'analyse par genre.")
    else:
        col_t, col_c = st.columns([2, 3])
        with col_t:
            def _highlight_bias(val):
                if isinstance(val, float) and val > 0.3:
                    return "background-color: #FFCCCC"
                return ""
            st.dataframe(
                fg.style
                  .format({c: "{:.3f}" for c in fg.columns if fg[c].dtype == float})
                  .map(_highlight_bias, subset=[c for c in ["Tx. FP", "Tx. FN"] if c in fg.columns]),
                width="stretch",
            )
            max_fp = fg["Tx. FP"].max() - fg["Tx. FP"].min()
            max_fn = fg["Tx. FN"].max() - fg["Tx. FN"].min()
            if max_fp > 0.10 or max_fn > 0.10:
                st.warning(
                    f"Ecart notable detecte — Tx. FP : {max_fp:.3f} | Tx. FN : {max_fn:.3f}. "
                    "Une revue humaine est recommandee avant tout usage decisionnaire."
                )
            else:
                st.success("Les metriques par genre sont homogenes (ecart < 10 %).")

        with col_c:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            metrics = ["Tx. FP", "Tx. FN"]
            colors  = [CLR_HIGH, CLR_MED]
            for ax, m, col in zip(axes, metrics, colors):
                ax.bar(fg["Genre"], fg[m], color=col)
                ax.set_title(m)
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
                for i, v in enumerate(fg[m]):
                    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
                sns.despine(ax=ax)
            plt.suptitle("Taux d'erreur par genre", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Analyse par departement ───────────────────────────────
    st.subheader("Metriques par departement")
    fd = results["fairness_dept"].copy()

    if fd.empty:
        st.info("Donnees insuffisantes pour l'analyse par departement.")
    else:
        st.dataframe(
            fd.style.format({c: "{:.3f}" for c in fd.columns if fd[c].dtype == float}),
            width="stretch",
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(fd))
        w = 0.35
        ax.bar(x - w/2, fd["Tx. FP"], w, label="Taux FP", color=CLR_HIGH)
        ax.bar(x + w/2, fd["Tx. FN"], w, label="Taux FN", color=CLR_MED)
        ax.set_xticks(x)
        ax.set_xticklabels(fd["Département"], rotation=25, ha="right")
        ax.set_title("Taux d'erreur par departement")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend()
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Recommandations ───────────────────────────────────────
    st.markdown("---")
    st.subheader("Recommandations pour la reduction des biais")
    st.markdown("""
1. **Reequilibrage des donnees** : verifier la representativite de chaque groupe dans le jeu d'apprentissage.
2. **Contraintes d'equite** : integrer une penalite d'equite lors de l'entrainement (ex. `fairlearn.reductions`).
3. **Revue humaine obligatoire** pour tout employe classe a risque dans un groupe sous-represente.
4. **Audit periodique** : re-evaluer les metriques de biais a chaque re-entrainement du modele.
5. **Transparence** : documenter les metriques d'equite dans la fiche technique du modele (AI Act Art. 11).
    """)


# ──────────────────────────────────────────────────────────────
# ONGLET 5 — SECURITE & RGPD
# ──────────────────────────────────────────────────────────────

def tab_security():
    st.header("Securite et Conformite RGPD")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mesures implementees")
        st.success(
            "Pseudonymisation des donnees\n"
            "Noms, identifiants, dates de naissance et codes postaux supprimes.\n\n"
            "Suppression des variables a fuite\n"
            "TermReason, EmploymentStatus et EmpStatusID exclus (post-evenement).\n\n"
            "Droit a l'explication — RGPD Art. 22\n"
            "Chaque prediction est accompagnee d'une explication SHAP individuelle.\n\n"
            "Calibration des probabilites\n"
            "Scores interpretables comme de vraies probabilites (modele sigmoid).\n\n"
            "Split stratifie train/test\n"
            "Aucune donnee de test utilisee a l'entrainement.\n\n"
            "Audit d'equite\n"
            "Metriques de biais calculees par genre et departement."
        )

    with col2:
        st.subheader("Recommandations operationnelles")
        st.warning(
            "Chiffrement des donnees\n"
            "AES-256 au repos, TLS 1.3 en transit.\n\n"
            "Controle d'acces (RBAC)\n"
            "Acces au dashboard limite aux responsables RH habilites.\n\n"
            "Journal d'audit\n"
            "Journalisation de chaque prediction (qui, quand, resultat).\n\n"
            "Tests adversariaux\n"
            "Verifier la robustesse du modele face aux entrees manipulees.\n\n"
            "Duree de conservation\n"
            "Politique de suppression conforme au RGPD (droit a l'oubli).\n\n"
            "Revue humaine obligatoire\n"
            "Toute decision impactante doit etre validee par un humain."
        )

    st.markdown("---")
    st.subheader("Variables exclues du modele")
    excl = pd.DataFrame({
        "Variable": [
            "Employee_Name", "EmpID", "DOB", "Zip", "ManagerName",
            "TermReason", "EmploymentStatus", "EmpStatusID", "DateofTermination",
        ],
        "Raison": [
            "Identifiant personnel (PII)",
            "Identifiant unique (PII)",
            "Date de naissance — remplacee par Age",
            "Code postal trop specifique (PII)",
            "Identifiant personnel (PII)",
            "Variable post-evenement (fuite de donnees)",
            "Encode la variable cible (fuite)",
            "Code de statut d'emploi (fuite)",
            "Date de licenciement (fuite + PII)",
        ],
        "Action": [
            "Supprimee", "Supprimee", "Transformee en Age",
            "Supprimee", "Supprimee",
            "Supprimee", "Supprimee", "Supprimee", "Supprimee",
        ],
    })
    st.dataframe(excl, width="stretch")

    st.markdown("---")
    st.subheader("Cartographie des risques IA")
    risks = [
        ("Biais algorithmique",    "Reproduction des biais historiques du jeu de donnees.", "Eleve"),
        ("Manipulation des entrees", "Modification intentionnelle des donnees pour biaiser les predictions.", "Modere"),
        ("Extraction du modele",   "Requetes repetees pour reconstituer le modele (model stealing).", "Modere"),
        ("Surconfiance des RH",    "Ignorer le contexte humain en faveur du score de l'IA.", "Eleve"),
        ("Fuite via SHAP",         "Les explications peuvent reveler indirectement des donnees sensibles.", "Faible"),
    ]
    risk_df = pd.DataFrame(risks, columns=["Risque", "Description", "Niveau"])

    def _color(v):
        if v == "Eleve":   return "background-color: #ffcccc"
        if v == "Modere":  return "background-color: #ffe0b2"
        return "background-color: #c8e6c9"

    st.dataframe(
        risk_df.style.map(_color, subset=["Niveau"]),
        width="stretch",
    )


# ──────────────────────────────────────────────────────────────
# ONGLET 6 — AUTO-ÉVALUATION EMPLOYÉ
# ──────────────────────────────────────────────────────────────

def analyze_text_sentiment(text: str) -> dict:
    """Analyse le sentiment et extrait les thèmes du texte employé."""
    text_lower = str(text).lower()
    
    # Keywords par thème
    salary_keywords = ["compensation", "salaire", "reconnaissance", "augment", "rémunération", "revalu"]
    growth_keywords = ["croissance", "carrière", "développement", "progression", "opportunité", "responsabilité", "avancement"]
    stress_keywords = ["charge", "pression", "équilibre", "surmenage", "fatigue", "stress", "workload", "pace"]
    mobility_keywords = ["interne", "mobilité", "changement", "autre équipe", "autre rôle", "transfer"]
    positive_keywords = ["apprécie", "positif", "stable", "clair", "valeur", "satisfait", "content", "bien"]
    negative_keywords = ["pas assez", "difficile", "pression", "dur", "mieux", "manque", "insatisfait", "mauvais"]
    
    # Comptage des keywords
    salary_score = sum(1 for k in salary_keywords if k in text_lower)
    growth_score = sum(1 for k in growth_keywords if k in text_lower)
    stress_score = sum(1 for k in stress_keywords if k in text_lower)
    mobility_score = sum(1 for k in mobility_keywords if k in text_lower)
    pos_score = sum(1 for k in positive_keywords if k in text_lower)
    neg_score = sum(1 for k in negative_keywords if k in text_lower)
    
    sentiment = pos_score - neg_score
    
    return {
        "sentiment_score": sentiment,
        "topic_salary": min(salary_score, 1),
        "topic_growth": min(growth_score, 1),
        "topic_stress": min(stress_score, 1),
        "topic_mobility": min(mobility_score, 1),
        "negative_text_flag": 1 if sentiment < 0 else 0,
    }


def calculate_composite_risk(inputs: dict, text_analysis: dict, looking_for: str, results: dict) -> tuple[float, str]:
    """
    Calcule un score de risque COMPOSITE basé sur :
    - Données métier directes (absence, satisfaction, engagement)
    - Analyse NLP (sentiment, thèmes)
    - Prédiction du modèle ML
    
    Retourne : (score_final, explication)
    """
    
    # ─ 1. SCORE MÉTIER (0-1) ─────────────────────────────────
    # Absences élevées = risque
    absences_score = min(inputs.get("Absences", 0) / 20, 1.0)  # 20 jours = 100% risque
    
    # Satisfaction/Engagement faibles = risque
    satisfaction_score = 1 - (inputs.get("EmpSatisfaction", 3) / 5)  # Inverse : 5 = 0% risque
    engagement_score = 1 - (inputs.get("EngagementSurvey", 4) / 5)
    
    # Peu de projets = risque (stagnation)
    projects_score = 1 - min(inputs.get("SpecialProjectsCount", 0) / 5, 1.0)
    
    # Tenure très court = risque (phase d'intégration mal passée)
    tenure = inputs.get("Tenure", 5)
    tenure_score = 1 - min(tenure / 5, 1.0)  # 0-5 ans = risque, 5+ = pas de risque tenure
    
    # Score métier moyen pondéré
    metric_score = (
        absences_score * 0.25 +
        satisfaction_score * 0.30 +
        engagement_score * 0.25 +
        projects_score * 0.10 +
        tenure_score * 0.10
    )
    
    # ─ 2. SCORE NLP (0-1) ────────────────────────────────────
    # Sentiment négatif = risque
    sentiment_risk = max(-text_analysis["sentiment_score"] / 3, 0)  # Plus de -3, c'est 100% risque
    sentiment_risk = min(sentiment_risk, 1.0)
    
    # Thèmes problématiques
    problematic_themes = (
        text_analysis["topic_salary"] * 0.30 +
        text_analysis["topic_stress"] * 0.35 +
        text_analysis["topic_mobility"] * 0.25 +
        text_analysis["negative_text_flag"] * 0.10
    )
    
    nlp_score = (sentiment_risk * 0.5 + problematic_themes * 0.5)
    nlp_score = min(nlp_score, 1.0)
    
    # ─ 3. BONUS/MALUS RECHERCHE D'EMPLOI ──────────────────
    looking_bonus = {
        "Non, je suis bien ici": -0.20,
        "Peut-être, dépend des opportunités": 0.0,
        "Oui, activement": 0.35
    }
    looking_delta = looking_bonus.get(looking_for, 0)
    
    # ─ 4. PRÉDICTION MODÈLE (si disponible) ──────────────
    try:
        X_row_model = build_feature_vector(
            inputs, results["feature_names"], results["scaler"]
        )
        model_score = float(results["calib_model"].predict_proba(X_row_model)[0, 1])
    except Exception:
        model_score = 0.5  # Valeur neutre en cas d'erreur
    
    # ─ 5. SCORE FINAL (COMPOSITE) ───────────────────────────
    # Combinaison pondérée
    composite_score = (
        metric_score * 0.35 +     # Données métier
        nlp_score * 0.35 +        # Analyse texte
        model_score * 0.30        # Modèle ML
    )
    
    # Appliquer le bonus/malus
    composite_score += looking_delta
    composite_score = np.clip(composite_score, 0, 1)
    
    # Déterminer l'explication
    risk_factors = []
    if absences_score > 0.6:
        risk_factors.append("Absences élevées")
    if satisfaction_score > 0.5:
        risk_factors.append("Satisfaction faible")
    if engagement_score > 0.5:
        risk_factors.append("Engagement faible")
    if text_analysis["topic_salary"]:
        risk_factors.append("Préoccupations salariales")
    if text_analysis["topic_stress"]:
        risk_factors.append("Stress / surcharge")
    if text_analysis["topic_mobility"]:
        risk_factors.append("Intérêt pour mobilité")
    if text_analysis["sentiment_score"] < -1:
        risk_factors.append("Ton négatif dans les commentaires")
    
    explanation = " | ".join(risk_factors) if risk_factors else "Peu de signaux de risque détectés"
    
    return composite_score, explanation


def tab_employee_self_assessment(results: dict):
    st.header("Auto-évaluation Employé")
    st.markdown(
        "Formulaire confidentiel destiné aux employés.  \n"
        "Remplissez ce formulaire pour que nous puissions mieux comprendre votre situation "
        "et vos attentes. Vos réponses resteront confidentielles."
    )
    
    with st.form("employee_self_form", border=True):
        st.subheader("Informations personnelles")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            dept_self = st.selectbox(
                "Votre département",
                ["Production", "IT/IS", "Sales", "Software Engineering",
                 "Admin Offices", "Executive Office"],
                key="es_dept"
            )
        with c2:
            tenure_self = st.slider("Ancienneté (années)", 0, 25, 3, key="es_tenure")
        with c3:
            satisfaction_self = st.slider("Satisfaction générale (1-5)", 1, 5, 3, key="es_satisfaction")
        
        st.subheader("Vos commentaires")
        comment1 = st.text_area(
            "Comment vous sentez-vous dans votre rôle actuel ? (Ce qui vous plaît, ce qui pose problème)",
            height=80, key="es_comment1",
            placeholder="Décrivez votre expérience..."
        )
        
        comment2 = st.text_area(
            "Quelles sont vos attentes pour l'avenir professionnellement ?",
            height=80, key="es_comment2",
            placeholder="Parlez de vos aspirations, carrière, développement..."
        )
        
        comment3 = st.text_area(
            "Y a-t-il des changements qui vous aideraient à rester/être plus engagé(e) ?",
            height=80, key="es_comment3",
            placeholder="Actions correctives, conditions, environnement..."
        )
        
        # Indicateurs de présence
        st.subheader("Situation actuelle")
        es1, es2, es3, es4 = st.columns(4)
        with es1:
            absences_self = st.slider("Jours d'absence (annuels)", 0, 20, 8, key="es_absences")
        with es2:
            engagement_self = st.slider("Score d'engagement ressenti", 1.0, 5.0, 3.5, step=0.1, key="es_engagement")
        with es3:
            projects_self = st.slider("Projets spéciaux réalisés", 0, 8, 2, key="es_projects")
        with es4:
            lookingfor = st.selectbox(
                "Cherchez-vous un autre rôle ?",
                ["Non, je suis bien ici", "Peut-être, dépend des opportunités", "Oui, activement"],
                key="es_looking"
            )
        
        submitted_self = st.form_submit_button("Envoyer ma réponse", use_container_width=True)
    
    if submitted_self:
        # Validation : au moins un commentaire
        if not comment1 and not comment2 and not comment3:
            st.error("Veuillez remplir au moins un champ de commentaire.")
            return
        
        # Concaténer et analyser le texte
        full_text = f"{comment1} {comment2} {comment3}"
        text_analysis = analyze_text_sentiment(full_text)
        
        # Bonus/malus selon la recherche d'emploi
        looking_bonus = {"Non, je suis bien ici": -0.15, 
                        "Peut-être, dépend des opportunités": 0.0,
                        "Oui, activement": 0.25}
        looking_delta = looking_bonus.get(lookingfor, 0)
        
        # Construire les inputs
        inputs_self = {
            "Department": dept_self,
            "Sex": "M",
            "Married": False,
            "DiversityHire": False,
            "Age": 35,
            "Tenure": tenure_self,
            "Salary": 62000,
            "PerformanceScore": "Fully Meets",
            "RecruitmentSource": "Employee Referral",
            "EngagementSurvey": engagement_self,
            "EmpSatisfaction": satisfaction_self,
            "Absences": absences_self,
            "DaysLateLast30": 0,
            "SpecialProjectsCount": projects_self,
        }
        
        # Calcul du score COMPOSITE (métier + NLP + modèle)
        prob_self, risk_explanation = calculate_composite_risk(
            inputs_self, text_analysis, lookingfor, results
        )
        
        # Déterminer le statut d'invitation
        if prob_self >= 0.70:
            invite_status = "URGENT"
        elif prob_self >= 0.40:
            invite_status = "RECOMMANDE"
        else:
            invite_status = "STABLE"
        
        # Sauvegarder le résultat
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        eval_record = {
            "Timestamp": timestamp,
            "Department": dept_self,
            "Anciennete": tenure_self,
            "Satisfaction": satisfaction_self,
            "Absences": absences_self,
            "Engagement": engagement_self,
            "Projets": projects_self,
            "Recherche_Emploi": lookingfor,
            "Topic_Salaire": text_analysis["topic_salary"],
            "Topic_Carriere": text_analysis["topic_growth"],
            "Topic_Stress": text_analysis["topic_stress"],
            "Topic_Mobilite": text_analysis["topic_mobility"],
            "Sentiment_Score": text_analysis["sentiment_score"],
            "Risque_Score": prob_self,
            "Status_Invitation": invite_status,
            "Explication_Risque": risk_explanation,
        }
        
        # Charger ou créer le CSV
        try:
            df_evals = pd.read_csv(SELF_EVAL_PATH)
        except FileNotFoundError:
            df_evals = pd.DataFrame()
        
        df_evals = pd.concat([df_evals, pd.DataFrame([eval_record])], ignore_index=True)
        df_evals.to_csv(SELF_EVAL_PATH, index=False)
        
        # Message de remerciement UNIQUEMENT
        st.markdown("---")
        st.success(
            "✅ **Merci d'avoir rempli ce formulaire !**\n\n"
            "Vos réponses ont été enregistrées de manière confidentielle. "
            "Les responsables RH examineront votre évaluation et vous contacteront si nécessaire. "
            "Nous apprécions votre transparence et votre implication."
        )
        st.balloons()


# ──────────────────────────────────────────────────────────────
# ONGLET 7 — RÉSULTATS AUTO-ÉVALUATIONS (RH UNIQUEMENT)
# ──────────────────────────────────────────────────────────────

def tab_self_eval_results():
    st.header("Résultats Auto-évaluations Employés (RH)")
    st.markdown(
        "Tableau de bord confidentiel — Résultats des auto-évaluations des employés  \n"
        "**Accès réservé aux responsables RH.**"
    )
    
    # Charger les résultats
    try:
        df_evals = pd.read_csv(SELF_EVAL_PATH)
    except FileNotFoundError:
        st.info("Aucune auto-évaluation enregistrée pour le moment.")
        return
    
    if df_evals.empty:
        st.info("Aucune auto-évaluation enregistrée pour le moment.")
        return
    
    # Ajouter la colonne Explication_Risque si elle n'existe pas (rétrocompatibilité)
    if "Explication_Risque" not in df_evals.columns:
        df_evals["Explication_Risque"] = "Non disponible"
    
    st.markdown("---")
    
    # Statistiques globales
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("Total évaluations", len(df_evals))
    
    with col_stats2:
        urgent = (df_evals["Status_Invitation"] == "URGENT").sum()
        st.metric("À inviter (URGENT)", urgent, delta=urgent)
    
    with col_stats3:
        recommande = (df_evals["Status_Invitation"] == "RECOMMANDE").sum()
        st.metric("À suivre (RECOMMANDÉ)", recommande)
    
    with col_stats4:
        stable = (df_evals["Status_Invitation"] == "STABLE").sum()
        st.metric("Profils stables", stable)
    
    st.markdown("---")
    
    # Tableau des résultats avec filtrage
    st.subheader("Détail des évaluations")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        status_filter = st.multiselect(
            "Filtrer par statut",
            options=["URGENT", "RECOMMANDE", "STABLE"],
            default=["URGENT", "RECOMMANDE", "STABLE"]
        )
    with filter_col2:
        dept_filter = st.multiselect(
            "Filtrer par département",
            options=df_evals["Department"].unique(),
            default=df_evals["Department"].unique()
        )
    with filter_col3:
        risk_range = st.slider(
            "Plage de risque (%)",
            0, 100, (0, 100),
            step=5
        )
    
    # Appliquer les filtres
    df_filtered = df_evals[
        (df_evals["Status_Invitation"].isin(status_filter)) &
        (df_evals["Department"].isin(dept_filter)) &
        (df_evals["Risque_Score"] >= risk_range[0]/100) &
        (df_evals["Risque_Score"] <= risk_range[1]/100)
    ].copy()
    
    st.info(f"**{len(df_filtered)} évaluation(s) correspondent aux critères**")
    
    # Afficher le tableau
    display_cols = [
        "Timestamp", "Department", "Anciennete", "Satisfaction", 
        "Engagement", "Risque_Score", "Status_Invitation", 
        "Explication_Risque", "Topic_Salaire", "Topic_Carriere", "Topic_Stress", "Topic_Mobilite"
    ]
    
    # Garder seulement les colonnes qui existent
    display_cols = [c for c in display_cols if c in df_filtered.columns]
    
    if not display_cols:
        st.error("Aucune donnée à afficher.")
        return
    
    df_display = df_filtered[display_cols].copy()
    df_display["Risque_Score"] = df_display["Risque_Score"].apply(lambda x: f"{x*100:.1f}%")
    
    # Colorier les status
    def color_status(val):
        if val == "URGENT":
            return f"background-color: {CLR_HIGH}; color: white; font-weight: bold"
        elif val == "RECOMMANDE":
            return f"background-color: {CLR_MED}; color: white; font-weight: bold"
        else:
            return f"background-color: {CLR_LOW}; color: white; font-weight: bold"
    
    st.dataframe(
        df_display.style.map(color_status, subset=["Status_Invitation"]),
        use_container_width=True,
        height=400
    )
    
    st.markdown("---")
    
    # Visualisations
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("Distribution des risques")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        bins = np.linspace(0, 1, 11)
        ax.hist(df_evals["Risque_Score"], bins=bins, color=CLR_PRIMARY, alpha=0.7, edgecolor="black")
        ax.axvline(df_evals["Risque_Score"].mean(), color=CLR_HIGH, linestyle="--", linewidth=2, label=f"Moyenne: {df_evals['Risque_Score'].mean():.2f}")
        ax.set_xlabel("Score de risque")
        ax.set_ylabel("Nombre d'employés")
        ax.set_title("Distribution des scores de risque")
        ax.legend()
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col_viz2:
        st.subheader("Thèmes détectés")
        themes_data = {
            "Salaire": df_evals["Topic_Salaire"].sum(),
            "Carrière": df_evals["Topic_Carriere"].sum(),
            "Stress": df_evals["Topic_Stress"].sum(),
            "Mobilité": df_evals["Topic_Mobilite"].sum(),
        }
        themes_data = {k: v for k, v in sorted(themes_data.items(), key=lambda x: x[1], reverse=True)}
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(list(themes_data.keys()), list(themes_data.values()), color=CLR_PRIMARY)
        ax.set_xlabel("Nombre de mentions")
        ax.set_title("Thèmes majeurs detects")
        for i, v in enumerate(themes_data.values()):
            ax.text(v + 0.1, i, str(int(v)), va="center", fontsize=10)
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Export
    st.markdown("---")
    st.subheader("Exporter les données")
    
    csv_buffer = df_evals.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger tous les résultats (CSV)",
        data=csv_buffer,
        file_name=f"auto_evaluations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    X, y, df_raw = load_and_prepare()
    results = train_model(str(list(X.columns)))
    render_sidebar(results, df_raw)

    st.title("HR Turnover Predictor — Analyse Predictive et IA Explicable")
    st.markdown(
        "Systeme de prediction des departs d'employes base sur le Machine Learning, "
        "la calibration probabiliste et l'IA Explicable (SHAP).  \n"
        "**Toutes les donnees personnelles ont ete anonymisees — Conformite RGPD.**"
    )
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Vue Globale",
        "Prediction",
        "Explications IA",
        "Analyse des Biais",
        "Securite & RGPD",
        "Auto-evaluation Employe",
        "Resultats Auto-eval (RH)",
    ])

    with tab1:
        tab_overview(df_raw, results)
    with tab2:
        tab_prediction(results)
    with tab3:
        tab_xai(results)
    with tab4:
        tab_fairness(results)
    with tab5:
        tab_security()
    with tab6:
        tab_employee_self_assessment(results)
    with tab7:
        tab_self_eval_results()


main()
