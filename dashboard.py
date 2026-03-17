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
    row = pd.Series(scaler.mean_, index=feature_names)

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
    tenure_safe = max(inputs.get("Tenure", 1), 1)
    numeric_map["AbsenteeismRate"]      = inputs.get("Absences", 10) / tenure_safe
    numeric_map["RiskScore_Engagement"] = (
        inputs.get("DaysLateLast30", 0) * 2 + inputs.get("Absences", 10)
    )
    for col, val in numeric_map.items():
        if col in row.index:
            row[col] = val

    for prefix in ["Department", "Position", "State", "Sex", "MaritalDesc",
                   "CitizenDesc", "HispanicLatino", "RaceDesc",
                   "RecruitmentSource", "PerformanceScore"]:
        for c in [c for c in feature_names if c.startswith(f"{prefix}_")]:
            row[c] = 0

    for prefix, key in [
        ("Department",        "Department"),
        ("PerformanceScore",  "PerformanceScore"),
        ("RecruitmentSource", "RecruitmentSource"),
        ("Sex",               "Sex"),
    ]:
        col = f"{prefix}_{inputs.get(key, '')}"
        if col in row.index:
            row[col] = 1

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Vue Globale",
        "Prediction",
        "Explications IA",
        "Analyse des Biais",
        "Securite & RGPD",
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


main()
