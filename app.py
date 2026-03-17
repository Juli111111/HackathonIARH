# -*- coding: utf-8 -*-

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="HR Turnover Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH = "HRDataset_v14.csv"
MODELS_DIR = "models"

CLR_HIGH = "#C62828"
CLR_MED = "#E65100"
CLR_LOW = "#2E7D32"
CLR_PRIMARY = "#1565C0"
CLR_TEXT = "#101828"
CLR_MUTED = "#667085"


# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main {
        background-color: #F8FAFC;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1300px;
    }
    .soft-card {
        background: white;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        border: 1px solid #E6EAF0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 0.8rem;
        color: #101828;
    }
    .risk-card {
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        color: white;
        margin-bottom: 0.8rem;
    }
    .action-card {
        background: white;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        border: 1px solid #E6EAF0;
        border-left: 6px solid #1565C0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 0.8rem;
        color: #101828;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        color: #101828;
    }
    .small-muted {
        color: #667085;
        font-size: 0.92rem;
    }
    .metric-title {
        color: #667085;
        font-size: 0.85rem;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #101828;
    }
    .pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "Risque élevé"
    if prob >= 0.40:
        return "Risque modéré"
    return "Risque faible"


def risk_color(prob: float) -> str:
    if prob >= 0.70:
        return CLR_HIGH
    if prob >= 0.40:
        return CLR_MED
    return CLR_LOW


def sentiment_score(text: str) -> int:
    text = str(text).lower()
    positive_keywords = ["appreciate", "positive", "stable", "clear", "value", "satisfied"]
    negative_keywords = ["not fully", "would like more", "difficult", "pressure", "hard", "better"]
    pos = sum(1 for k in positive_keywords if k in text)
    neg = sum(1 for k in negative_keywords if k in text)
    return pos - neg


def contains_any(text: str, keywords: list[str]) -> int:
    text = str(text).lower()
    return int(any(k in text for k in keywords))


def render_metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pill(text: str, bg: str, color: str = "#101828"):
    st.markdown(
        f"""<span class="pill" style="background:{bg}; color:{color};">{text}</span>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    results = joblib.load(os.path.join(MODELS_DIR, "results_summary.pkl"))
    return best_model, feature_names, results


def enrich_with_nlp(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_nlp = X.copy()

    if "Salary" in X_nlp.columns:
        X_nlp["Salary"] = pd.to_numeric(X_nlp["Salary"], errors="coerce")
        X_nlp["salary_pct"] = X_nlp["Salary"].rank(pct=True)
    else:
        X_nlp["salary_pct"] = 0.5

    if "EngagementSurvey" in X_nlp.columns:
        X_nlp["EngagementSurvey"] = pd.to_numeric(X_nlp["EngagementSurvey"], errors="coerce")
        X_nlp["low_engagement_flag"] = (X_nlp["EngagementSurvey"] < X_nlp["EngagementSurvey"].median()).astype(int)
    else:
        X_nlp["low_engagement_flag"] = 0

    if "Absences" in X_nlp.columns:
        X_nlp["Absences"] = pd.to_numeric(X_nlp["Absences"], errors="coerce")
        X_nlp["high_absence_flag"] = (X_nlp["Absences"] >= X_nlp["Absences"].quantile(0.75)).astype(int)
    else:
        X_nlp["high_absence_flag"] = 0

    if "Tenure" in X_nlp.columns:
        X_nlp["Tenure"] = pd.to_numeric(X_nlp["Tenure"], errors="coerce")
        X_nlp["high_tenure_flag"] = (X_nlp["Tenure"] >= X_nlp["Tenure"].median()).astype(int)
    else:
        X_nlp["high_tenure_flag"] = 0

    if "SpecialProjectsCount" in X_nlp.columns:
        X_nlp["SpecialProjectsCount"] = pd.to_numeric(X_nlp["SpecialProjectsCount"], errors="coerce")
        X_nlp["low_projects_flag"] = (X_nlp["SpecialProjectsCount"] <= X_nlp["SpecialProjectsCount"].median()).astype(int)
    else:
        X_nlp["low_projects_flag"] = 0

    perf_col = None
    for c in ["PerfScoreID", "PerformanceScore"]:
        if c in X_nlp.columns:
            perf_col = c
            break

    if perf_col is not None:
        X_nlp[perf_col] = pd.to_numeric(X_nlp[perf_col], errors="coerce")
        X_nlp["high_perf_flag"] = (X_nlp[perf_col] >= X_nlp[perf_col].median()).astype(int)
    else:
        X_nlp["high_perf_flag"] = 0

    X_nlp["low_salary_flag"] = (X_nlp["salary_pct"] <= 0.35).astype(int)

    positive_survey = [
        "I appreciate the team environment and my responsibilities are clear. I feel stable in my current role.",
        "My work environment is positive and I value the collaboration within the team. I would like to continue developing my skills.",
        "I am generally satisfied with my role and the organization. Communication and expectations are mostly clear."
    ]
    growth_survey = [
        "My current role is stable overall, but I would like more clarity on future opportunities and development.",
        "I would appreciate better visibility on progression and responsibilities.",
        "I think there is room to improve communication about career development."
    ]
    salary_survey = [
        "I feel my compensation and career progression are not fully aligned with my contribution.",
        "I would like more visibility on salary progression and future growth.",
        "My responsibilities have grown, but recognition and progression do not seem to have followed at the same pace."
    ]
    stress_survey = [
        "My workload has become difficult to manage consistently and I would appreciate more support.",
        "I have experienced pressure in my day-to-day work and I would like better balance.",
        "The current pace is sometimes hard to sustain and I would value workload adjustments."
    ]
    transfer_growth = [
        "I would like to explore an internal move to a role with more responsibility and clearer growth prospects.",
        "I am interested in internal mobility because I would like broader responsibilities and a stronger development path.",
        "I would appreciate discussing a transfer opportunity to continue progressing within the company."
    ]
    transfer_fit = [
        "I would like to discuss a move to another team where my skills could be better used.",
        "I am interested in an internal transfer to a role that better matches my professional objectives.",
        "I would like to explore internal mobility toward a position with a stronger fit for my experience."
    ]

    rng = np.random.default_rng(42)

    def choose(lst):
        return lst[rng.integers(0, len(lst))]

    def generate_survey(row):
        if row["low_salary_flag"] == 1 and row["high_perf_flag"] == 1:
            return choose(salary_survey)
        if row["high_absence_flag"] == 1 and row["low_engagement_flag"] == 1:
            return choose(stress_survey)
        if row["high_tenure_flag"] == 1 and row["low_projects_flag"] == 1:
            return choose(growth_survey)
        if row["low_engagement_flag"] == 1:
            return choose(growth_survey)
        return choose(positive_survey)

    def generate_transfer(row):
        score = (
            0.35 * row["low_salary_flag"] +
            0.25 * row["high_perf_flag"] +
            0.20 * row["high_tenure_flag"] +
            0.20 * row["low_engagement_flag"]
        )
        if rng.random() > score:
            return ""
        if row["high_tenure_flag"] == 1 or row["low_projects_flag"] == 1:
            return choose(transfer_growth)
        return choose(transfer_fit)

    demo_text_df = pd.DataFrame(index=X_nlp.index)
    demo_text_df["survey_comment"] = X_nlp.apply(generate_survey, axis=1)
    demo_text_df["transfer_request_text"] = X_nlp.apply(generate_transfer, axis=1)

    salary_keywords = ["compensation", "salary", "recognition", "progression"]
    growth_keywords = ["growth", "development", "career", "progressing", "opportunities", "responsibility"]
    stress_keywords = ["workload", "pressure", "support", "balance", "pace"]
    mobility_keywords = ["internal move", "internal mobility", "transfer", "another team", "another role"]

    text_combined = (
        demo_text_df["survey_comment"].fillna("") + " " + demo_text_df["transfer_request_text"].fillna("")
    ).str.strip()

    X_nlp["text_sentiment_score"] = text_combined.apply(sentiment_score)
    X_nlp["topic_salary"] = text_combined.apply(lambda x: contains_any(x, salary_keywords))
    X_nlp["topic_growth"] = text_combined.apply(lambda x: contains_any(x, growth_keywords))
    X_nlp["topic_stress"] = text_combined.apply(lambda x: contains_any(x, stress_keywords))
    X_nlp["topic_mobility"] = text_combined.apply(lambda x: contains_any(x, mobility_keywords))
    X_nlp["negative_text_flag"] = (X_nlp["text_sentiment_score"] < 0).astype(int)
    X_nlp["mobility_request_present"] = demo_text_df["transfer_request_text"].fillna("").str.len().gt(10).astype(int)

    return X_nlp, demo_text_df


@st.cache_data
def load_and_prepare_data(feature_names: tuple):
    df_raw = pd.read_csv(DATA_PATH)

    for col in df_raw.select_dtypes(include="object").columns:
        df_raw[col] = df_raw[col].astype(str).str.strip()

    df = df_raw.copy()

    df["DOB"] = pd.to_datetime(df["DOB"], dayfirst=False, errors="coerce")
    df["DateofHire"] = pd.to_datetime(df["DateofHire"], dayfirst=False, errors="coerce")
    df["LastPerformanceReview_Date"] = pd.to_datetime(
        df["LastPerformanceReview_Date"], dayfirst=False, errors="coerce"
    )

    ref_date = pd.Timestamp("2019-03-01")
    df["Age"] = (ref_date - df["DOB"]).dt.days // 365
    df["Tenure"] = (ref_date - df["DateofHire"]).dt.days // 365
    df["DaysSinceLastReview"] = (ref_date - df["LastPerformanceReview_Date"]).dt.days
    df["ManagerID"] = df["ManagerID"].fillna(df["ManagerID"].median())

    meta_cols = [
        "EmpID", "Department", "Position", "ManagerID", "Salary", "Age", "Tenure",
        "EngagementSurvey", "EmpSatisfaction", "Absences", "DaysLateLast30",
        "SpecialProjectsCount", "RecruitmentSource", "PerformanceScore", "State",
        "Sex", "MaritalDesc", "CitizenDesc", "HispanicLatino", "RaceDesc", "Termd"
    ]
    meta_df = df[[c for c in meta_cols if c in df.columns]].copy()

    pii_cols = [
        "Employee_Name", "EmpID", "Zip", "ManagerName",
        "DOB", "DateofHire", "LastPerformanceReview_Date"
    ]
    leaky_cols = [
        "TermReason", "EmploymentStatus", "EmpStatusID", "DateofTermination"
    ]

    df_model = df.drop(columns=pii_cols + leaky_cols, errors="ignore")

    y = df_model["Termd"].copy()
    X = df_model.drop(columns=["Termd"], errors="ignore")
    X = X.drop(columns=["DaysSinceLastReview"], errors="ignore")

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    X, demo_text_df = enrich_with_nlp(X)

    if "Absences" in X.columns and "Tenure" in X.columns:
        tenure_safe = X["Tenure"].clip(lower=1)
        X["AbsenteeismRate"] = X["Absences"] / tenure_safe

    if "DaysLateLast30" in X.columns and "Absences" in X.columns:
        X["RiskScore_Engagement"] = X["DaysLateLast30"] * 2 + X["Absences"]

    X = X.reindex(columns=list(feature_names), fill_value=0)

    meta_df = meta_df.join(demo_text_df)
    meta_df["EmpID"] = pd.to_numeric(meta_df["EmpID"], errors="coerce").astype("Int64")

    return meta_df, X, y


# ─────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────

def get_individual_shap(results: dict, X_row: pd.DataFrame):
    try_order = []
    if results["best_key"] in results:
        try_order.append((results["best_key"], results[results["best_key"]]["model"]))
    if "rf" in results and results["best_key"] != "rf":
        try_order.append(("rf", results["rf"]["model"]))
    if "xgb" in results and results["best_key"] != "xgb":
        try_order.append(("xgb", results["xgb"]["model"]))

    for model_key, model in try_order:
        try:
            explainer = shap.TreeExplainer(model, results["X_train"])
            shap_values = explainer.shap_values(X_row, check_additivity=False)

            if isinstance(shap_values, list):
                sv_row = shap_values[1][0]
            elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
                sv_row = shap_values[0, :, 1]
            else:
                sv_row = shap_values[0]

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_arr = np.atleast_1d(expected_value)
                base_val = expected_arr[1] if len(expected_arr) > 1 else expected_arr[0]
            else:
                base_val = expected_value

            model_name = "Random Forest" if model_key == "rf" else "XGBoost"
            return sv_row, base_val, model_name
        except Exception:
            continue

    return None, None, None


# ─────────────────────────────────────────────
# RECOMMANDATIONS
# ─────────────────────────────────────────────

def generate_retention_recommendations(x_row: pd.Series, shap_df: pd.DataFrame) -> list[str]:
    recs = []

    def positive(feature: str) -> bool:
        return not shap_df[(shap_df["Variable"] == feature) & (shap_df["SHAP"] > 0)].empty

    if x_row.get("low_salary_flag", 0) == 1 or x_row.get("topic_salary", 0) == 1:
        recs.append("Lancer une revue de rémunération ou clarifier rapidement la trajectoire d'évolution salariale.")

    if x_row.get("topic_growth", 0) == 1 or (x_row.get("high_tenure_flag", 0) == 1 and x_row.get("low_projects_flag", 0) == 1):
        recs.append("Prévoir un entretien de carrière pour clarifier les perspectives d'évolution, la montée en responsabilité et les prochaines étapes.")

    if x_row.get("topic_stress", 0) == 1 or x_row.get("high_absence_flag", 0) == 1 or positive("AbsenteeismRate"):
        recs.append("Vérifier la charge de travail et mettre en place un plan de soutien managérial ou un ajustement court terme.")

    if x_row.get("topic_mobility", 0) == 1 or x_row.get("mobility_request_present", 0) == 1:
        recs.append("Explorer une mobilité interne ou un changement de périmètre afin de retenir le collaborateur dans l'entreprise.")

    if x_row.get("low_engagement_flag", 0) == 1 or x_row.get("text_sentiment_score", 0) < 0:
        recs.append("Organiser rapidement un point RH avec le manager pour identifier les irritants concrets et sécuriser un plan d'action.")

    if positive("ManagerID"):
        recs.append("Analyser le contexte managérial de l'équipe et renforcer le suivi du collaborateur à court terme.")

    if not recs:
        recs.append("Maintenir un suivi régulier, avec feedback managérial, reconnaissance et échange de carrière.")

    return recs[:4]


def build_executive_summary(prob: float, top_pos: pd.DataFrame, top_neg: pd.DataFrame) -> str:
    pos_feats = top_pos["Variable"].head(3).tolist()
    neg_feats = top_neg["Variable"].head(2).tolist()

    pos_txt = ", ".join(pos_feats) if pos_feats else "aucun facteur dominant"
    neg_txt = ", ".join(neg_feats) if neg_feats else "peu de facteurs protecteurs visibles"

    return (
        f"Le modèle estime un {risk_label(prob).lower()} de départ. "
        f"Les facteurs qui augmentent le plus ce risque sont : {pos_txt}. "
        f"Les éléments qui jouent plutôt en faveur de la rétention sont : {neg_txt}."
    )


# ─────────────────────────────────────────────
# INTERFACE
# ─────────────────────────────────────────────

def render_header(results: dict):
    best_key = results["best_key"]

    st.title("HR Turnover Predictor")
    st.caption("Fiche employé et analyse IA pour la rétention RH")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Modèle retenu", results["best_name"])
    with c2:
        render_metric_card("AUC", f"{results[best_key]['auc']:.3f}")
    with c3:
        render_metric_card("F1 Démission", f"{results[best_key]['f1']:.3f}")
    with c4:
        render_metric_card("Recall Démission", f"{results[best_key]['recall']:.3f}")


def render_employee_selector(meta_df: pd.DataFrame) -> int:
    st.markdown('<div class="section-title">Sélection employé</div>', unsafe_allow_html=True)

    valid_meta = meta_df.dropna(subset=["EmpID"]).copy()
    emp_choices = valid_meta["EmpID"].astype(int).tolist()
    empid_to_idx = {int(row["EmpID"]): idx for idx, row in valid_meta.iterrows()}

    selected_empid = st.selectbox(
        "Choisir un EmpID",
        options=emp_choices,
        format_func=lambda x: f"EmpID {x}",
    )

    return empid_to_idx[int(selected_empid)]


def render_profile_tab(meta_row: pd.Series):
    st.markdown('<div class="section-title">Profil employé</div>', unsafe_allow_html=True)

    left, right = st.columns([1.0, 1.2])

    with left:
        st.markdown(
            f"""
            <div class="soft-card">
                <div class="metric-title">Identifiant</div>
                <div class="metric-value">EmpID {int(meta_row['EmpID']) if pd.notna(meta_row['EmpID']) else 'N/A'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        info_lines = []
        if "Position" in meta_row.index:
            info_lines.append(f"<b>Poste</b> : {meta_row.get('Position', 'N/A')}")
        if "Department" in meta_row.index:
            info_lines.append(f"<b>Département</b> : {meta_row.get('Department', 'N/A')}")
        if "ManagerID" in meta_row.index:
            info_lines.append(f"<b>ManagerID</b> : {meta_row.get('ManagerID', 'N/A')}")

        st.markdown(
            f"""
            <div class="soft-card">
                {'<br>'.join(info_lines)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "Termd" in meta_row.index:
            status_text = "Départ confirmé" if meta_row["Termd"] == 1 else "Actif"
            status_color = "#FEE4E2" if meta_row["Termd"] == 1 else "#EAF7EE"
            text_color = "#B42318" if meta_row["Termd"] == 1 else "#1B5E20"
            st.markdown(
                f"""
                <div class="soft-card">
                    <div class="metric-title">Statut historique dans le dataset</div>
                    <span class="pill" style="background:{status_color}; color:{text_color};">{status_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Âge", f"{int(meta_row.get('Age', 0))}")
            render_metric_card("Salaire", f"${int(meta_row.get('Salary', 0)):,}".replace(",", " "))
        with c2:
            render_metric_card("Ancienneté", f"{int(meta_row.get('Tenure', 0))} ans")
            render_metric_card("Engagement", f"{float(meta_row.get('EngagementSurvey', 0)):.2f}")
        with c3:
            render_metric_card("Satisfaction", f"{int(meta_row.get('EmpSatisfaction', 0))}/5")
            render_metric_card("Absences", f"{int(meta_row.get('Absences', 0))} j")

    st.markdown('<div class="section-title">Synthèse</div>', unsafe_allow_html=True)

    tags_col, text_col = st.columns([1, 2])

    with tags_col:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        render_pill(f"Source: {meta_row.get('RecruitmentSource', 'N/A')}", "#E6F0FF", "#0B4F9C")
        render_pill(f"Performance: {meta_row.get('PerformanceScore', 'N/A')}", "#EAF7EE", "#1B5E20")
        render_pill(f"État: {meta_row.get('State', 'N/A')}", "#FFF4E5", "#9A3412")
        st.markdown("</div>", unsafe_allow_html=True)

    with text_col:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("**Signal textuel synthétique**")
        st.write(meta_row.get("survey_comment", "N/A"))
        st.markdown("**Mobilité interne**")
        transfer = meta_row.get("transfer_request_text", "")
        st.write(transfer if str(transfer).strip() else "Aucune demande explicite détectée.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_analysis_tab(results: dict, X_row: pd.DataFrame):
    model = results[results["best_key"]]["model"]
    prob = float(model.predict_proba(X_row)[0, 1])

    st.markdown('<div class="section-title">Analyse IA</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="risk-card" style="background:{risk_color(prob)};">
            <div style="font-size:0.95rem; opacity:0.95;">Probabilité estimée de départ</div>
            <div style="font-size:2rem; font-weight:800; margin-top:0.2rem;">{prob * 100:.1f}%</div>
            <div style="font-size:1rem; margin-top:0.25rem;">{risk_label(prob)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(prob)

    sv_row, base_val, shap_model_name = get_individual_shap(results, X_row)

    if sv_row is None:
        st.error("Impossible de calculer SHAP pour cet employé.")
        return

    shap_df = pd.DataFrame({
        "Variable": X_row.columns,
        "SHAP": sv_row,
        "Valeur": X_row.iloc[0].values
    })
    shap_df["abs_shap"] = shap_df["SHAP"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False)

    top_pos = shap_df[shap_df["SHAP"] > 0].head(5).copy()
    top_neg = shap_df[shap_df["SHAP"] < 0].head(5).copy()

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown("**Lecture RH simplifiée**")
    st.write(build_executive_summary(prob, top_pos, top_neg))
    st.caption(f"Explication calculée avec : {shap_model_name}")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.05, 0.95])

    with col1:
        st.markdown('<div class="section-title">Facteurs qui augmentent le risque</div>', unsafe_allow_html=True)
        if top_pos.empty:
            st.info("Aucun facteur de hausse marqué.")
        else:
            pos_show = top_pos[["Variable", "SHAP", "Valeur"]].copy()
            pos_show["SHAP"] = pos_show["SHAP"].round(4)
            st.dataframe(pos_show, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Facteurs qui réduisent le risque</div>', unsafe_allow_html=True)
        if top_neg.empty:
            st.info("Peu de facteurs protecteurs visibles.")
        else:
            neg_show = top_neg[["Variable", "SHAP", "Valeur"]].copy()
            neg_show["SHAP"] = neg_show["SHAP"].round(4)
            st.dataframe(neg_show, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-title">Impact des principaux facteurs</div>', unsafe_allow_html=True)
        plot_df = shap_df.head(10).copy().sort_values("SHAP")
        colors = [CLR_LOW if v < 0 else CLR_HIGH for v in plot_df["SHAP"]]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(plot_df["Variable"], plot_df["SHAP"], color=colors)
        ax.axvline(0, color="black", lw=1)
        ax.set_title("Top contributions individuelles")
        ax.set_xlabel("Valeur SHAP")
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    recs = generate_retention_recommendations(X_row.iloc[0], shap_df)

    st.markdown('<div class="section-title">Recommandations de rétention</div>', unsafe_allow_html=True)
    for i, rec in enumerate(recs, 1):
        st.markdown(
            f"""
            <div class="action-card">
                <div style="font-weight:700; margin-bottom:0.3rem; color:#101828;">Action {i}</div>
                <div style="color:#101828;">{rec}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Voir l'explication SHAP détaillée"):
        try:
            exp = shap.Explanation(
                values=sv_row,
                base_values=base_val,
                data=X_row.values[0],
                feature_names=X_row.columns.tolist(),
            )
            fig, _ = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(exp, max_display=15, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Waterfall SHAP indisponible : {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if not os.path.exists(DATA_PATH):
        st.error("HRDataset_v14.csv introuvable.")
        return

    if not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
        st.error("Artefacts du modèle introuvables. Lancez d'abord hr_pipeline.py.")
        return

    _, feature_names, results = load_artifacts()
    meta_df, X_full, _ = load_and_prepare_data(tuple(feature_names))

    render_header(results)

    selected_idx = render_employee_selector(meta_df)
    meta_row = meta_df.loc[selected_idx]
    X_row = X_full.loc[[selected_idx]]

    tab1, tab2 = st.tabs(["Profil employé", "Analyse IA"])

    with tab1:
        render_profile_tab(meta_row)

    with tab2:
        render_analysis_tab(results, X_row)


if __name__ == "__main__":
    main()
