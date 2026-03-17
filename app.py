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

DATA_PATH  = "HRDataset_v14.csv"
MODELS_DIR = "models"

CLR_HIGH      = "#C62828"
CLR_MED       = "#E65100"
CLR_LOW       = "#2E7D32"
CLR_PRIMARY   = "#1565C0"
CLR_TEXT      = "#101828"
CLR_MUTED     = "#667085"
CLR_BG        = "#F8FAFC"
CLR_BORDER    = "#E6EAF0"


# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main { background-color: #F8FAFC; }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 1.2rem;
        max-width: 1300px;
    }

    /* General content card */
    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        border: 1px solid #E6EAF0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 0.9rem;
        color: #101828;
        overflow: hidden;
    }

    /* Coloured risk banner */
    .risk-banner {
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        color: #ffffff;
        margin-bottom: 1rem;
    }

    /* Left-accented action card for recommendations */
    .action-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        border: 1px solid #E6EAF0;
        border-left: 5px solid #1565C0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 0.75rem;
        color: #101828;
    }

    /* Section heading */
    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        color: #101828;
        margin-top: 0.6rem;
        margin-bottom: 0.4rem;
    }

    /* Metric card components */
    .metric-label {
        color: #667085;
        font-size: 0.82rem;
        margin-bottom: 0.15rem;
        font-weight: 500;
    }
    .metric-val {
        font-size: 1.3rem;
        font-weight: 700;
        color: #101828;
    }

    /* Pill / badge */
    .pill {
        display: inline-block;
        padding: 0.22rem 0.6rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
        border: 1px solid rgba(0,0,0,0.06);
    }

    /* Divider line */
    .divider {
        border: none;
        border-top: 1px solid #E6EAF0;
        margin: 0.8rem 0;
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
        return "High Risk"
    if prob >= 0.40:
        return "Moderate Risk"
    return "Low Risk"


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


def card(content: str) -> None:
    """Render a white card containing arbitrary HTML."""
    st.markdown(f'<div class="card">{content}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-val">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, bg: str, color: str = "#101828") -> None:
    st.markdown(
        f'<span class="pill" style="background:{bg}; color:{color};">{text}</span>',
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    best_model    = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    results       = joblib.load(os.path.join(MODELS_DIR, "results_summary.pkl"))
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
    demo_text_df["survey_comment"]        = X_nlp.apply(generate_survey, axis=1)
    demo_text_df["transfer_request_text"] = X_nlp.apply(generate_transfer, axis=1)

    salary_keywords   = ["compensation", "salary", "recognition", "progression"]
    growth_keywords   = ["growth", "development", "career", "progressing", "opportunities", "responsibility"]
    stress_keywords   = ["workload", "pressure", "support", "balance", "pace"]
    mobility_keywords = ["internal move", "internal mobility", "transfer", "another team", "another role"]

    text_combined = (
        demo_text_df["survey_comment"].fillna("") + " " + demo_text_df["transfer_request_text"].fillna("")
    ).str.strip()

    X_nlp["text_sentiment_score"]    = text_combined.apply(sentiment_score)
    X_nlp["topic_salary"]            = text_combined.apply(lambda x: contains_any(x, salary_keywords))
    X_nlp["topic_growth"]            = text_combined.apply(lambda x: contains_any(x, growth_keywords))
    X_nlp["topic_stress"]            = text_combined.apply(lambda x: contains_any(x, stress_keywords))
    X_nlp["topic_mobility"]          = text_combined.apply(lambda x: contains_any(x, mobility_keywords))
    X_nlp["negative_text_flag"]      = (X_nlp["text_sentiment_score"] < 0).astype(int)
    X_nlp["mobility_request_present"] = demo_text_df["transfer_request_text"].fillna("").str.len().gt(10).astype(int)

    return X_nlp, demo_text_df


@st.cache_data
def load_and_prepare_data(feature_names: tuple):
    df_raw = pd.read_csv(DATA_PATH)

    for col in df_raw.select_dtypes(include="object").columns:
        df_raw[col] = df_raw[col].astype(str).str.strip()

    df = df_raw.copy()

    df["DOB"]     = pd.to_datetime(df["DOB"],     dayfirst=False, errors="coerce")
    df["DateofHire"] = pd.to_datetime(df["DateofHire"], dayfirst=False, errors="coerce")
    df["LastPerformanceReview_Date"] = pd.to_datetime(
        df["LastPerformanceReview_Date"], dayfirst=False, errors="coerce"
    )

    ref_date = pd.Timestamp("2019-03-01")
    df["Age"]    = (ref_date - df["DOB"]).dt.days // 365
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

    pii_cols   = ["Employee_Name", "EmpID", "Zip", "ManagerName",
                  "DOB", "DateofHire", "LastPerformanceReview_Date"]
    leaky_cols = ["TermReason", "EmploymentStatus", "EmpStatusID", "DateofTermination"]

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
# SHAP ENGINE
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
            explainer   = shap.TreeExplainer(model, results["X_train"])
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
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def generate_retention_recommendations(x_row: pd.Series, shap_df: pd.DataFrame) -> list[str]:
    recs = []

    def positive(feature: str) -> bool:
        return not shap_df[(shap_df["Variable"] == feature) & (shap_df["SHAP"] > 0)].empty

    if x_row.get("low_salary_flag", 0) == 1 or x_row.get("topic_salary", 0) == 1:
        recs.append(
            "Initiate a compensation review or clearly communicate the salary "
            "progression path at the earliest opportunity."
        )

    if x_row.get("topic_growth", 0) == 1 or (
        x_row.get("high_tenure_flag", 0) == 1 and x_row.get("low_projects_flag", 0) == 1
    ):
        recs.append(
            "Schedule a career development conversation to clarify growth prospects, "
            "increasing responsibilities, and near-term next steps."
        )

    if (
        x_row.get("topic_stress", 0) == 1
        or x_row.get("high_absence_flag", 0) == 1
        or positive("AbsenteeismRate")
    ):
        recs.append(
            "Assess current workload and establish a short-term managerial support "
            "plan or a workload adjustment."
        )

    if x_row.get("topic_mobility", 0) == 1 or x_row.get("mobility_request_present", 0) == 1:
        recs.append(
            "Explore an internal mobility opportunity or a scope change as a retention "
            "lever to keep the employee within the organization."
        )

    if x_row.get("low_engagement_flag", 0) == 1 or x_row.get("text_sentiment_score", 0) < 0:
        recs.append(
            "Arrange an urgent HR check-in with the direct manager to surface specific "
            "pain points and agree on a concrete action plan."
        )

    if positive("ManagerID"):
        recs.append(
            "Review the team's managerial context and strengthen near-term follow-up "
            "and support for this employee."
        )

    if not recs:
        recs.append(
            "Maintain regular check-ins combining managerial feedback, recognition, "
            "and career dialogue."
        )

    return recs[:4]


def build_executive_summary(prob: float, top_pos: pd.DataFrame, top_neg: pd.DataFrame) -> str:
    pos_feats = top_pos["Variable"].head(3).tolist()
    neg_feats = top_neg["Variable"].head(2).tolist()

    pos_txt = ", ".join(pos_feats) if pos_feats else "no dominant contributing factors"
    neg_txt = ", ".join(neg_feats) if neg_feats else "no significant protective factors"

    return (
        f"The model estimates a {risk_label(prob).lower()} of departure. "
        f"The factors most strongly increasing this risk are: {pos_txt}. "
        f"The factors most supportive of retention are: {neg_txt}."
    )


# ─────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────

def render_header(results: dict) -> None:
    best_key = results["best_key"]

    st.title("HR Turnover Predictor")
    st.caption("AI-powered employee retention analysis")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Active Model", results["best_name"])
    with c2:
        metric_card("AUC-ROC", f"{results[best_key]['auc']:.3f}")
    with c3:
        metric_card("F1 — Turnover Class", f"{results[best_key]['f1']:.3f}")
    with c4:
        metric_card("Recall — Turnover Class", f"{results[best_key]['recall']:.3f}")


def render_employee_selector(meta_df: pd.DataFrame) -> int:
    section_title("Employee Selection")

    valid_meta    = meta_df.dropna(subset=["EmpID"]).copy()
    emp_choices   = valid_meta["EmpID"].astype(int).tolist()
    empid_to_idx  = {int(row["EmpID"]): idx for idx, row in valid_meta.iterrows()}

    selected_empid = st.selectbox(
        "Select an Employee ID",
        options=emp_choices,
        format_func=lambda x: f"Employee ID: {x}",
    )

    return empid_to_idx[int(selected_empid)]


def render_profile_tab(meta_row: pd.Series) -> None:

    # ── Identity & key figures ─────────────────────────────────
    section_title("Employee Profile")
    left, right = st.columns([1.0, 1.2])

    with left:
        emp_id    = int(meta_row["EmpID"]) if pd.notna(meta_row.get("EmpID")) else "N/A"
        position  = meta_row.get("Position", "N/A")
        dept      = meta_row.get("Department", "N/A")
        mgr_id    = meta_row.get("ManagerID", "N/A")

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">Employee ID</div>
                <div class="metric-val">ID: {emp_id}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card">
                <div style="line-height:1.8;">
                    <b>Position</b>: {position}<br>
                    <b>Department</b>: {dept}<br>
                    <b>Manager ID</b>: {mgr_id}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "Termd" in meta_row.index:
            terminated    = meta_row["Termd"] == 1
            status_text   = "Terminated" if terminated else "Active"
            status_bg     = "#FEE4E2" if terminated else "#ECFDF3"
            status_color  = "#B42318" if terminated else "#027A48"
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">Historical Status in Dataset</div>
                    <span class="pill" style="background:{status_bg}; color:{status_color};">
                        {status_text}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Age",    f"{int(meta_row.get('Age', 0))}")
            metric_card("Salary", f"${int(meta_row.get('Salary', 0)):,}".replace(",", " "))
        with c2:
            metric_card("Tenure",      f"{int(meta_row.get('Tenure', 0))} yrs")
            metric_card("Engagement",  f"{float(meta_row.get('EngagementSurvey', 0)):.2f} / 5")
        with c3:
            metric_card("Satisfaction", f"{int(meta_row.get('EmpSatisfaction', 0))} / 5")
            metric_card("Absences",     f"{int(meta_row.get('Absences', 0))} days")

    # ── Summary tags & text signals ───────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    section_title("Summary")

    tags_col, text_col = st.columns([1, 2])

    with tags_col:
        recruitment = meta_row.get("RecruitmentSource", "N/A")
        performance = meta_row.get("PerformanceScore",  "N/A")
        state       = meta_row.get("State", "N/A")

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label" style="margin-bottom:0.5rem;">Tags</div>
                <span class="pill" style="background:#E6F0FF; color:#0B4F9C;">
                    Source: {recruitment}
                </span><br>
                <span class="pill" style="background:#ECFDF3; color:#027A48;">
                    Performance: {performance}
                </span><br>
                <span class="pill" style="background:#FFF4E5; color:#9A3412;">
                    State: {state}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with text_col:
        survey   = meta_row.get("survey_comment", "N/A")
        transfer = str(meta_row.get("transfer_request_text", "")).strip()

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">Generated Survey Comment</div>
                <div style="color:#101828; margin-bottom:0.75rem;">{survey}</div>
                <div class="metric-label">Internal Mobility Request</div>
                <div style="color:#101828;">
                    {transfer if transfer else "No internal transfer request detected."}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analysis_tab(results: dict, X_row: pd.DataFrame) -> None:

    model = results[results["best_key"]]["model"]
    prob  = float(model.predict_proba(X_row)[0, 1])

    # ── Risk banner ────────────────────────────────────────────
    section_title("AI Analysis")
    st.markdown(
        f"""
        <div class="risk-banner" style="background:{risk_color(prob)};">
            <div style="font-size:0.92rem; opacity:0.9; font-weight:500;">
                Estimated Turnover Probability
            </div>
            <div style="font-size:2.1rem; font-weight:800; margin:0.15rem 0;">
                {prob * 100:.1f}%
            </div>
            <div style="font-size:0.95rem; font-weight:600; letter-spacing:0.02em;">
                {risk_label(prob)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(prob)

    # ── SHAP computation ───────────────────────────────────────
    sv_row, base_val, shap_model_name = get_individual_shap(results, X_row)

    if sv_row is None:
        st.error("SHAP explanation could not be computed for this employee.")
        return

    shap_df = pd.DataFrame({
        "Variable": X_row.columns,
        "SHAP":     sv_row,
        "Value":    X_row.iloc[0].values,
    })
    shap_df["abs_shap"] = shap_df["SHAP"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False)

    top_pos = shap_df[shap_df["SHAP"] > 0].head(5).copy()
    top_neg = shap_df[shap_df["SHAP"] < 0].head(5).copy()

    # ── Plain-language summary ─────────────────────────────────
    summary_text = build_executive_summary(prob, top_pos, top_neg)
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label" style="margin-bottom:0.3rem;">Plain-Language HR Summary</div>
            <div style="color:#101828; line-height:1.6;">{summary_text}</div>
            <div style="color:#667085; font-size:0.8rem; margin-top:0.5rem;">
                Explanation computed using: {shap_model_name}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── SHAP factor tables + chart ─────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col_tables, col_chart = st.columns([1.05, 0.95])

    with col_tables:
        section_title("Factors Increasing Turnover Risk")
        if top_pos.empty:
            st.info("No significant risk-increasing factors detected.")
        else:
            display_pos = top_pos[["Variable", "SHAP", "Value"]].copy()
            display_pos["SHAP"] = display_pos["SHAP"].round(4)
            st.dataframe(display_pos, use_container_width=True, hide_index=True)

        section_title("Factors Reducing Turnover Risk")
        if top_neg.empty:
            st.info("No significant protective factors detected.")
        else:
            display_neg = top_neg[["Variable", "SHAP", "Value"]].copy()
            display_neg["SHAP"] = display_neg["SHAP"].round(4)
            st.dataframe(display_neg, use_container_width=True, hide_index=True)

    with col_chart:
        section_title("Top Feature Contributions")
        plot_df = shap_df.head(10).copy().sort_values("SHAP")
        colors  = [CLR_LOW if v < 0 else CLR_HIGH for v in plot_df["SHAP"]]

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.barh(plot_df["Variable"], plot_df["SHAP"], color=colors, edgecolor="none", height=0.6)
        ax.axvline(0, color="#667085", lw=0.8, linestyle="--")
        ax.set_title("Individual SHAP Contributions", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("SHAP Value", fontsize=9)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Retention recommendations ──────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    section_title("Retention Recommendations")

    recs = generate_retention_recommendations(X_row.iloc[0], shap_df)
    for i, rec in enumerate(recs, 1):
        st.markdown(
            f"""
            <div class="action-card">
                <div style="font-weight:700; font-size:0.85rem; color:#1565C0;
                            margin-bottom:0.3rem; text-transform:uppercase;
                            letter-spacing:0.04em;">Action {i}</div>
                <div style="color:#101828; line-height:1.6;">{rec}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Detailed SHAP waterfall ────────────────────────────────
    with st.expander("View Detailed SHAP Waterfall"):
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
            st.warning(f"SHAP waterfall plot unavailable: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if not os.path.exists(DATA_PATH):
        st.error("HRDataset_v14.csv was not found in the working directory.")
        return

    if not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
        st.error(
            "Model artifacts not found. "
            "Run hr_pipeline.py first to train and save the models."
        )
        return

    _, feature_names, results = load_artifacts()
    meta_df, X_full, _        = load_and_prepare_data(tuple(feature_names))

    render_header(results)
    st.markdown("---")

    selected_idx = render_employee_selector(meta_df)
    meta_row     = meta_df.loc[selected_idx]
    X_row        = X_full.loc[[selected_idx]]

    st.markdown("---")

    tab1, tab2 = st.tabs(["Employee Profile", "AI Analysis"])

    with tab1:
        render_profile_tab(meta_row)

    with tab2:
        render_analysis_tab(results, X_row)


if __name__ == "__main__":
    main()
