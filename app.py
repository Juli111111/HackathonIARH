# -*- coding: utf-8 -*-

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    initial_sidebar_state="expanded",
)

DATA_PATH  = "HRDataset_v14.csv"
MODELS_DIR = "models"

CLR_HIGH    = "#D32F2F"
CLR_MED     = "#E65100"
CLR_LOW     = "#2E7D32"
CLR_PRIMARY = "#1A237E"
CLR_ACCENT  = "#1565C0"
CLR_TEXT    = "#0D1117"
CLR_MUTED   = "#6B7280"
CLR_BG      = "#F0F2F6"
CLR_CARD    = "#FFFFFF"
CLR_BORDER  = "#E5E7EB"


# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F0F2F6; }
[data-testid="stSidebar"]          { background: #0D1B2A; }
[data-testid="stSidebar"] * { color: #E5E7EB !important; }

.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

.hero {
    background: linear-gradient(135deg, #1A237E 0%, #1565C0 60%, #0288D1 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 { font-size: 1.9rem; font-weight: 800; margin: 0 0 0.2rem 0; color: white; }
.hero p  { font-size: 0.98rem; opacity: 0.85; margin: 0; }

.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    border: 1px solid #E5E7EB;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    text-align: center;
}
.kpi-label { color: #6B7280; font-size: 0.78rem; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem; }
.kpi-value { color: #0D1117; font-size: 1.6rem; font-weight: 800; }
.kpi-sub   { color: #6B7280; font-size: 0.75rem; margin-top: 0.15rem; }

.card {
    background: white;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #E5E7EB;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
    color: #0D1117;
}

.risk-banner {
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    color: white;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 2rem;
}
.risk-pct         { font-size: 3.2rem; font-weight: 900; line-height: 1; }
.risk-label-text  { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.03em; }
.risk-subtitle    { font-size: 0.85rem; opacity: 0.85; }

.score-row    { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.5rem; }
.score-name   { width: 110px; font-size: 0.82rem; color: #4B5563; font-weight: 500; }
.score-bar-bg { flex: 1; height: 8px; background: #F3F4F6; border-radius: 999px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 999px; }
.score-val    { width: 40px; text-align: right; font-size: 0.82rem; font-weight: 700; color: #0D1117; }

.info-row    { display: flex; justify-content: space-between; padding: 0.45rem 0;
               border-bottom: 1px solid #F3F4F6; font-size: 0.87rem; }
.info-row:last-child { border-bottom: none; }
.info-key    { color: #6B7280; font-weight: 500; }
.info-value  { color: #0D1117; font-weight: 600; }

.sec-title {
    font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: #6B7280; margin-bottom: 0.6rem; margin-top: 0.2rem;
}

.badge {
    display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 999px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.03em;
}

.action-card {
    background: white; border-radius: 12px; padding: 0.9rem 1.1rem;
    border: 1px solid #E5E7EB; border-left: 5px solid #1565C0;
    margin-bottom: 0.7rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.action-num  { font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
               letter-spacing: 0.06em; color: #1565C0; margin-bottom: 0.25rem; }
.action-text { color: #0D1117; font-size: 0.9rem; line-height: 1.55; }

.emp-sidebar-card {
    background: rgba(255,255,255,0.07); border-radius: 12px; padding: 1rem; margin-top: 1rem;
    border: 1px solid rgba(255,255,255,0.1);
}
.emp-sidebar-label { font-size: 0.75rem; opacity: 0.65; margin-bottom: 0.1rem; }
.emp-sidebar-value { font-size: 1rem; font-weight: 700; }

[data-testid="stTabs"] button { font-weight: 600 !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def risk_label(prob: float) -> str:
    if prob >= 0.70: return "High Risk"
    if prob >= 0.40: return "Moderate Risk"
    return "Low Risk"

def risk_color(prob: float) -> str:
    if prob >= 0.70: return CLR_HIGH
    if prob >= 0.40: return CLR_MED
    return CLR_LOW

def sentiment_score(text: str) -> int:
    text = str(text).lower()
    pos = sum(1 for k in ["appreciate","positive","stable","clear","value","satisfied"] if k in text)
    neg = sum(1 for k in ["not fully","would like more","difficult","pressure","hard","better"] if k in text)
    return pos - neg

def contains_any(text: str, keywords: list[str]) -> int:
    return int(any(k in str(text).lower() for k in keywords))

def score_bar(label: str, value: float, max_val: float, color: str = "#1565C0") -> str:
    pct = min(value / max_val * 100, 100)
    return f"""
    <div class="score-row">
        <div class="score-name">{label}</div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct:.0f}%; background:{color};"></div>
        </div>
        <div class="score-val">{value:.1f}</div>
    </div>"""

def info_row(key: str, value: str) -> str:
    return (f'<div class="info-row">'
            f'<span class="info-key">{key}</span>'
            f'<span class="info-value">{value}</span></div>')


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

    for col, flag, fn in [
        ("Salary",              "salary_pct",           lambda s: s.rank(pct=True)),
        ("EngagementSurvey",    "low_engagement_flag",  lambda s: (s < s.median()).astype(int)),
        ("Absences",            "high_absence_flag",    lambda s: (s >= s.quantile(0.75)).astype(int)),
        ("Tenure",              "high_tenure_flag",     lambda s: (s >= s.median()).astype(int)),
        ("SpecialProjectsCount","low_projects_flag",    lambda s: (s <= s.median()).astype(int)),
    ]:
        if col in X_nlp.columns:
            X_nlp[col] = pd.to_numeric(X_nlp[col], errors="coerce")
            X_nlp[flag] = fn(X_nlp[col])
        else:
            X_nlp[flag] = 0 if flag != "salary_pct" else 0.5

    perf_col = next((c for c in ["PerfScoreID","PerformanceScore"] if c in X_nlp.columns), None)
    if perf_col:
        X_nlp[perf_col] = pd.to_numeric(X_nlp[perf_col], errors="coerce")
        X_nlp["high_perf_flag"] = (X_nlp[perf_col] >= X_nlp[perf_col].median()).astype(int)
    else:
        X_nlp["high_perf_flag"] = 0

    X_nlp["low_salary_flag"] = (X_nlp["salary_pct"] <= 0.35).astype(int)

    pos_s = [
        "I appreciate the team environment and my responsibilities are clear. I feel stable in my current role.",
        "My work environment is positive and I value the collaboration within the team. I would like to continue developing my skills.",
        "I am generally satisfied with my role and the organization. Communication and expectations are mostly clear.",
    ]
    gro_s = [
        "My current role is stable overall, but I would like more clarity on future opportunities and development.",
        "I would appreciate better visibility on progression and responsibilities.",
        "I think there is room to improve communication about career development.",
    ]
    sal_s = [
        "I feel my compensation and career progression are not fully aligned with my contribution.",
        "I would like more visibility on salary progression and future growth.",
        "My responsibilities have grown, but recognition and progression do not seem to have followed at the same pace.",
    ]
    str_s = [
        "My workload has become difficult to manage consistently and I would appreciate more support.",
        "I have experienced pressure in my day-to-day work and I would like better balance.",
        "The current pace is sometimes hard to sustain and I would value workload adjustments.",
    ]
    tr_g = [
        "I would like to explore an internal move to a role with more responsibility and clearer growth prospects.",
        "I am interested in internal mobility because I would like broader responsibilities and a stronger development path.",
        "I would appreciate discussing a transfer opportunity to continue progressing within the company.",
    ]
    tr_f = [
        "I would like to discuss a move to another team where my skills could be better used.",
        "I am interested in an internal transfer to a role that better matches my professional objectives.",
        "I would like to explore internal mobility toward a position with a stronger fit for my experience.",
    ]

    rng    = np.random.default_rng(42)
    choose = lambda lst: lst[rng.integers(0, len(lst))]

    def gen_survey(row):
        if row["low_salary_flag"] == 1 and row["high_perf_flag"] == 1:   return choose(sal_s)
        if row["high_absence_flag"] == 1 and row["low_engagement_flag"] == 1: return choose(str_s)
        if row["high_tenure_flag"] == 1 and row["low_projects_flag"] == 1:    return choose(gro_s)
        if row["low_engagement_flag"] == 1:                                   return choose(gro_s)
        return choose(pos_s)

    def gen_transfer(row):
        score = (0.35 * row["low_salary_flag"] + 0.25 * row["high_perf_flag"]
                 + 0.20 * row["high_tenure_flag"] + 0.20 * row["low_engagement_flag"])
        if rng.random() > score: return ""
        return choose(tr_g if row["high_tenure_flag"] == 1 or row["low_projects_flag"] == 1 else tr_f)

    demo = pd.DataFrame(index=X_nlp.index)
    demo["survey_comment"]        = X_nlp.apply(gen_survey,    axis=1)
    demo["transfer_request_text"] = X_nlp.apply(gen_transfer,  axis=1)

    text = (demo["survey_comment"].fillna("") + " " + demo["transfer_request_text"].fillna("")).str.strip()
    X_nlp["text_sentiment_score"]     = text.apply(sentiment_score)
    X_nlp["topic_salary"]             = text.apply(lambda x: contains_any(x, ["compensation","salary","recognition","progression"]))
    X_nlp["topic_growth"]             = text.apply(lambda x: contains_any(x, ["growth","development","career","progressing","opportunities","responsibility"]))
    X_nlp["topic_stress"]             = text.apply(lambda x: contains_any(x, ["workload","pressure","support","balance","pace"]))
    X_nlp["topic_mobility"]           = text.apply(lambda x: contains_any(x, ["internal move","internal mobility","transfer","another team","another role"]))
    X_nlp["negative_text_flag"]       = (X_nlp["text_sentiment_score"] < 0).astype(int)
    X_nlp["mobility_request_present"] = demo["transfer_request_text"].fillna("").str.len().gt(10).astype(int)

    return X_nlp, demo


@st.cache_data
def load_and_prepare_data(feature_names: tuple):
    df_raw = pd.read_csv(DATA_PATH)
    for col in df_raw.select_dtypes("object").columns:
        df_raw[col] = df_raw[col].astype(str).str.strip()

    df = df_raw.copy()
    df["DOB"]      = pd.to_datetime(df["DOB"],      dayfirst=False, errors="coerce")
    df["DateofHire"] = pd.to_datetime(df["DateofHire"], dayfirst=False, errors="coerce")
    df["LastPerformanceReview_Date"] = pd.to_datetime(
        df["LastPerformanceReview_Date"], dayfirst=False, errors="coerce")

    ref = pd.Timestamp("2019-03-01")
    df["Age"]    = (ref - df["DOB"]).dt.days // 365
    df["Tenure"] = (ref - df["DateofHire"]).dt.days // 365
    df["DaysSinceLastReview"] = (ref - df["LastPerformanceReview_Date"]).dt.days
    df["ManagerID"] = df["ManagerID"].fillna(df["ManagerID"].median())

    meta_cols = ["EmpID","Department","Position","ManagerID","Salary","Age","Tenure",
                 "EngagementSurvey","EmpSatisfaction","Absences","DaysLateLast30",
                 "SpecialProjectsCount","RecruitmentSource","PerformanceScore",
                 "State","Sex","MaritalDesc","Termd"]
    meta_df = df[[c for c in meta_cols if c in df.columns]].copy()

    pii   = ["Employee_Name","EmpID","Zip","ManagerName","DOB","DateofHire","LastPerformanceReview_Date"]
    leaky = ["TermReason","EmploymentStatus","EmpStatusID","DateofTermination"]
    df_m  = df.drop(columns=pii + leaky, errors="ignore")

    y = df_m["Termd"].copy()
    X = df_m.drop(columns=["Termd","DaysSinceLastReview"], errors="ignore")
    X = pd.get_dummies(X, columns=X.select_dtypes("object").columns.tolist(), drop_first=True)
    X[X.select_dtypes("bool").columns] = X.select_dtypes("bool").astype(int)
    X, demo = enrich_with_nlp(X)

    if "Absences" in X.columns and "Tenure" in X.columns:
        X["AbsenteeismRate"] = X["Absences"] / X["Tenure"].clip(lower=1)
    if "DaysLateLast30" in X.columns and "Absences" in X.columns:
        X["RiskScore_Engagement"] = X["DaysLateLast30"] * 2 + X["Absences"]

    X = X.reindex(columns=list(feature_names), fill_value=0)
    meta_df = meta_df.join(demo)
    meta_df["EmpID"] = pd.to_numeric(meta_df["EmpID"], errors="coerce").astype("Int64")
    return meta_df, X, y


# ─────────────────────────────────────────────
# SHAP ENGINE
# ─────────────────────────────────────────────

def get_individual_shap(results: dict, X_row: pd.DataFrame):
    order = [(results["best_key"], results[results["best_key"]]["model"])]
    for k in ("rf", "xgb"):
        if k in results and k != results["best_key"]:
            order.append((k, results[k]["model"]))

    for key, model in order:
        try:
            exp = shap.TreeExplainer(model, results["X_train"])
            sv  = exp.shap_values(X_row, check_additivity=False)
            if isinstance(sv, list):    sv_row = sv[1][0]
            elif sv.ndim == 3:          sv_row = sv[0, :, 1]
            else:                       sv_row = sv[0]
            ev   = np.atleast_1d(exp.expected_value)
            base = ev[1] if len(ev) > 1 else ev[0]
            return sv_row, float(base), "Random Forest" if key == "rf" else "XGBoost"
        except Exception:
            continue
    return None, None, None


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def generate_recommendations(x_row: pd.Series, shap_df: pd.DataFrame) -> list[str]:
    recs = []
    pos  = lambda f: not shap_df[(shap_df["Variable"] == f) & (shap_df["SHAP"] > 0)].empty

    if x_row.get("low_salary_flag", 0) == 1 or x_row.get("topic_salary", 0) == 1:
        recs.append("Initiate a compensation review or clearly communicate the salary progression path at the earliest opportunity.")
    if x_row.get("topic_growth", 0) == 1 or (x_row.get("high_tenure_flag", 0) == 1 and x_row.get("low_projects_flag", 0) == 1):
        recs.append("Schedule a career development conversation to clarify growth prospects, increasing responsibilities, and near-term next steps.")
    if x_row.get("topic_stress", 0) == 1 or x_row.get("high_absence_flag", 0) == 1 or pos("AbsenteeismRate"):
        recs.append("Assess current workload and establish a short-term managerial support plan or a workload adjustment.")
    if x_row.get("topic_mobility", 0) == 1 or x_row.get("mobility_request_present", 0) == 1:
        recs.append("Explore an internal mobility opportunity or a scope change as a retention lever.")
    if x_row.get("low_engagement_flag", 0) == 1 or x_row.get("text_sentiment_score", 0) < 0:
        recs.append("Arrange an urgent HR check-in with the direct manager to surface pain points and agree on a concrete action plan.")
    if pos("ManagerID"):
        recs.append("Review the team's managerial context and strengthen near-term follow-up for this employee.")
    if not recs:
        recs.append("Maintain regular check-ins combining managerial feedback, recognition, and career dialogue.")
    return recs[:4]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar(results: dict, meta_df: pd.DataFrame) -> int:
    best_key = results["best_key"]

    st.sidebar.markdown("## HR Turnover Predictor")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model**")
    st.sidebar.markdown(f"`{results['best_name']}`")
    st.sidebar.markdown(
        f"AUC **{results[best_key]['auc']:.3f}** &nbsp;·&nbsp; F1 **{results[best_key]['f1']:.3f}**",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Select Employee**")

    valid   = meta_df.dropna(subset=["EmpID"]).copy()
    choices = valid["EmpID"].astype(int).tolist()
    idx_map = {int(r["EmpID"]): i for i, r in valid.iterrows()}

    selected_id  = st.sidebar.selectbox("Employee ID", options=choices,
                                         format_func=lambda x: f"ID {x}",
                                         label_visibility="collapsed")
    selected_idx = idx_map[int(selected_id)]
    row          = meta_df.loc[selected_idx]

    st.sidebar.markdown(
        f"""
        <div class="emp-sidebar-card">
            <div class="emp-sidebar-label">Department</div>
            <div class="emp-sidebar-value">{row.get('Department','N/A')}</div>
            <div style="margin-top:0.6rem;">
            <div class="emp-sidebar-label">Position</div>
            <div class="emp-sidebar-value" style="font-size:0.85rem;">{row.get('Position','N/A')}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    total       = len(meta_df)
    n_term      = int(meta_df["Termd"].sum())
    st.sidebar.markdown("**Dataset**")
    st.sidebar.markdown(f"Employees: **{total}**")
    st.sidebar.markdown(f"Turnover rate: **{n_term/total*100:.1f}%**")

    return selected_idx


# ─────────────────────────────────────────────
# PROFILE TAB
# ─────────────────────────────────────────────

def render_profile_tab(meta_row: pd.Series) -> None:
    col_id, col_scores, col_text = st.columns([1, 1.1, 1.4])

    # ── Identity card ─────────────────────────────────────────
    with col_id:
        terminated  = meta_row.get("Termd", 0) == 1
        badge_bg    = "#FEE2E2" if terminated else "#DCFCE7"
        badge_color = "#B91C1C" if terminated else "#15803D"
        badge_text  = "Terminated" if terminated else "Active"

        st.markdown(
            f"""
            <div class="card">
                <div class="sec-title">Identity</div>
                {info_row("Employee ID", str(int(meta_row.get("EmpID", 0))))}
                {info_row("Department",  str(meta_row.get("Department", "N/A")))}
                {info_row("Position",    str(meta_row.get("Position",   "N/A")))}
                {info_row("Manager ID",  str(meta_row.get("ManagerID",  "N/A")))}
                {info_row("State",       str(meta_row.get("State",      "N/A")))}
                {info_row("Gender",      str(meta_row.get("Sex",        "N/A")))}
                {info_row("Marital",     str(meta_row.get("MaritalDesc","N/A")))}
                <div style="margin-top:0.8rem;">
                    <span class="badge" style="background:{badge_bg}; color:{badge_color};">
                        {badge_text}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Engagement & score bars ────────────────────────────────
    with col_scores:
        eng      = float(meta_row.get("EngagementSurvey",   0))
        sat      = float(meta_row.get("EmpSatisfaction",    0))
        absences = int(meta_row.get("Absences",             0))
        late     = int(meta_row.get("DaysLateLast30",       0))
        projects = int(meta_row.get("SpecialProjectsCount", 0))
        salary   = float(meta_row.get("Salary", 0))
        age      = int(meta_row.get("Age",    0))
        tenure   = int(meta_row.get("Tenure", 0))

        eng_color = CLR_LOW if eng >= 3.5 else (CLR_MED if eng >= 2 else CLR_HIGH)
        sat_color = CLR_LOW if sat >= 3   else (CLR_MED if sat >= 2 else CLR_HIGH)
        abs_color = CLR_HIGH if absences >= 15 else (CLR_MED if absences >= 8 else CLR_LOW)

        st.markdown(
            f"""
            <div class="card">
                <div class="sec-title">Engagement & Attendance</div>
                {score_bar("Engagement",  eng,      5.0, eng_color)}
                {score_bar("Satisfaction",sat,      5.0, sat_color)}
                {score_bar("Absences",    absences, 20,  abs_color)}
                {score_bar("Late days",   late,     10,  CLR_HIGH if late >= 3 else CLR_LOW)}
                {score_bar("Projects",    projects, 8,   CLR_ACCENT)}
                <div style="margin-top:0.9rem;">
                    {info_row("Age",    f"{age} years")}
                    {info_row("Tenure", f"{tenure} years")}
                    {info_row("Salary", f"${salary:,.0f}")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Text signals ──────────────────────────────────────────
    with col_text:
        perf     = str(meta_row.get("PerformanceScore",  "N/A"))
        source   = str(meta_row.get("RecruitmentSource", "N/A"))
        survey   = str(meta_row.get("survey_comment",    "N/A"))
        transfer = str(meta_row.get("transfer_request_text", "")).strip()

        perf_styles = {
            "Exceeds":           ("#DCFCE7", "#15803D"),
            "Fully Meets":       ("#DBEAFE", "#1E40AF"),
            "Needs Improvement": ("#FEF9C3", "#92400E"),
            "PIP":               ("#FEE2E2", "#B91C1C"),
        }
        p_bg, p_clr = perf_styles.get(perf, ("#F3F4F6", "#374151"))

        st.markdown(
            f"""
            <div class="card">
                <div class="sec-title">Profile Summary</div>
                <div style="margin-bottom:0.9rem;">
                    <span class="badge" style="background:{p_bg}; color:{p_clr};">{perf}</span>
                    <span class="badge" style="background:#EFF6FF; color:#1D4ED8; margin-left:0.4rem;">
                        {source}
                    </span>
                </div>
                <div class="sec-title">Survey Comment</div>
                <div style="color:#374151; font-size:0.87rem; line-height:1.6;
                            margin-bottom:0.9rem; font-style:italic;">
                    &ldquo;{survey}&rdquo;
                </div>
                <div class="sec-title">Internal Mobility Request</div>
                <div style="color:#374151; font-size:0.87rem; line-height:1.6;">
                    {transfer if transfer else
                     '<span style="color:#9CA3AF;">No transfer request detected.</span>'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# AI ANALYSIS TAB
# ─────────────────────────────────────────────

def render_analysis_tab(results: dict, X_row: pd.DataFrame) -> None:
    model = results[results["best_key"]]["model"]
    prob  = float(model.predict_proba(X_row)[0, 1])
    color = risk_color(prob)

    sv_row, base_val, shap_name = get_individual_shap(results, X_row)

    # ── Risk banner + donut ────────────────────────────────────
    col_banner, col_donut = st.columns([2, 1])

    with col_banner:
        st.markdown(
            f"""
            <div class="risk-banner" style="background:linear-gradient(135deg, {color} 0%, {color}CC 100%);">
                <div class="risk-pct">{prob*100:.1f}<span style="font-size:1.4rem;">%</span></div>
                <div>
                    <div class="risk-label-text">{risk_label(prob)}</div>
                    <div class="risk-subtitle">Estimated probability of departure</div>
                    <div style="font-size:0.8rem; opacity:0.8; margin-top:0.4rem;">
                        Explained by: {shap_name if shap_name else 'N/A'}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(prob)

    with col_donut:
        fig, ax = plt.subplots(figsize=(3.2, 2.4), subplot_kw={"aspect": "equal"})
        fig.patch.set_facecolor("white")
        ax.pie([prob, 1 - prob], colors=[color, "#F3F4F6"], startangle=90,
               wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2})
        ax.text(0, -0.08, f"{prob*100:.0f}%", ha="center", va="center",
                fontsize=18, fontweight="bold", color=color)
        ax.text(0, -0.42, risk_label(prob), ha="center", va="center",
                fontsize=7.5, fontweight="600", color="#6B7280")
        ax.set_title("Risk Score", fontsize=9, color="#6B7280", pad=5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    if sv_row is None:
        st.error("SHAP explanation could not be computed for this employee.")
        return

    # ── Build SHAP dataframe ───────────────────────────────────
    shap_df = pd.DataFrame({
        "Variable": X_row.columns,
        "SHAP":     sv_row,
        "Value":    X_row.iloc[0].values,
    })
    shap_df["abs_shap"] = shap_df["SHAP"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False)

    top_pos = shap_df[shap_df["SHAP"] > 0].head(5)
    top_neg = shap_df[shap_df["SHAP"] < 0].head(5)
    pos_txt = ", ".join(top_pos["Variable"].head(3).tolist()) or "no dominant factors"
    neg_txt = ", ".join(top_neg["Variable"].head(2).tolist()) or "no significant protective factors"

    st.info(
        f"The model estimates a **{risk_label(prob).lower()}** of departure. "
        f"Key risk drivers: **{pos_txt}**. "
        f"Protective factors: **{neg_txt}**."
    )

    # ── SHAP chart + tables ────────────────────────────────────
    st.markdown("---")
    col_chart, col_tables = st.columns([1.2, 0.9])

    with col_chart:
        st.markdown('<div class="sec-title">Top Feature Contributions (SHAP)</div>',
                    unsafe_allow_html=True)
        plot_df = shap_df.head(12).sort_values("SHAP")
        colors  = [CLR_LOW if v < 0 else CLR_HIGH for v in plot_df["SHAP"]]

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        bars = ax.barh(plot_df["Variable"], plot_df["SHAP"],
                       color=colors, edgecolor="none", height=0.55)
        ax.axvline(0, color="#9CA3AF", lw=1, linestyle="--")

        for bar, val in zip(bars, plot_df["SHAP"]):
            ax.text(
                val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8, color="#374151",
            )

        ax.set_xlabel("SHAP Value  (+  increases risk  /  −  reduces risk)",
                      fontsize=8.5, color="#6B7280")
        ax.tick_params(labelsize=8.5, colors="#374151")
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color("#E5E7EB")
        ax.legend(
            handles=[mpatches.Patch(color=CLR_HIGH, label="Increases turnover risk"),
                     mpatches.Patch(color=CLR_LOW,  label="Reduces turnover risk")],
            fontsize=8, frameon=False, loc="lower right",
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_tables:
        st.markdown('<div class="sec-title">Risk Factors</div>', unsafe_allow_html=True)
        if not top_pos.empty:
            d = top_pos[["Variable","SHAP","Value"]].copy()
            d["SHAP"]  = d["SHAP"].round(4)
            d["Value"] = d["Value"].round(3)
            st.dataframe(d, use_container_width=True, hide_index=True,
                         column_config={"SHAP": st.column_config.NumberColumn(format="%.4f")})
        else:
            st.info("No significant risk-increasing factors.")

        st.markdown('<div class="sec-title" style="margin-top:0.8rem;">Protective Factors</div>',
                    unsafe_allow_html=True)
        if not top_neg.empty:
            d = top_neg[["Variable","SHAP","Value"]].copy()
            d["SHAP"]  = d["SHAP"].round(4)
            d["Value"] = d["Value"].round(3)
            st.dataframe(d, use_container_width=True, hide_index=True,
                         column_config={"SHAP": st.column_config.NumberColumn(format="%.4f")})
        else:
            st.info("No significant protective factors.")

    # ── Recommendations ────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-title">Retention Recommendations</div>',
                unsafe_allow_html=True)

    for i, rec in enumerate(generate_recommendations(X_row.iloc[0], shap_df), 1):
        st.markdown(
            f"""
            <div class="action-card">
                <div class="action-num">Action {i}</div>
                <div class="action-text">{rec}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── SHAP waterfall ─────────────────────────────────────────
    with st.expander("View Detailed SHAP Waterfall"):
        try:
            exp = shap.Explanation(
                values=sv_row, base_values=base_val,
                data=X_row.values[0], feature_names=X_row.columns.tolist(),
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
        st.error("Model artifacts not found. Run hr_pipeline.py first to train and save the models.")
        return

    _, feature_names, results = load_artifacts()
    meta_df, X_full, _        = load_and_prepare_data(tuple(feature_names))

    selected_idx = render_sidebar(results, meta_df)
    meta_row     = meta_df.loc[selected_idx]
    X_row        = X_full.loc[[selected_idx]]

    # ── Hero + KPI bar ─────────────────────────────────────────
    best_key = results["best_key"]
    st.markdown(
        f"""
        <div class="hero">
            <h1>HR Turnover Predictor</h1>
            <p>AI-powered employee retention analysis &nbsp;·&nbsp;
               Model: {results['best_name']} &nbsp;·&nbsp;
               AUC {results[best_key]['auc']:.3f} &nbsp;·&nbsp;
               F1 {results[best_key]['f1']:.3f} &nbsp;·&nbsp;
               Recall {results[best_key]['recall']:.3f}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total   = len(meta_df)
    n_term  = int(meta_df["Termd"].sum())
    avg_eng = meta_df["EngagementSurvey"].mean()
    avg_abs = meta_df["Absences"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, lbl, val, sub in [
        (k1, "Total Employees",  str(total),                       "in dataset"),
        (k2, "Turnover Rate",    f"{n_term/total*100:.1f}%",        "confirmed departures"),
        (k3, "Avg Engagement",   f"{avg_eng:.2f} / 5",             "survey score"),
        (k4, "Avg Absences",     f"{avg_abs:.1f} days",            "per year"),
        (k5, "Active Model",     results["best_name"],             f"AUC {results[best_key]['auc']:.3f}"),
    ]:
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Employee Profile", "AI Analysis"])
    with tab1:
        render_profile_tab(meta_row)
    with tab2:
        render_analysis_tab(results, X_row)


if __name__ == "__main__":
    main()
