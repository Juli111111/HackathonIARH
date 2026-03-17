# -*- coding: utf-8 -*-
"""
App Streamlit pour HR Turnover Predictor
Lancez avec :
    streamlit run app.py
"""

import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

DATA_PATH = "HRDataset_v14.csv"
MODELS_DIR = "models"
PLOTS_DIR = "plots"

CLR_HIGH = "#C62828"
CLR_MED = "#E65100"
CLR_LOW = "#2E7D32"
CLR_PRIMARY = "#1565C0"
CLR_SECONDARY = "#37474F"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "RISQUE ÉLEVÉ"
    if prob >= 0.40:
        return "RISQUE MODÉRÉ"
    return "RISQUE FAIBLE"


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


def empirical_percentile(value: float, values: np.ndarray) -> float:
    if values is None or len(values) == 0:
        return 0.5
    return float((values <= value).mean())


def safe_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else None


def build_mapping(df: pd.DataFrame, key_col: str, value_col: str) -> dict:
    if key_col not in df.columns or value_col not in df.columns:
        return {}
    tmp = df[[key_col, value_col]].dropna()
    if tmp.empty:
        return {}
    grouped = tmp.groupby(key_col)[value_col].agg(safe_mode)
    return grouped.to_dict()


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    results = joblib.load(os.path.join(MODELS_DIR, "results_summary.pkl"))
    return best_model, feature_names, results


@st.cache_data
def load_reference_data():
    if not os.path.exists(DATA_PATH):
        return None, {}

    df = pd.read_csv(DATA_PATH)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    if "ManagerID" in df.columns:
        df["ManagerID"] = df["ManagerID"].fillna(df["ManagerID"].median())

    if "DOB" in df.columns:
        df["DOB"] = pd.to_datetime(df["DOB"], dayfirst=False, errors="coerce")
    if "DateofHire" in df.columns:
        df["DateofHire"] = pd.to_datetime(df["DateofHire"], dayfirst=False, errors="coerce")

    ref_date = pd.Timestamp("2019-03-01")
    if "DOB" in df.columns:
        df["Age"] = (ref_date - df["DOB"]).dt.days // 365
    if "DateofHire" in df.columns:
        df["Tenure"] = (ref_date - df["DateofHire"]).dt.days // 365

    stats = {
        "salary_values": np.sort(df["Salary"].dropna().values) if "Salary" in df.columns else np.array([]),
        "engagement_median": float(df["EngagementSurvey"].median()) if "EngagementSurvey" in df.columns else 4.0,
        "absence_q75": float(df["Absences"].quantile(0.75)) if "Absences" in df.columns else 10.0,
        "tenure_median": float(df["Tenure"].median()) if "Tenure" in df.columns else 5.0,
        "projects_median": float(df["SpecialProjectsCount"].median()) if "SpecialProjectsCount" in df.columns else 0.0,
        "perf_median": float(df["PerfScoreID"].median()) if "PerfScoreID" in df.columns else 3.0,
        "manager_default": float(df["ManagerID"].median()) if "ManagerID" in df.columns else 15.0,
        "deptid_map": build_mapping(df, "Department", "DeptID"),
        "positionid_map": build_mapping(df, "Position", "PositionID"),
        "genderid_map": build_mapping(df, "Sex", "GenderID"),
        "perf_map": build_mapping(df, "PerformanceScore", "PerfScoreID"),
        "maritalstatus_map": build_mapping(df, "MaritalDesc", "MaritalStatusID"),
        "options": {}
    }

    option_cols = [
        "Department", "Position", "State", "Sex", "MaritalDesc",
        "CitizenDesc", "HispanicLatino", "RaceDesc", "RecruitmentSource",
        "PerformanceScore"
    ]
    for col in option_cols:
        stats["options"][col] = sorted(df[col].dropna().astype(str).unique().tolist()) if col in df.columns else []

    if "ManagerID" in df.columns:
        managers = sorted(pd.Series(df["ManagerID"].dropna().unique()).astype(int).tolist())
    else:
        managers = [15]
    stats["options"]["ManagerID"] = managers

    return df, stats


def build_feature_vector(inputs: dict, feature_names: list[str], results: dict, ref_stats: dict) -> pd.DataFrame:
    base_means = results["X_train"].mean(numeric_only=True).reindex(feature_names, fill_value=0.0)
    row = pd.Series(base_means, index=feature_names, dtype=float)

    for prefix in [
        "Department", "Position", "State", "Sex", "MaritalDesc",
        "CitizenDesc", "HispanicLatino", "RaceDesc",
        "RecruitmentSource", "PerformanceScore"
    ]:
        prefixed_cols = [c for c in feature_names if c.startswith(f"{prefix}_")]
        for c in prefixed_cols:
            row[c] = 0.0

    perf_map_fallback = {
        "PIP": 1,
        "Needs Improvement": 2,
        "Fully Meets": 3,
        "Exceeds": 4,
    }

    gender_map = ref_stats.get("genderid_map", {})
    perf_map = ref_stats.get("perf_map", {})
    maritalstatus_map = ref_stats.get("maritalstatus_map", {})
    deptid_map = ref_stats.get("deptid_map", {})
    positionid_map = ref_stats.get("positionid_map", {})

    salary = float(inputs.get("Salary", row.get("Salary", 62000)))
    age = float(inputs.get("Age", row.get("Age", 35)))
    tenure = float(inputs.get("Tenure", row.get("Tenure", 5)))
    engagement = float(inputs.get("EngagementSurvey", row.get("EngagementSurvey", 4.0)))
    satisfaction = float(inputs.get("EmpSatisfaction", row.get("EmpSatisfaction", 3)))
    absences = float(inputs.get("Absences", row.get("Absences", 10)))
    days_late = float(inputs.get("DaysLateLast30", row.get("DaysLateLast30", 0)))
    special_projects = float(inputs.get("SpecialProjectsCount", row.get("SpecialProjectsCount", 0)))
    manager_id = float(inputs.get("ManagerID", ref_stats.get("manager_default", row.get("ManagerID", 15))))
    sex = str(inputs.get("Sex", "M"))
    married = bool(inputs.get("Married", False))
    diversity_hire = bool(inputs.get("DiversityHire", False))
    department = str(inputs.get("Department", "Production"))
    position = str(inputs.get("Position", "Production Technician I"))
    state = str(inputs.get("State", "MA"))
    marital_desc = str(inputs.get("MaritalDesc", "Single"))
    citizen_desc = str(inputs.get("CitizenDesc", "US Citizen"))
    hispanic = str(inputs.get("HispanicLatino", "No"))
    race = str(inputs.get("RaceDesc", "White"))
    recruit_src = str(inputs.get("RecruitmentSource", "Indeed"))
    perf_label = str(inputs.get("PerformanceScore", "Fully Meets"))

    perf_id = perf_map.get(perf_label, perf_map_fallback.get(perf_label, row.get("PerfScoreID", 3)))
    gender_id = gender_map.get(sex, 1 if sex == "M" else 0)
    marital_status_id = maritalstatus_map.get(marital_desc, row.get("MaritalStatusID", 1))
    dept_id = deptid_map.get(department, row.get("DeptID", 5))
    position_id = positionid_map.get(position, row.get("PositionID", 19))

    numeric_map = {
        "Salary": salary,
        "Age": age,
        "Tenure": tenure,
        "EngagementSurvey": engagement,
        "EmpSatisfaction": satisfaction,
        "Absences": absences,
        "DaysLateLast30": days_late,
        "SpecialProjectsCount": special_projects,
        "ManagerID": manager_id,
        "GenderID": gender_id,
        "MarriedID": int(married),
        "FromDiversityJobFairID": int(diversity_hire),
        "PerfScoreID": perf_id,
        "MaritalStatusID": marital_status_id,
        "DeptID": dept_id,
        "PositionID": position_id,
    }

    for col, val in numeric_map.items():
        if col in row.index:
            row[col] = float(val)

    for prefix, value in [
        ("Department", department),
        ("Position", position),
        ("State", state),
        ("Sex", sex),
        ("MaritalDesc", marital_desc),
        ("CitizenDesc", citizen_desc),
        ("HispanicLatino", hispanic),
        ("RaceDesc", race),
        ("RecruitmentSource", recruit_src),
        ("PerformanceScore", perf_label),
    ]:
        col = f"{prefix}_{value}"
        if col in row.index:
            row[col] = 1.0

    tenure_safe = max(tenure, 1.0)
    if "AbsenteeismRate" in row.index:
        row["AbsenteeismRate"] = absences / tenure_safe
    if "RiskScore_Engagement" in row.index:
        row["RiskScore_Engagement"] = days_late * 2 + absences

    salary_pct = empirical_percentile(salary, ref_stats.get("salary_values", np.array([])))
    low_engagement_flag = int(engagement < ref_stats.get("engagement_median", 4.0))
    high_absence_flag = int(absences >= ref_stats.get("absence_q75", 10.0))
    high_tenure_flag = int(tenure >= ref_stats.get("tenure_median", 5.0))
    low_projects_flag = int(special_projects <= ref_stats.get("projects_median", 0.0))
    high_perf_flag = int(perf_id >= ref_stats.get("perf_median", 3.0))
    low_salary_flag = int(salary_pct <= 0.35)

    survey_comment = str(inputs.get("survey_comment", "")).strip()
    transfer_request_text = str(inputs.get("transfer_request_text", "")).strip()
    combined_text = f"{survey_comment} {transfer_request_text}".strip()

    salary_keywords = ["compensation", "salary", "recognition", "progression"]
    growth_keywords = ["growth", "development", "career", "progressing", "opportunities", "responsibility"]
    stress_keywords = ["workload", "pressure", "support", "balance", "pace"]
    mobility_keywords = ["internal move", "internal mobility", "transfer", "another team", "another role"]

    derived_map = {
        "salary_pct": salary_pct,
        "low_engagement_flag": low_engagement_flag,
        "high_absence_flag": high_absence_flag,
        "high_tenure_flag": high_tenure_flag,
        "low_projects_flag": low_projects_flag,
        "high_perf_flag": high_perf_flag,
        "low_salary_flag": low_salary_flag,
        "text_sentiment_score": sentiment_score(combined_text),
        "topic_salary": contains_any(combined_text, salary_keywords),
        "topic_growth": contains_any(combined_text, growth_keywords),
        "topic_stress": contains_any(combined_text, stress_keywords),
        "topic_mobility": contains_any(combined_text, mobility_keywords),
        "negative_text_flag": int(sentiment_score(combined_text) < 0),
        "mobility_request_present": int(len(transfer_request_text) > 10),
    }

    for col, val in derived_map.items():
        if col in row.index:
            row[col] = float(val)

    row = row.fillna(0.0)
    return pd.DataFrame([row], columns=feature_names)


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
            return explainer, sv_row, base_val, model_name
        except Exception:
            continue

    return None, None, None, None


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar(raw_df: pd.DataFrame | None, results: dict):
    st.sidebar.title("HR Turnover Predictor")
    st.sidebar.caption("Dashboard connecté aux artefacts de hr_pipeline.py")
    st.sidebar.markdown("---")

    if raw_df is not None and "Termd" in raw_df.columns:
        total = len(raw_df)
        departures = int(raw_df["Termd"].sum())
        st.sidebar.metric("Employés analysés", total)
        st.sidebar.metric("Taux de départ", f"{departures / total * 100:.1f} %")

    best_key = results["best_key"]
    st.sidebar.metric("Modèle retenu", results["best_name"])
    st.sidebar.metric("AUC", f"{results[best_key]['auc']:.3f}")
    st.sidebar.metric("F1 Démission", f"{results[best_key]['f1']:.3f}")
    st.sidebar.metric("Recall Démission", f"{results[best_key]['recall']:.3f}")
    st.sidebar.metric("Precision Démission", f"{results[best_key]['precision']:.3f}")
    st.sidebar.metric("Seuil", f"{results[best_key]['threshold']:.2f}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Conformité RGPD : PII supprimées, variables de fuite retirées.")


# ─────────────────────────────────────────────
# ONGLET 1 — VUE GLOBALE
# ─────────────────────────────────────────────

def tab_overview(raw_df: pd.DataFrame | None, results: dict):
    st.header("Vue globale")

    best_key = results["best_key"]
    metrics = results[best_key]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modèle", results["best_name"])
    c2.metric("AUC", f"{metrics['auc']:.3f}")
    c3.metric("F1 Démission", f"{metrics['f1']:.3f}")
    c4.metric("Recall Démission", f"{metrics['recall']:.3f}")

    if raw_df is not None:
        df = raw_df.copy()
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].astype(str).str.strip()

        st.markdown("---")
        left, right = st.columns(2)

        with left:
            if "Department" in df.columns and "Termd" in df.columns:
                st.subheader("Taux de départ par département")
                dept = df.groupby("Department")["Termd"].agg(["sum", "count"])
                dept["rate"] = dept["sum"] / dept["count"] * 100
                dept = dept.sort_values("rate")

                fig, ax = plt.subplots(figsize=(7, 4))
                colors = [CLR_HIGH if r > 40 else CLR_MED if r > 20 else CLR_LOW for r in dept["rate"]]
                dept["rate"].plot(kind="barh", ax=ax, color=colors)
                for i, v in enumerate(dept["rate"]):
                    ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
                ax.set_xlabel("Taux de départ (%)")
                ax.set_title("Taux de départ par département")
                sns.despine(ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with right:
            if "EmpSatisfaction" in df.columns and "Termd" in df.columns:
                st.subheader("Satisfaction selon le statut")
                df["Statut"] = df["Termd"].map({0: "Actif", 1: "Départ confirmé"})
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.violinplot(
                    data=df, x="Statut", y="EmpSatisfaction",
                    palette={"Actif": CLR_LOW, "Départ confirmé": CLR_HIGH}, ax=ax
                )
                ax.set_xlabel("")
                ax.set_title("Distribution de satisfaction")
                sns.despine(ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        roc_path = os.path.join(PLOTS_DIR, "06_roc_curves.png")
        if os.path.exists(roc_path):
            st.subheader("Courbes ROC")
            st.image(roc_path, use_container_width=True)

    with col_b:
        fi_path = os.path.join(PLOTS_DIR, "07_feature_importance.png")
        if os.path.exists(fi_path):
            st.subheader("Variables importantes")
            st.image(fi_path, use_container_width=True)


# ─────────────────────────────────────────────
# ONGLET 2 — PRÉDICTION
# ─────────────────────────────────────────────

def tab_prediction(feature_names: list[str], results: dict, ref_stats: dict):
    st.header("Prédiction du risque")
    st.markdown("Renseignez un profil employé pour obtenir une probabilité de départ et une explication locale.")

    opts = ref_stats.get("options", {})

    with st.form("employee_form"):
        st.subheader("Profil RH")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            department = st.selectbox("Département", opts.get("Department", ["Production"]))
            position = st.selectbox("Poste", opts.get("Position", ["Production Technician I"]))
            manager_id = st.selectbox("ManagerID", opts.get("ManagerID", [15]))
        with c2:
            sex = st.selectbox("Genre", opts.get("Sex", ["M", "F"]))
            marital_desc = st.selectbox("Statut marital", opts.get("MaritalDesc", ["Single"]))
            state = st.selectbox("État", opts.get("State", ["MA"]))
        with c3:
            citizen_desc = st.selectbox("Citoyenneté", opts.get("CitizenDesc", ["US Citizen"]))
            hispanic = st.selectbox("Hispanic/Latino", opts.get("HispanicLatino", ["No"]))
            race = st.selectbox("Origine", opts.get("RaceDesc", ["White"]))
        with c4:
            recruit_src = st.selectbox("Source de recrutement", opts.get("RecruitmentSource", ["Indeed"]))
            perf_label = st.selectbox("Performance", opts.get("PerformanceScore", ["Fully Meets"]))
            married = st.checkbox("Marié(e)")
            diversity_hire = st.checkbox("Recruté via Diversity Job Fair")

        st.subheader("Indicateurs de suivi")
        n1, n2, n3, n4, n5 = st.columns(5)
        with n1:
            age = st.slider("Âge", 18, 70, 35)
        with n2:
            tenure = st.slider("Ancienneté", 0, 25, 5)
        with n3:
            salary = st.number_input("Salaire annuel ($)", 40000, 300000, 62000, step=1000)
        with n4:
            engagement = st.slider("Engagement", 1.0, 5.0, 4.0, step=0.1)
        with n5:
            satisfaction = st.slider("Satisfaction", 1, 5, 3)

        n6, n7, n8 = st.columns(3)
        with n6:
            absences = st.slider("Absences (j/an)", 0, 20, 10)
        with n7:
            days_late = st.slider("Retards (30 derniers jours)", 0, 6, 0)
        with n8:
            special_projects = st.slider("Projets spéciaux", 0, 8, 0)

        st.subheader("Signaux textuels optionnels")
        survey_comment = st.text_area(
            "Commentaire employé",
            placeholder="Ex: I would like more visibility on career growth and compensation."
        )
        transfer_request_text = st.text_area(
            "Demande de mobilité interne",
            placeholder="Ex: I would like to explore an internal move to another team."
        )

        submitted = st.form_submit_button("Analyser le profil")

    if submitted:
        inputs = {
            "Department": department,
            "Position": position,
            "ManagerID": manager_id,
            "Sex": sex,
            "MaritalDesc": marital_desc,
            "State": state,
            "CitizenDesc": citizen_desc,
            "HispanicLatino": hispanic,
            "RaceDesc": race,
            "RecruitmentSource": recruit_src,
            "PerformanceScore": perf_label,
            "Married": married,
            "DiversityHire": diversity_hire,
            "Age": age,
            "Tenure": tenure,
            "Salary": salary,
            "EngagementSurvey": engagement,
            "EmpSatisfaction": satisfaction,
            "Absences": absences,
            "DaysLateLast30": days_late,
            "SpecialProjectsCount": special_projects,
            "survey_comment": survey_comment,
            "transfer_request_text": transfer_request_text,
        }

        X_row = build_feature_vector(inputs, feature_names, results, ref_stats)
        prob = float(results[results["best_key"]]["model"].predict_proba(X_row)[0, 1])

        st.session_state["last_inputs"] = inputs
        st.session_state["last_row"] = X_row
        st.session_state["last_prob"] = prob

    if "last_prob" not in st.session_state:
        return

    prob = st.session_state["last_prob"]
    X_row = st.session_state["last_row"]

    st.markdown("---")
    st.subheader("Résultat")
    left, right = st.columns([2, 1])

    with left:
        st.markdown(
            f"<h2 style='color:{risk_color(prob)}; margin-bottom:0;'>{risk_label(prob)}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(f"### Probabilité de départ : {prob * 100:.1f}%")
        st.progress(float(prob))

        if prob >= 0.70:
            st.error("Ce profil présente un risque élevé. Une action RH rapide est recommandée.")
        elif prob >= 0.40:
            st.warning("Risque modéré. Un suivi ciblé paraît pertinent.")
        else:
            st.success("Risque faible selon le modèle actuel.")

    with right:
        fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw={"aspect": "equal"})
        theta = np.linspace(0, np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), color="#E0E0E0", lw=12)
        ax.plot(
            np.cos(theta[:max(1, int(prob * 300))]),
            np.sin(theta[:max(1, int(prob * 300))]),
            color=risk_color(prob), lw=12
        )
        ax.text(0, -0.15, f"{prob * 100:.0f}%", ha="center", va="center",
                fontsize=24, fontweight="bold", color=risk_color(prob))
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.4, 1.3)
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Explication locale SHAP")

    explainer, sv_row, base_val, shap_model_name = get_individual_shap(results, X_row)

    if sv_row is None:
        st.info("SHAP n'a pas pu être calculé pour ce profil.")
        return

    st.caption(f"Modèle utilisé pour l'explication : {shap_model_name}")

    try:
        exp = shap.Explanation(
            values=sv_row,
            base_values=base_val,
            data=X_row.values[0],
            feature_names=feature_names,
        )
        fig, _ = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(exp, max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Waterfall SHAP indisponible : {e}")

    shap_df = (
        pd.DataFrame({"Variable": feature_names, "SHAP": sv_row, "Valeur": X_row.iloc[0].values})
        .assign(abs_shap=lambda d: d["SHAP"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(12)
    )
    shap_df["Direction"] = shap_df["SHAP"].apply(
        lambda v: "Augmente le risque" if v > 0 else "Réduit le risque"
    )
    st.dataframe(
        shap_df[["Variable", "Valeur", "SHAP", "Direction"]].style.format({"Valeur": "{:.3f}", "SHAP": "{:.4f}"}),
        use_container_width=True
    )


# ─────────────────────────────────────────────
# ONGLET 3 — EXPLICATIONS GLOBALES
# ─────────────────────────────────────────────

def tab_xai(results: dict, feature_names: list[str]):
    st.header("Explications globales")

    c1, c2 = st.columns(2)

    with c1:
        shap_summary = os.path.join(PLOTS_DIR, "08_shap_summary.png")
        if os.path.exists(shap_summary):
            st.subheader("SHAP summary")
            st.image(shap_summary, use_container_width=True)

    with c2:
        shap_bar = os.path.join(PLOTS_DIR, "09_shap_bar.png")
        if os.path.exists(shap_bar):
            st.subheader("SHAP bar")
            st.image(shap_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Top variables du modèle")

    model = results[results["best_key"]]["model"]
    if hasattr(model, "feature_importances_"):
        fi = (
            pd.Series(model.feature_importances_, index=feature_names)
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fi.columns = ["Variable", "Importance"]

        fig, ax = plt.subplots(figsize=(9, 6))
        fi.sort_values("Importance").plot(
            kind="barh", x="Variable", y="Importance",
            ax=ax, color=CLR_PRIMARY, legend=False
        )
        ax.set_title("Top 15 variables")
        ax.set_xlabel("Importance")
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(fi.style.format({"Importance": "{:.4f}"}), use_container_width=True)


# ─────────────────────────────────────────────
# ONGLET 4 — SÉCURITÉ & RGPD
# ─────────────────────────────────────────────

def tab_security():
    st.header("Sécurité & RGPD")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mesures implémentées")
        st.success(
            "Pseudonymisation des données RH\n\n"
            "Suppression des PII : nom, identifiant, date de naissance, code postal.\n\n"
            "Suppression des variables de fuite\n\n"
            "Retrait de DaysSinceLastReview du modèle final.\n\n"
            "Séparation train / validation / test.\n\n"
            "Explicabilité avec SHAP.\n\n"
            "Sauvegarde locale des artefacts du modèle."
        )

    with col2:
        st.subheader("Recommandations")
        st.warning(
            "Chiffrement des données sensibles.\n\n"
            "Contrôle d'accès au dashboard.\n\n"
            "Journal d'audit des prédictions.\n\n"
            "Audit des biais par groupes protégés.\n\n"
            "Revue humaine avant toute décision RH.\n\n"
            "Politique de conservation conforme au RGPD."
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    best_model, feature_names, results = load_artifacts()
    raw_df, ref_stats = load_reference_data()

    render_sidebar(raw_df, results)

    st.title("HR Turnover Predictor")
    st.markdown(
        "Application Streamlit branchée sur les artefacts de `hr_pipeline.py` "
        "pour visualiser les performances du modèle, explorer les facteurs de risque "
        "et simuler des prédictions individuelles."
    )
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Vue globale",
        "Prédiction",
        "Explications IA",
        "Sécurité & RGPD",
    ])

    with tab1:
        tab_overview(raw_df, results)

    with tab2:
        tab_prediction(feature_names, results, ref_stats)

    with tab3:
        tab_xai(results, feature_names)

    with tab4:
        tab_security()


if __name__ == "__main__":
    main()
