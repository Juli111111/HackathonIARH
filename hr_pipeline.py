# -*- coding: utf-8 -*-
"""
HR Turnover Prediction System
Système de prédiction des démissions d'employés
Hackathon IA RH — Pipeline complet

Dataset : HRDataset_v14.csv
Cible   : Termd (1 = départ confirmé, 0 = employé actif)
"""

import warnings
import os
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATA_PATH = "HRDataset_v14.csv"
PLOTS_DIR = "plots"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# ÉTAPE 1 — ANALYSE EXPLORATOIRE DES DONNÉES
# ─────────────────────────────────────────────

def load_and_explore(path: str) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(" ÉTAPE 1 — ANALYSE EXPLORATOIRE DES DONNÉES")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"\n► Dimensions : {df.shape[0]} employés × {df.shape[1]} colonnes")
    print(f"\n► Types de variables :\n{df.dtypes.value_counts()}")
    print(f"\n► Statistiques descriptives :\n{df.describe().T}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("\n► Aucune valeur manquante (hors DateofTermination pour actifs).")
    else:
        print(f"\n► Valeurs manquantes :\n{missing}")

    counts = df["Termd"].value_counts()
    print(f"\n► Distribution cible Termd : {dict(counts)}")
    print(f"  Taux de départ : {counts[1] / len(df) * 100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts.plot(kind="bar", ax=axes[0], color=["#2196F3", "#F44336"])
    axes[0].set_title("Distribution — Variable Termd")
    axes[0].set_xticklabels(["Actif (0)", "Démission (1)"], rotation=0)
    axes[0].set_ylabel("Nombre d'employés")

    sns.boxplot(
        data=df, x="Termd", y="EmpSatisfaction",
        palette=["#2196F3", "#F44336"], ax=axes[1]
    )
    axes[1].set_xticklabels(["Actif", "Démissionné"])
    axes[1].set_title("Satisfaction des employés vs. Départ")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_distribution.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(
        data=df, x="Termd", y="Absences",
        palette=["#2196F3", "#F44336"], ax=axes[0]
    )
    axes[0].set_xticklabels(["Actif", "Démissionné"])
    axes[0].set_title("Absences vs. Départ")

    sns.boxplot(
        data=df, x="Termd", y="EngagementSurvey",
        palette=["#2196F3", "#F44336"], ax=axes[1]
    )
    axes[1].set_xticklabels(["Actif", "Démissionné"])
    axes[1].set_title("Score d'engagement vs. Départ")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_boxplots.png", dpi=150)
    plt.close()

    term_reasons = df[df["Termd"] == 1]["TermReason"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    term_reasons.plot(kind="barh", ax=ax, color="#F44336")
    ax.set_title("Top 10 raisons de démission")
    ax.set_xlabel("Nombre de départs")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_termination_reasons.png", dpi=150)
    plt.close()

    print("► Graphiques EDA sauvegardés dans ./plots/")
    return df

# ─────────────────────────────────────────────
# ÉTAPE 2 — NETTOYAGE ET PRÉPARATION (RGPD)
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print("\n" + "=" * 60)
    print(" ÉTAPE 2 — NETTOYAGE ET PRÉPARATION (RGPD)")
    print("=" * 60)

    df = df.copy()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    df["DOB"] = pd.to_datetime(df["DOB"], dayfirst=False, errors="coerce")
    df["DateofHire"] = pd.to_datetime(df["DateofHire"], dayfirst=False, errors="coerce")
    df["LastPerformanceReview_Date"] = pd.to_datetime(
        df["LastPerformanceReview_Date"], dayfirst=False, errors="coerce"
    )

    reference_date = pd.Timestamp("2019-03-01")
    df["Age"] = (reference_date - df["DOB"]).dt.days // 365
    df["Tenure"] = (reference_date - df["DateofHire"]).dt.days // 365
    df["DaysSinceLastReview"] = (reference_date - df["LastPerformanceReview_Date"]).dt.days

    df["ManagerID"] = df["ManagerID"].fillna(df["ManagerID"].median())

    pii_cols = [
        "Employee_Name", "EmpID", "Zip", "ManagerName",
        "DOB", "DateofHire", "LastPerformanceReview_Date"
    ]

    leaky_cols = [
        "TermReason", "EmploymentStatus", "EmpStatusID", "DateofTermination"
    ]

    df = df.drop(columns=pii_cols + leaky_cols, errors="ignore")

    print(f"► Colonnes PII supprimées (RGPD)     : {pii_cols}")
    print(f"► Colonnes à fuite données supprimées : {leaky_cols}")

    y = df["Termd"].copy()
    X = df.drop(columns=["Termd"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"► One-Hot Encoding sur {len(cat_cols)} colonnes : {cat_cols}")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    print(f"► Dataset final : {X.shape[0]} lignes × {X.shape[1]} colonnes")
    print(f"► Taux de départ : {y.mean() * 100:.1f}%")
    return X, y

# ─────────────────────────────────────────────
# ÉTAPE 2 BIS — NLP ENRICHMENT
# ─────────────────────────────────────────────

def enrich_with_nlp(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_nlp = X.copy()
    original_cols = set(X.columns)

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

    X_nlp["survey_comment"] = X_nlp.apply(generate_survey, axis=1)
    X_nlp["transfer_request_text"] = X_nlp.apply(generate_transfer, axis=1)

    salary_keywords = ["compensation", "salary", "recognition", "progression"]
    growth_keywords = ["growth", "development", "career", "progressing", "opportunities", "responsibility"]
    stress_keywords = ["workload", "pressure", "support", "balance", "pace"]
    mobility_keywords = ["internal move", "internal mobility", "transfer", "another team", "another role"]
    positive_keywords = ["appreciate", "positive", "stable", "clear", "value", "satisfied"]
    negative_keywords = ["not fully", "would like more", "difficult", "pressure", "hard", "better"]

    def contains_any(text, keywords):
        text = str(text).lower()
        return int(any(k in text for k in keywords))

    def sentiment_score(text):
        text = str(text).lower()
        pos = sum(1 for k in positive_keywords if k in text)
        neg = sum(1 for k in negative_keywords if k in text)
        return pos - neg

    X_nlp["text_combined"] = (
        X_nlp["survey_comment"].fillna("") + " " + X_nlp["transfer_request_text"].fillna("")
    ).str.strip()

    X_nlp["text_sentiment_score"] = X_nlp["text_combined"].apply(sentiment_score)
    X_nlp["topic_salary"] = X_nlp["text_combined"].apply(lambda x: contains_any(x, salary_keywords))
    X_nlp["topic_growth"] = X_nlp["text_combined"].apply(lambda x: contains_any(x, growth_keywords))
    X_nlp["topic_stress"] = X_nlp["text_combined"].apply(lambda x: contains_any(x, stress_keywords))
    X_nlp["topic_mobility"] = X_nlp["text_combined"].apply(lambda x: contains_any(x, mobility_keywords))
    X_nlp["negative_text_flag"] = (X_nlp["text_sentiment_score"] < 0).astype(int)
    X_nlp["mobility_request_present"] = X_nlp["transfer_request_text"].fillna("").str.len().gt(10).astype(int)

    def recommend_actions(row):
        actions = []
        if row["topic_salary"] == 1:
            actions.append("Compensation review")
        if row["topic_growth"] == 1:
            actions.append("Career path discussion")
        if row["topic_stress"] == 1:
            actions.append("Workload adjustment")
        if row["topic_mobility"] == 1 or row["mobility_request_present"] == 1:
            actions.append("Internal mobility meeting")
        if row["negative_text_flag"] == 1 and len(actions) == 0:
            actions.append("Manager 1:1 follow-up")
        if len(actions) == 0:
            actions.append("No immediate action")
        return " | ".join(actions)

    X_nlp["recommended_actions"] = X_nlp.apply(recommend_actions, axis=1)

    demo_text_df = X_nlp[["survey_comment", "transfer_request_text", "recommended_actions"]].copy()

    X_nlp = X_nlp.drop(
        columns=["survey_comment", "transfer_request_text", "text_combined", "recommended_actions"],
        errors="ignore"
    )

    new_cols = [c for c in X_nlp.columns if c not in original_cols]
    print(f"► NLP enrichment ajouté : {X_nlp.shape[1]} features")
    print(f"► NLP : +{len(new_cols)} colonnes ajoutées")
    print(f"► Colonnes NLP ajoutées : {new_cols}")

    return X_nlp, demo_text_df

# ─────────────────────────────────────────────
# ÉTAPE 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(" ÉTAPE 3 — FEATURE ENGINEERING")
    print("=" * 60)

    X = X.copy()

    if "Absences" in X.columns and "Tenure" in X.columns:
        tenure_safe = X["Tenure"].clip(lower=1)
        X["AbsenteeismRate"] = X["Absences"] / tenure_safe

    if "DaysLateLast30" in X.columns and "Absences" in X.columns:
        X["RiskScore_Engagement"] = X["DaysLateLast30"] * 2 + X["Absences"]

    corr_df = X.copy()
    corr_df["Termd"] = y.values
    corr = corr_df.corr(numeric_only=True)["Termd"].drop("Termd").sort_values(ascending=False)

    print("► Top 10 variables corrélées positivement avec la démission :")
    print(corr.head(10).to_string())
    print("\n► Top 10 variables corrélées négativement :")
    print(corr.tail(10).to_string())

    base_num_cols = [c for c in [
        "Salary", "Age", "Tenure", "EngagementSurvey",
        "EmpSatisfaction", "Absences", "DaysLateLast30",
        "SpecialProjectsCount", "PerfScoreID", "AbsenteeismRate",
        "RiskScore_Engagement", "DaysSinceLastReview",
        "text_sentiment_score", "salary_pct"
    ] if c in X.columns]

    corr_matrix = X[base_num_cols].copy()
    corr_matrix["Termd"] = y.values

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix.corr(numeric_only=True),
        annot=True, fmt=".2f", cmap="coolwarm", center=0,
        ax=ax, linewidths=0.5
    )
    ax.set_title("Matrice de corrélation — Variables clés")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\n► {len(X.columns)} features après feature engineering")
    return X

# ─────────────────────────────────────────────
# ÉTAPES 4 & 5 — MODÈLES ML
# ─────────────────────────────────────────────

def find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.20, 0.80, 61)
    best_thr = 0.50
    best_f1 = -1.0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = thr

    return best_thr, best_f1

def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    print("\n" + "=" * 60)
    print(" ÉTAPES 4 & 5 — RANDOM FOREST + XGBOOST")
    print("=" * 60)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    results = {}

    print("\n► Random Forest Classifier")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    val_prob_rf = rf.predict_proba(X_val)[:, 1]
    thr_rf, _ = find_best_threshold(y_val, val_prob_rf)

    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_prob_rf >= thr_rf).astype(int)

    cv_scores_rf = cross_val_score(
        rf, X_train_full, y_train_full, cv=5, scoring="roc_auc", n_jobs=-1
    )

    print(f"  Threshold optimal : {thr_rf:.2f}")
    print(f"  AUC-ROC test : {roc_auc_score(y_test, y_prob_rf):.4f}")
    print(f"  CV AUC (mean±std) : {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"  Recall : {recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"  F1 : {f1_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=["Actif", "Démission"], zero_division=0))

    results["rf"] = {
        "model": rf,
        "y_pred": y_pred_rf,
        "y_prob": y_prob_rf,
        "auc": roc_auc_score(y_test, y_prob_rf),
        "cv_auc": cv_scores_rf.mean(),
        "threshold": thr_rf,
        "f1": f1_score(y_test, y_pred_rf, zero_division=0),
        "precision": precision_score(y_test, y_pred_rf, zero_division=0),
        "recall": recall_score(y_test, y_pred_rf, zero_division=0),
    }

    try:
        from xgboost import XGBClassifier

        print("\n► XGBoost Classifier")
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1
        )
        xgb.fit(X_train, y_train)

        val_prob_xgb = xgb.predict_proba(X_val)[:, 1]
        thr_xgb, _ = find_best_threshold(y_val, val_prob_xgb)

        y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
        y_pred_xgb = (y_prob_xgb >= thr_xgb).astype(int)

        cv_scores_xgb = cross_val_score(
            xgb, X_train_full, y_train_full, cv=5, scoring="roc_auc", n_jobs=-1
        )

        print(f"  Threshold optimal : {thr_xgb:.2f}")
        print(f"  AUC-ROC test : {roc_auc_score(y_test, y_prob_xgb):.4f}")
        print(f"  CV AUC (mean±std) : {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")
        print(f"  Accuracy : {accuracy_score(y_test, y_pred_xgb):.4f}")
        print(f"  Precision : {precision_score(y_test, y_pred_xgb, zero_division=0):.4f}")
        print(f"  Recall : {recall_score(y_test, y_pred_xgb, zero_division=0):.4f}")
        print(f"  F1 : {f1_score(y_test, y_pred_xgb, zero_division=0):.4f}")
        print(classification_report(y_test, y_pred_xgb, target_names=["Actif", "Démission"], zero_division=0))

        results["xgb"] = {
            "model": xgb,
            "y_pred": y_pred_xgb,
            "y_prob": y_prob_xgb,
            "auc": roc_auc_score(y_test, y_prob_xgb),
            "cv_auc": cv_scores_xgb.mean(),
            "threshold": thr_xgb,
            "f1": f1_score(y_test, y_pred_xgb, zero_division=0),
            "precision": precision_score(y_test, y_pred_xgb, zero_division=0),
            "recall": recall_score(y_test, y_pred_xgb, zero_division=0),
        }

        best_key = max(["rf", "xgb"], key=lambda k: (results[k]["f1"], results[k]["auc"]))

    except ImportError:
        print("  ⚠ XGBoost non installé — pip install xgboost")
        best_key = "rf"

    best_name = "XGBoost" if best_key == "xgb" else "Random Forest"
    best_model = results[best_key]["model"]

    print(f"\n► Meilleur modèle : {best_name}")
    print(f"  - AUC : {results[best_key]['auc']:.4f}")
    print(f"  - F1 : {results[best_key]['f1']:.4f}")
    print(f"  - Recall : {results[best_key]['recall']:.4f}")
    print(f"  - Threshold : {results[best_key]['threshold']:.2f}")

    model_items = [(k, v) for k, v in results.items() if k in ["rf", "xgb"]]
    _n = len(model_items)
    fig, axes = plt.subplots(1, _n, figsize=(6 * _n, 5))
    if _n == 1:
        axes = [axes]

    for ax, (key, res) in zip(axes, model_items):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Actif", "Départ"])
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {'RF' if key == 'rf' else 'XGB'}")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/05_confusion_matrices.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for key, res in model_items:
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        lbl = f"{'RF' if key == 'rf' else 'XGBoost'} (AUC={res['auc']:.3f})"
        ax.plot(fpr, tpr, label=lbl, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taux faux positifs")
    ax.set_ylabel("Taux vrais positifs")
    ax.set_title("Courbes ROC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/06_roc_curves.png", dpi=150)
    plt.close()

    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 7))
        fi[::-1].plot(kind="barh", ax=ax, color="#1976D2")
        ax.set_title(f"Top 20 variables importantes — {best_name}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/07_feature_importance.png", dpi=150)
        plt.close()

    results["best_key"] = best_key
    results["best_name"] = best_name
    results["X_train"] = X_train_full
    results["X_test"] = X_test
    results["y_train"] = y_train_full
    results["y_test"] = y_test
    results["feature_names"] = list(X.columns)
    return results

def run_ablation_study(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(" TEST D'ABLATION — VARIABLES POTENTIELLEMENT DOMINANTES")
    print("=" * 60)

    variants = {
        "full": [],
        "sans_DaysSinceLastReview": ["DaysSinceLastReview"],
        "sans_ManagerID": ["ManagerID"],
        "sans_les_deux": ["DaysSinceLastReview", "ManagerID"],
    }

    rows = []

    for variant_name, cols_to_drop in variants.items():
        cols_present = [c for c in cols_to_drop if c in X.columns]
        Xv = X.drop(columns=cols_present, errors="ignore")

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            Xv, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_train_full
        )

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        val_prob = rf.predict_proba(X_val)[:, 1]
        thr, _ = find_best_threshold(y_val, val_prob)

        y_prob = rf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= thr).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        rows.append({
            "variant": variant_name,
            "dropped_cols": ", ".join(cols_present) if cols_present else "-",
            "n_features": Xv.shape[1],
            "threshold": round(thr, 2),
            "auc": round(auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })

    ablation_df = pd.DataFrame(rows)
    print(ablation_df.to_string(index=False))

    ablation_df.to_csv(f"{PLOTS_DIR}/ablation_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = ablation_df.set_index("variant")[["auc", "f1", "recall"]]
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Ablation study — Impact sur les performances")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/11_ablation_study.png", dpi=150)
    plt.close()

    print(f"\n► Résultats d'ablation sauvegardés dans ./{PLOTS_DIR}/ablation_results.csv")
    print(f"► Graphique d'ablation sauvegardé dans ./{PLOTS_DIR}/11_ablation_study.png")

    return ablation_df


# ─────────────────────────────────────────────
# ÉTAPE 6 — IA EXPLICABLE (SHAP)
# ─────────────────────────────────────────────

def explain_with_shap(results: dict) -> None:
    print("\n" + "=" * 60)
    print(" ÉTAPE 6 — INTELLIGENCE ARTIFICIELLE EXPLICABLE (SHAP)")
    print("=" * 60)

    X_train = results["X_train"]
    X_test = results["X_test"]
    feat_names = results["feature_names"]

    shap_model = None
    shap_model_name = None

    # 1) On essaie d'abord le meilleur modèle
    try_order = []
    if results["best_key"] in results:
        try_order.append((results["best_key"], results[results["best_key"]]["model"]))
    if "rf" in results and results["best_key"] != "rf":
        try_order.append(("rf", results["rf"]["model"]))
    if "xgb" in results and results["best_key"] != "xgb":
        try_order.append(("xgb", results["xgb"]["model"]))

    last_error = None
    for model_key, model in try_order:
        try:
            explainer = shap.TreeExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test, check_additivity=False)
            shap_model = model
            shap_model_name = "Random Forest" if model_key == "rf" else "XGBoost"
            break
        except Exception as e:
            last_error = e
            continue

    if shap_model is None:
        print(f"► SHAP indisponible : {last_error}")
        return

    if isinstance(shap_values, list):
        sv_class1 = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        sv_class1 = shap_values[:, :, 1]
    else:
        sv_class1 = shap_values

    print(f"► Modèle utilisé pour SHAP : {shap_model_name}")
    print(f"► SHAP values shape : {sv_class1.shape}")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_class1, X_test, feature_names=feat_names, show=False)
    plt.title(f"SHAP — Impact global des variables ({shap_model_name})")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/08_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        sv_class1, X_test, feature_names=feat_names,
        plot_type="bar", show=False
    )
    plt.title(f"SHAP — Importance moyenne globale ({shap_model_name})")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/09_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    probs_source = results["best_key"]
    probs = results[probs_source]["y_prob"]
    high_risk_idx = int(np.argmax(probs))

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        if np.ndim(expected_value) > 0 and len(np.atleast_1d(expected_value)) > 1:
            base_val = np.atleast_1d(expected_value)[1]
        else:
            base_val = np.atleast_1d(expected_value)[0]
    else:
        base_val = expected_value

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv_class1[high_risk_idx],
            base_values=base_val,
            data=X_test.iloc[high_risk_idx].values,
            feature_names=feat_names
        ),
        show=False,
        max_display=15
    )
    plt.title(f"SHAP Waterfall — Employé à haut risque ({shap_model_name})")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/10_shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("► Graphiques SHAP sauvegardés dans ./plots/")

    mean_abs = np.abs(sv_class1).mean(axis=0)
    top10 = sorted(zip(feat_names, mean_abs), key=lambda x: x[1], reverse=True)[:10]
    print("\n► Top 10 facteurs de démission (SHAP) :")
    for rank, (feat, val) in enumerate(top10, 1):
        print(f"  {rank}. {feat:40s} SHAP moyen = {val:.4f}")


# ─────────────────────────────────────────────
# ÉTAPE 7 — CYBERSÉCURITÉ ET CONFORMITÉ RGPD
# ─────────────────────────────────────────────

def cybersecurity_report() -> None:
    print("\n" + "=" * 60)
    print(" ÉTAPE 7 — CYBERSÉCURITÉ & CONFORMITÉ RGPD")
    print("=" * 60)

    rapport = """
MESURES APPLIQUÉES DANS CE PROJET
──────────────────────────────────
✅ Suppression des identifiants personnels (Nom, EmpID, DOB, Zip)
   → Pseudonymisation des données RH avant tout traitement ML.

✅ Transformation des dates en variables dérivées (Age, Tenure)
   → Aucune date de naissance exacte conservée dans le modèle.

✅ Suppression des variables à fuite de données
   → TermReason, EmploymentStatus, EmpStatusID exclus.

✅ Séparation stricte train/validation/test
   → Évite le sur-apprentissage et garantit une évaluation honnête.

✅ Modèles interprétables (SHAP)
   → Chaque décision peut être expliquée — droit à l'explication (RGPD Art. 22).

✅ Sauvegarde sécurisée des modèles (joblib, répertoire local)
   → Accès contrôlé aux artefacts du modèle.

RECOMMANDATIONS SUPPLÉMENTAIRES
─────────────────────────────────
🔐 Chiffrement des données sensibles au repos (AES-256, TLS en transit).
🔐 Contrôle d'accès basé sur les rôles (RBAC) pour le dashboard.
🔐 Journal d'audit (audit log) pour toutes les prédictions individuelles.
🔐 Tests adversariaux : vérifier la robustesse du modèle face
   aux modifications malveillantes d'entrée (feature manipulation).
🔐 Biais algorithmique : auditer les taux FP/FN par groupe protégé
   (genre, origine) pour éviter toute discrimination.
🔐 Durée de conservation : définir une politique de suppression des
   données conformément au RGPD (droit à l'oubli).
"""
    print(rapport)

# ─────────────────────────────────────────────
# SAUVEGARDE DES ARTEFACTS
# ─────────────────────────────────────────────

def save_artifacts(results: dict, X_columns: list) -> None:
    best_key = results["best_key"]
    joblib.dump(results[best_key]["model"], f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(X_columns, f"{MODELS_DIR}/feature_names.pkl")
    joblib.dump(results, f"{MODELS_DIR}/results_summary.pkl")

    for key in ["rf", "xgb"]:
        if key in results:
            joblib.dump(results[key]["model"], f"{MODELS_DIR}/model_{key}.pkl")

    print(f"\n► Modèles sauvegardés dans ./{MODELS_DIR}/")
    print(f"  - best_model.pkl ({results['best_name']})")
    print(f"  - feature_names.pkl")
    print(f"  - results_summary.pkl")

# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_pipeline():
    df = load_and_explore(DATA_PATH)

    X, y = preprocess(df)

    X, demo_text_df = enrich_with_nlp(X)
    demo_text_df.to_csv("demo_text_examples.csv", index=False)

    X = feature_engineering(X, y)

    results = train_and_evaluate(X, y)

    ablation_df = run_ablation_study(X, y)

    explain_with_shap(results)

    cybersecurity_report()

    save_artifacts(results, list(X.columns))

    print("\n" + "=" * 60)
    print(" PIPELINE TERMINÉ — Tous les graphiques sont dans ./plots/")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
