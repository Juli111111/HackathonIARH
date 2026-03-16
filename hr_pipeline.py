# -*- coding: utf-8 -*-
"""
HR Turnover Prediction System
Système de prédiction des démissions d'employés
Hackathon IA RH — Pipeline complet (Étapes 1-7)

Dataset : HRDataset_v14.csv (311 employés, 36 colonnes)
Cible   : Termd (1 = départ confirmé, 0 = employé actif)
"""

import warnings
import os
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # backend non-interactif pour sauvegarder les plots
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

import shap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = "HRDataset_v14.csv"
PLOTS_DIR   = "plots"
MODELS_DIR  = "models"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# ÉTAPE 1 — ANALYSE EXPLORATOIRE DES DONNÉES
# ─────────────────────────────────────────────

def load_and_explore(path: str) -> pd.DataFrame:
    """Charge le dataset et produit une analyse descriptive."""
    print("\n" + "="*60)
    print(" ÉTAPE 1 — ANALYSE EXPLORATOIRE DES DONNÉES")
    print("="*60)

    df = pd.read_csv(path)
    print(f"\n► Dimensions : {df.shape[0]} employés × {df.shape[1]} colonnes")
    print(f"\n► Types de variables :\n{df.dtypes.value_counts()}")
    print(f"\n► Statistiques descriptives :\n{df.describe().T}")

    # Valeurs manquantes
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("\n► Aucune valeur manquante (hors DateofTermination pour actifs).")
    else:
        print(f"\n► Valeurs manquantes :\n{missing}")

    # Distribution de la variable cible
    counts = df["Termd"].value_counts()
    print(f"\n► Distribution cible Termd : {dict(counts)}")
    print(f"  Taux de départ : {counts[1]/len(df)*100:.1f}%")

    # Graphique 1 — Distribution de la cible
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts.plot(kind="bar", ax=axes[0], color=["#2196F3", "#F44336"])
    axes[0].set_title("Distribution — Variable Termd")
    axes[0].set_xticklabels(["Actif (0)", "Démission (1)"], rotation=0)
    axes[0].set_ylabel("Nombre d'employés")

    # Graphique 2 — Satisfaction vs. Départ
    sns.boxplot(
        data=df, x="Termd", y="EmpSatisfaction",
        palette=["#2196F3", "#F44336"], ax=axes[1]
    )
    axes[1].set_xticklabels(["Actif", "Démissionné"])
    axes[1].set_title("Satisfaction des employés vs. Départ")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_distribution.png", dpi=150)
    plt.close()

    # Graphique 3 — Absentéisme vs. Départ
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

    # Graphique 4 — Raisons de départ
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
    """
    Nettoie les données, respecte le RGPD et évite les fuites de données.

    ⚠ Fuites de données (data leakage) évitées :
        - TermReason      : raison post-événement
        - EmploymentStatus: encode directement l'issue
        - EmpStatusID     : idem
        - DateofTermination: disponible seulement après départ

    RGPD — colonnes supprimées (données personnelles) :
        - Employee_Name, EmpID, Zip, ManagerName, DOB → remplacé par Age
    """
    print("\n" + "="*60)
    print(" ÉTAPE 2 — NETTOYAGE ET PRÉPARATION (RGPD)")
    print("="*60)

    df = df.copy()

    # Conversion dates
    df["DOB"]      = pd.to_datetime(df["DOB"], dayfirst=False, errors="coerce")
    df["DateofHire"] = pd.to_datetime(df["DateofHire"], dayfirst=False, errors="coerce")
    df["LastPerformanceReview_Date"] = pd.to_datetime(
        df["LastPerformanceReview_Date"], dayfirst=False, errors="coerce"
    )

    # ÉTAPE 3 — Feature engineering (dates → variables utiles)
    reference_date = pd.Timestamp("2019-03-01")   # date de référence du dataset
    df["Age"]    = (reference_date - df["DOB"]).dt.days // 365
    df["Tenure"] = (reference_date - df["DateofHire"]).dt.days // 365
    df["DaysSinceLastReview"] = (
        reference_date - df["LastPerformanceReview_Date"]
    ).dt.days

    # Valeurs manquantes ManagerID → médiane
    df["ManagerID"] = df["ManagerID"].fillna(df["ManagerID"].median())

    # Colonnes PII à supprimer (RGPD)
    pii_cols = ["Employee_Name", "EmpID", "Zip", "ManagerName", "DOB",
                "DateofHire", "LastPerformanceReview_Date"]

    # Colonnes avec fuite de données (post-événement)
    leaky_cols = ["TermReason", "EmploymentStatus", "EmpStatusID",
                  "DateofTermination"]

    cols_to_drop = pii_cols + leaky_cols
    df = df.drop(columns=cols_to_drop, errors="ignore")

    print(f"► Colonnes PII supprimées (RGPD)     : {pii_cols}")
    print(f"► Colonnes à fuite données supprimées : {leaky_cols}")

    # Séparation cible / features
    y = df["Termd"].copy()
    X = df.drop(columns=["Termd"])

    # Encodage des variables catégorielles
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"► One-Hot Encoding sur {len(cat_cols)} colonnes : {cat_cols}")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Normalisation HispanicLatino peut être "Yes"/"No"/"no" → remplacer
    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    print(f"► Dataset final : {X.shape[0]} lignes × {X.shape[1]} colonnes")
    print(f"► Taux de départ : {y.mean()*100:.1f}%")
    return X, y


# ─────────────────────────────────────────────
# ÉTAPE 3 — FEATURE ENGINEERING (complémentaire)
# ─────────────────────────────────────────────

def feature_engineering(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Crée des variables additionnelles pour améliorer les modèles.
    (Age, Tenure, DaysSinceLastReview sont déjà créées dans preprocess)
    """
    print("\n" + "="*60)
    print(" ÉTAPE 3 — FEATURE ENGINEERING")
    print("="*60)

    X = X.copy()

    # Taux d'absentéisme ajusté à l'ancienneté (évite le biais des nouveaux employés)
    if "Absences" in X.columns and "Tenure" in X.columns:
        tenure_safe = X["Tenure"].clip(lower=1)  # éviter division par zéro
        X["AbsenteeismRate"] = X["Absences"] / tenure_safe

    # Score de risque composite — Late × Absences
    if "DaysLateLast30" in X.columns and "Absences" in X.columns:
        X["RiskScore_Engagement"] = (
            X["DaysLateLast30"] * 2 + X["Absences"]
        )

    # Analyse de corrélation avec la cible
    corr_df = X.copy()
    corr_df["Termd"] = y.values
    corr = corr_df.corr()["Termd"].drop("Termd").sort_values(ascending=False)

    print("► Top 10 variables corrélées positivement avec la démission :")
    print(corr.head(10).to_string())
    print("\n► Top 10 variables corrélées négativement :")
    print(corr.tail(10).to_string())

    # Heatmap de corrélation sur variables numériques de base
    base_num_cols = [c for c in [
        "Salary", "Age", "Tenure", "EngagementSurvey",
        "EmpSatisfaction", "Absences", "DaysLateLast30",
        "SpecialProjectsCount", "PerfScoreID", "AbsenteeismRate",
        "RiskScore_Engagement", "DaysSinceLastReview"
    ] if c in X.columns]

    corr_matrix = X[base_num_cols].copy()
    corr_matrix["Termd"] = y.values

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix.corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Matrice de corrélation — Variables clés")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\n► {len(X.columns)} features après feature engineering")
    return X


# ─────────────────────────────────────────────
# ÉTAPES 4 & 5 — MODÈLES ML
# ─────────────────────────────────────────────

def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Entraîne Random Forest + XGBoost, évalue et compare les deux modèles.
    Retourne un dict avec modèles, scaler et résultats.
    """
    print("\n" + "="*60)
    print(" ÉTAPES 4 & 5 — RANDOM FOREST + XGBOOST")
    print("="*60)

    # Split stratifié (préserve le ratio de classe)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Normalisation — gardée comme DataFrame pour SHAP
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr  = scaler.transform(X_test)

    X_train_sc = pd.DataFrame(X_train_arr, columns=X.columns, index=X_train.index)
    X_test_sc  = pd.DataFrame(X_test_arr,  columns=X.columns, index=X_test.index)

    results = {}

    # ── RANDOM FOREST ──────────────────────────────────
    print("\n► Random Forest Classifier")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",   # gère le déséquilibre de classes
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train_sc, y_train)
    y_pred_rf  = rf.predict(X_test_sc)
    y_prob_rf  = rf.predict_proba(X_test_sc)[:, 1]

    # Cross-validation
    cv_scores_rf = cross_val_score(
        rf, X_train_sc, y_train, cv=5, scoring="roc_auc"
    )
    print(f"  AUC-ROC test : {roc_auc_score(y_test, y_prob_rf):.4f}")
    print(f"  CV AUC (mean±std) : {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
    print(classification_report(y_test, y_pred_rf,
                                 target_names=["Actif", "Démission"]))

    results["rf"] = {
        "model": rf,
        "y_pred": y_pred_rf,
        "y_prob": y_prob_rf,
        "auc": roc_auc_score(y_test, y_prob_rf),
        "cv_auc": cv_scores_rf.mean(),
    }

    # ── XGBOOST ───────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        print("\n► XGBoost Classifier")
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
        xgb.fit(X_train_sc, y_train)
        y_pred_xgb = xgb.predict(X_test_sc)
        y_prob_xgb = xgb.predict_proba(X_test_sc)[:, 1]

        cv_scores_xgb = cross_val_score(
            xgb, X_train_sc, y_train, cv=5, scoring="roc_auc"
        )
        print(f"  AUC-ROC test : {roc_auc_score(y_test, y_prob_xgb):.4f}")
        print(f"  CV AUC (mean±std) : {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")
        print(f"  Accuracy : {accuracy_score(y_test, y_pred_xgb):.4f}")
        print(classification_report(y_test, y_pred_xgb,
                                     target_names=["Actif", "Démission"]))

        results["xgb"] = {
            "model": xgb,
            "y_pred": y_pred_xgb,
            "y_prob": y_prob_xgb,
            "auc": roc_auc_score(y_test, y_prob_xgb),
            "cv_auc": cv_scores_xgb.mean(),
        }
        best_key = "xgb" if results["xgb"]["auc"] > results["rf"]["auc"] else "rf"
    except ImportError:
        print("  ⚠ XGBoost non installé — pip install xgboost")
        best_key = "rf"

    best_name  = "XGBoost" if best_key == "xgb" else "Random Forest"
    best_model = results[best_key]["model"]
    print(f"\n► Meilleur modèle : {best_name} (AUC = {results[best_key]['auc']:.4f})")

    # Matrices de confusion
    _n = len(results)
    fig, axes = plt.subplots(1, _n, figsize=(6 * _n, 5))
    if _n == 1:
        axes = [axes]
    for ax, (key, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Actif", "Départ"])
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {'RF' if key == 'rf' else 'XGB'}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/05_confusion_matrices.png", dpi=150)
    plt.close()

    # Courbes ROC comparées
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        lbl = f"{'RF' if key=='rf' else 'XGBoost'} (AUC={res['auc']:.3f})"
        ax.plot(fpr, tpr, label=lbl, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taux faux positifs")
    ax.set_ylabel("Taux vrais positifs")
    ax.set_title("Courbes ROC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/06_roc_curves.png", dpi=150)
    plt.close()

    # Feature importance (meilleur modèle)
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(
            best_model.feature_importances_, index=X.columns
        ).sort_values(ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 7))
        fi[::-1].plot(kind="barh", ax=ax, color="#1976D2")
        ax.set_title(f"Top 20 variables importantes — {best_name}")
        ax.set_xlabel("Importance (Gini)")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/07_feature_importance.png", dpi=150)
        plt.close()

    results["best_key"]   = best_key
    results["best_name"]  = best_name
    results["scaler"]     = scaler
    results["X_train"]    = X_train_sc
    results["X_test"]     = X_test_sc
    results["y_train"]    = y_train
    results["y_test"]     = y_test
    results["feature_names"] = list(X.columns)
    return results


# ─────────────────────────────────────────────
# ÉTAPE 6 — IA EXPLICABLE (SHAP)
# ─────────────────────────────────────────────

def explain_with_shap(results: dict) -> None:
    """
    Utilise SHAP pour expliquer les prédictions du meilleur modèle.
    Génère : summary plot, waterfall, bar chart.
    """
    print("\n" + "="*60)
    print(" ÉTAPE 6 — INTELLIGENCE ARTIFICIELLE EXPLICABLE (SHAP)")
    print("="*60)

    best_key   = results["best_key"]
    model      = results[best_key]["model"]
    X_train    = results["X_train"]
    X_test     = results["X_test"]
    feat_names = results["feature_names"]

    # TreeExplainer — adapté RF et XGBoost
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    # Compatibilité ancienne/nouvelle API SHAP
    # Ancienne : list[class0_arr, class1_arr]  → shape (n, p)
    # Nouvelle : ndarray 3D                    → shape (n, p, c)
    if isinstance(shap_values, list):
        sv_class1 = shap_values[1]           # classe "démission"
    elif shap_values.ndim == 3:
        sv_class1 = shap_values[:, :, 1]    # classe "démission"
    else:
        sv_class1 = shap_values             # binaire direct

    print(f"► SHAP values shape : {sv_class1.shape}")

    # 1. Summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv_class1, X_test, feature_names=feat_names, show=False)
    plt.title("SHAP — Impact global des variables (démission)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/08_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Bar chart moyen
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv_class1, X_test, feature_names=feat_names,
                      plot_type="bar", show=False)
    plt.title("SHAP — Importance moyenne globale")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/09_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Waterfall pour un employé à haut risque
    probs = results[best_key]["y_prob"]
    high_risk_idx = int(np.argmax(probs))    # employé le plus à risque
    base_val = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values       = sv_class1[high_risk_idx],
        base_values  = base_val,
        data         = X_test.iloc[high_risk_idx].values,
        feature_names= feat_names
    ), show=False, max_display=15)
    plt.title(f"SHAP Waterfall — Employé à haut risque (idx {high_risk_idx})")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/10_shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("► Graphiques SHAP sauvegardés dans ./plots/")

    # Top 5 facteurs explicatifs
    mean_abs = np.abs(sv_class1).mean(axis=0)
    top5 = sorted(zip(feat_names, mean_abs), key=lambda x: x[1], reverse=True)[:5]
    print("\n► Top 5 facteurs de démission (SHAP) :")
    for rank, (feat, val) in enumerate(top5, 1):
        print(f"  {rank}. {feat:40s}  SHAP moyen = {val:.4f}")


# ─────────────────────────────────────────────
# ÉTAPE 7 — CYBERSÉCURITÉ ET CONFORMITÉ RGPD
# ─────────────────────────────────────────────

def cybersecurity_report() -> None:
    """Affiche les mesures de sécurité appliquées et recommandations."""
    print("\n" + "="*60)
    print(" ÉTAPE 7 — CYBERSÉCURITÉ & CONFORMITÉ RGPD")
    print("="*60)

    rapport = """
MESURES APPLIQUÉES DANS CE PROJET
──────────────────────────────────
✅ Suppression des identifiants personnels (Nom, EmpID, DOB, Zip)
   → Pseudonymisation des données RH avant tout traitement ML.

✅ Transformation des dates en variables derivées (Age, Tenure)
   → Aucune date de naissance exacte conservée dans le modèle.

✅ Suppression des variables à fuite de données
   → TermReason, EmploymentStatus, EmpStatusID exclues.

✅ Séparation stricte train/test
   → Evite le sur-apprentissage et garantit une évaluation honnête.

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
# SAUVEGARDE DES MODÈLES
# ─────────────────────────────────────────────

def save_artifacts(results: dict, X_columns: list) -> None:
    """Sauvegarde modèles, scaler et feature names pour le dashboard."""
    best_key = results["best_key"]
    joblib.dump(results[best_key]["model"],  f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(results["scaler"],            f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(X_columns,                    f"{MODELS_DIR}/feature_names.pkl")

    # Sauvegarde aussi RF et XGB si disponible
    for key in ["rf", "xgb"]:
        if key in results:
            joblib.dump(results[key]["model"], f"{MODELS_DIR}/model_{key}.pkl")

    print(f"\n► Modèles sauvegardés dans ./{MODELS_DIR}/")
    print(f"  - best_model.pkl  ({results['best_name']})")
    print(f"  - scaler.pkl")
    print(f"  - feature_names.pkl")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_pipeline():
    # 1. Chargement et EDA
    df = load_and_explore(DATA_PATH)

    # 2. Préprocessing + suppression PII et variables leaky
    X, y = preprocess(df)

    # 3. Feature engineering (variables dérivées déjà créées dans preprocess,
    #    ici on ajoute les features composites)
    X = feature_engineering(X, y)

    # 4 & 5. Entraînement et évaluation des modèles
    results = train_and_evaluate(X, y)

    # 6. IA Explicable — SHAP
    explain_with_shap(results)

    # 7. Rapport cybersécurité
    cybersecurity_report()

    # Sauvegarde
    save_artifacts(results, list(X.columns))

    print("\n" + "="*60)
    print(" PIPELINE TERMINÉ — Tous les graphiques sont dans ./plots/")
    print("="*60)


if __name__ == "__main__":
    run_pipeline()
