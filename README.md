# HR Turnover Predictor — Système de Prédiction des Démissions

Système d'intelligence artificielle appliqué aux ressources humaines, capable de prédire
le risque de départ d'un employé à partir de ses données RH, d'expliquer les facteurs
déterminants, et de simuler l'effet d'actions correctives.

Développé dans le cadre du **Hackathon IA Ressources Humaines**.

---

## Sommaire

1. [Objectif du projet](#1-objectif-du-projet)
2. [Architecture du système](#2-architecture-du-système)
3. [Structure des fichiers](#3-structure-des-fichiers)
4. [Installation](#4-installation)
5. [Utilisation](#5-utilisation)
6. [Description des étapes (Pipeline)](#6-description-des-étapes-pipeline)
7. [Description du Dashboard](#7-description-du-dashboard)
8. [Méthodologie ML](#8-méthodologie-ml)
9. [IA Explicable — SHAP](#9-ia-explicable--shap)
10. [Sécurité et conformité RGPD](#10-sécurité-et-conformité-rgpd)
11. [Technologies utilisées](#11-technologies-utilisées)

---

## 1. Objectif du projet

Les départs non anticipés d'employés représentent un coût élevé pour les entreprises
(estimé entre 6 et 18 mois de salaire pour remplacer un poste qualifié).
Ce projet fournit un **outil d'aide à la décision RH** permettant de :

- **Prédire** la probabilité qu'un employé quitte l'entreprise dans un avenir proche
- **Expliquer** les facteurs qui contribuent à ce risque (SHAP)
- **Simuler** l'impact d'actions correctives (revalorisation salariale, amélioration de l'engagement…)
- **Auditer** les biais du modèle par genre et par département (conformité AI Act UE)
- **Respecter** la vie privée des employés (pseudonymisation, conformité RGPD)

---

## 2. Architecture du système

```
HRDataset_v14.csv
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  hr_pipeline.py  —  Pipeline d'analyse complet (8 étapes)           │
│                                                                     │
│  Étape 1 : Analyse exploratoire des données (EDA)                   │
│  Étape 2 : Nettoyage, anonymisation RGPD, suppression fuites        │
│  Étape 3 : Feature engineering (Age, Tenure, AbsenteeismRate…)      │
│  Étapes 4-5 : Random Forest + XGBoost, évaluation, ROC              │
│  Étape 6 : IA Explicable — SHAP (summary, waterfall, bar)           │
│  Étape 7 : Rapport cybersécurité et conformité                      │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼  (modèles sauvegardés dans ./models/)
┌─────────────────────────────────────────────────────────────────────┐
│  dashboard.py  —  Interface Streamlit interactive                   │
│                                                                     │
│  Tab 1 : Vue Globale (KPIs, graphiques, courbe ROC, calibration)    │
│  Tab 2 : Prédiction individuelle + SHAP + Simulateur What-If        │
│  Tab 3 : Explications IA (beeswarm, bar chart, Gini)                │
│  Tab 4 : Analyse des Biais (fairness par genre et département)      │
│  Tab 5 : Sécurité & RGPD                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Structure des fichiers

```
HackathonIARH/
├── HRDataset_v14.csv          # Dataset source (311 employés, 36 colonnes)
├── hr_pipeline.py             # Pipeline ML complet (étapes 1 à 7)
├── dashboard.py               # Application Streamlit interactive (étape 8)
├── requirements.txt           # Dépendances Python
├── README.md                  # Documentation du projet
├── ExplainabilityAI.ipynb     # Notebook exploratoire initial
├── models/                    # Modèles entraînés (créé par hr_pipeline.py)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── plots/                     # Graphiques générés (créé par hr_pipeline.py)
    ├── 01_distribution.png
    ├── 02_boxplots.png
    ├── ...
    └── 10_shap_waterfall.png
```

---

## 4. Installation

### Prérequis

- Python 3.9 ou supérieur
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Contenu de `requirements.txt`

| Bibliothèque | Version minimale | Rôle |
|---|---|---|
| pandas | 2.0 | Manipulation des données |
| numpy | 1.24 | Calcul numérique |
| scikit-learn | 1.3 | Modèles ML, calibration |
| xgboost | 1.7 | Modèle XGBoost (optionnel) |
| shap | 0.44 | IA Explicable |
| matplotlib | 3.7 | Visualisations |
| seaborn | 0.12 | Graphiques statistiques |
| streamlit | 1.28 | Dashboard interactif |
| joblib | 1.3 | Sauvegarde des modèles |

---

## 5. Utilisation

### Option A — Exécuter le pipeline complet d'analyse

```bash
python hr_pipeline.py
```

Ce script exécute les 7 premières étapes et produit :

- Les graphiques dans `./plots/`
- Les modèles entraînés dans `./models/`
- Un rapport de cybersécurité affiché dans la console

### Option B — Lancer le dashboard interactif

```bash
python -m streamlit run dashboard.py
```

> **Note Windows** : la commande `streamlit` n'est pas toujours dans le PATH.
> Utiliser systématiquement `python -m streamlit run dashboard.py`.

L'application s'ouvre dans le navigateur à l'adresse `http://localhost:8501`.

Le modèle est entraîné automatiquement au premier lancement
(environ 20-40 secondes). Les résultats sont mis en cache pour les navigations
suivantes.

---

## 6. Description des étapes (Pipeline)

### Étape 1 — Analyse exploratoire des données (EDA)

Chargement et inspection du dataset `HRDataset_v14.csv` :

- 311 employés, 36 colonnes
- Variable cible : `Termd` (0 = actif, 1 = départ confirmé) — taux de 33 %
- Analyse des distributions, valeurs manquantes, corrélations
- Visualisation des patterns liés aux départs (absentéisme, satisfaction, engagement)

### Étape 2 — Nettoyage et conformité RGPD

**Variables PII (Personally Identifiable Information) supprimées :**

| Variable | Raison |
|---|---|
| `Employee_Name` | Nom complet — identifiant direct |
| `EmpID` | Identifiant unique |
| `DOB` | Date de naissance → remplacée par `Age` |
| `Zip` | Code postal trop spécifique |
| `ManagerName` | Identifiant personnel |

**Variables à fuite de données (data leakage) supprimées :**

| Variable | Raison |
|---|---|
| `TermReason` | Raison de départ — disponible uniquement après l'événement |
| `EmploymentStatus` | Encode directement la variable cible |
| `EmpStatusID` | Idem (version numérique) |
| `DateofTermination` | Date de licenciement — post-événement |

> Ces variables rendraient le modèle inutilisable en conditions réelles :
> elles ne sont pas connues avant le départ de l'employé.

### Étape 3 — Feature engineering

Création de nouvelles variables à partir des données existantes :

| Variable créée | Formule | Intérêt |
|---|---|---|
| `Age` | `(2019-03-01 - DOB).days / 365` | Moins sensible que la date brute |
| `Tenure` | `(2019-03-01 - DateofHire).days / 365` | Ancienneté en années |
| `DaysSinceLastReview` | `(2019-03-01 - LastPerformanceReview_Date).days` | Suivi managérial |
| `AbsenteeismRate` | `Absences / max(Tenure, 1)` | Taux d'absentéisme normalisé |
| `RiskScore_Engagement` | `DaysLateLast30 * 2 + Absences` | Signal composite de désengagement |

### Étape 4 — Random Forest (modèle de base)

- Split stratifié 80/20 (`stratify=y` pour conserver le ratio de classes)
- `class_weight="balanced"` pour compenser le déséquilibre (67% actifs / 33% départs)
- Évaluation : Accuracy, Précision, Rappel, F1-Score, AUC-ROC
- Validation croisée 5-fold

### Étape 5 — XGBoost (amélioration)

- Paramétrage du `scale_pos_weight` pour le déséquilibre de classes
- Comparaison avec Random Forest via les métriques et courbes ROC
- Sélection automatique du meilleur modèle

### Étape 6 — IA Explicable (SHAP)

Voir [section dédiée](#9-ia-explicable--shap).

### Étape 7 — Cybersécurité et conformité

Rapport texte affiché en console couvrant :

- Mesures de pseudonymisation appliquées
- Recommandations de chiffrement et contrôle d'accès
- Risques liés aux systèmes d'IA (biais, adversarial attacks, model stealing)
- Obligations légales (RGPD, AI Act UE)

---

## 7. Description du Dashboard

### Onglet 1 — Vue Globale

Vue d'ensemble des données RH et des performances du modèle :

- **KPIs** : effectif total, taux de départs, satisfaction moyenne, absences moyennes
- **KPIs modèle** : AUC-ROC, Précision, Rappel, F1-Score
- Taux de départ par département (barres colorées par niveau de risque)
- Distribution de la satisfaction (violin plot actifs vs. partis)
- Absentéisme par statut (boxplot)
- Top 8 raisons de départ
- **Courbe ROC** du modèle calibré
- **Courbe de calibration** (avant vs. après calibration sigmoid)

### Onglet 2 — Prédiction

Analyse d'un profil individuel :

1. **Formulaire de saisie** : département, genre, âge, ancienneté, salaire, performance, engagement, absences…
2. **Score de risque calibré** affiché avec jauge visuelle et niveau (faible / modéré / élevé)
3. **Explication SHAP** : waterfall plot des 15 facteurs les plus impactants + tableau
4. **Simulateur d'actions correctives (What-If)** :
   - Curseurs : ajustement salarial, engagement, absences, satisfaction
   - Comparaison graphique risque initial vs. risque simulé
   - Message contextuel d'interprétation

### Onglet 3 — Explications IA

Vue globale de l'explicabilité du modèle :

- **SHAP Beeswarm** : impact de chaque variable sur l'ensemble des prédictions
- **SHAP Bar chart** : importance moyenne par variable
- **Gini (Random Forest)** : importance selon le critère de Gini
- Synthèse des 5 principaux facteurs de risque

### Onglet 4 — Analyse des Biais

Audit d'équité algorithmique (fairness) :

- Métriques par **genre** (M / F) : Accuracy, Précision, Rappel, Taux FP, Taux FN
- Métriques par **département** : même indicateurs
- Alerte automatique si l'écart de Taux FP ou FN dépasse 10 % entre groupes
- Recommandations conformes à l'**AI Act UE (Art. 11)** et au **RGPD (Art. 22)**

### Onglet 5 — Sécurité & RGPD

- Récapitulatif des mesures de sécurité appliquées
- Recommandations opérationnelles (chiffrement, RBAC, journal d'audit)
- Tableau des variables exclues avec justification
- Cartographie des risques IA (biais, attaques, surconfiance…)

---

## 8. Méthodologie ML

### Modèle

**Random Forest** avec calibration des probabilités :

```
RandomForestClassifier
  ├── n_estimators = 300
  ├── class_weight = "balanced"   (déséquilibre 67/33)
  └── random_state = 42

CalibratedClassifierCV
  ├── method = "sigmoid"          (régression de Platt)
  └── cv     = "prefit"           (jeu de calibration dédié, 25% du train)
```

### Pourquoi calibrer les probabilités ?

Un modèle non calibré peut prédire "74%" mais que cela corresponde réellement à
46% de cas réels. La **calibration sigmoid** (méthode de Platt) corrige ce biais
et rend les scores directement interprétables par les équipes RH.

La courbe de calibration (onglet Vue Globale) permet de le vérifier visuellement.

### Gestion du déséquilibre de classes

| Méthode | Implémentation |
|---|---|
| `class_weight="balanced"` | Poids inversement proportionnel à la fréquence |
| Split stratifié | `stratify=y` préserve le ratio 67/33 dans train et test |

### Métriques d'évaluation

| Métrique | Pourquoi elle compte en RH |
|---|---|
| **AUC-ROC** | Mesure la capacité de classement indépendamment du seuil |
| **Rappel** | Minimiser les faux négatifs (départs non détectés) est prioritaire |
| **Précision** | Éviter de sur-alerter les RH avec de faux positifs |
| **F1-Score** | Équilibre précision / rappel |

---

## 9. IA Explicable — SHAP

### Principe

SHAP (SHapley Additive exPlanations) calcule pour chaque prédiction la contribution
de chaque variable en se basant sur la théorie des jeux coopératifs de Shapley (1953).

```
Prédiction = valeur_base + somme(SHAP_i pour chaque variable i)
```

- Valeur SHAP **positive** → pousse vers la démission
- Valeur SHAP **négative** → pousse vers le maintien en poste

### Visualisations disponibles

| Graphique | Description | Où |
|---|---|---|
| **Waterfall** | Facteurs explicatifs pour un individu | Onglet Prédiction |
| **Beeswarm** | Distribution des impacts sur l'ensemble du jeu de test | Onglet Explications IA |
| **Bar chart** | Importance moyenne globale | Onglet Explications IA |

### Conformité RGPD Art. 22

Le RGPD impose un droit à l'explication pour toute décision automatisée.
Les explications SHAP individuelles permettent de justifier chaque score
de risque auprès de l'employé concerné.

---

## 10. Sécurité et conformité RGPD

### Mesures appliquées dans le code

| Mesure | Implémentation |
|---|---|
| Pseudonymisation | Suppression de `Employee_Name`, `EmpID`, `DOB`, `Zip`, `ManagerName` |
| Prévention des fuites | `TermReason`, `EmploymentStatus`, `EmpStatusID` exclus du modèle |
| Droit à l'explication | SHAP individuel pour chaque prédiction (RGPD Art. 22) |
| Audit d'équité | Taux FP/FN calculés par genre et département |
| Calibration | Probabilités fiables, interprétables et justifiables |

### Recommandations pour la mise en production

- **Chiffrement** : AES-256 au repos, TLS 1.3 en transit
- **Contrôle d'accès** : RBAC — dashboard réservé aux RH accrédités
- **Journal d'audit** : traçabilité de chaque consultation et prédiction
- **Durée de conservation** : politique de suppression conforme au RGPD
- **Revue humaine** : toute décision impactante doit être validée par un humain
- **Re-entraînement** : audit des métriques de biais à chaque re-entraînement

---

## 11. Technologies utilisées

| Technologie | Version | Usage |
|---|---|---|
| Python | 3.9+ | Langage principal |
| scikit-learn | 1.7+ | Random Forest, calibration, métriques |
| XGBoost | 3.1+ | Modèle gradient boosting |
| SHAP | 0.51+ | IA Explicable |
| Streamlit | 1.50+ | Dashboard interactif |
| pandas | 2.3+ | Manipulation des données |
| numpy | 2.3+ | Calcul numérique |
| matplotlib | 3.10+ | Visualisations |
| seaborn | 0.13+ | Graphiques statistiques |

---

## Dataset

**HRDataset_v14.csv** — Données RH fictives à des fins de recherche

| Caractéristique | Valeur |
|---|---|
| Source | Dataset public HRDataset (Carla Patalano, New England College) |
| Effectif | 311 employés |
| Colonnes | 36 variables |
| Période | ~2006–2019 |
| Variable cible | `Termd` (0 = actif, 1 = départ confirmé) |
| Taux de départ | ~33 % |

---

*Projet Hackathon IA Ressources Humaines — Toutes les données personnelles ont été anonymisées.*
