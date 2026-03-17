# HR Turnover Predictor

A machine learning project that predicts whether an employee is likely to leave a company, and explains **why** using AI.

Built for the **HR AI Hackathon**.

---

## What does it do?

It takes an employee's HR data (salary, engagement, absences, performance...) and outputs:
- A **risk score** (e.g. 72% chance of leaving)
- An **explanation** of which factors matter most (using SHAP)
- A **what-if simulator** to test the impact of HR actions (raise salary, improve engagement...)

---

## How does it work?

```
HR Data (311 employees)
        ↓
  Clean & prepare data
        ↓
  Train a Random Forest model
        ↓
  Calibrate probabilities
        ↓
  Explain with SHAP
        ↓
  Show results in a dashboard
```

**Random Forest** is a model that builds hundreds of decision trees and combines their results for a more reliable prediction.
**SHAP** explains each prediction by showing how much each variable pushed the score up or down.

---

## Project files

```
HackathonIARH/
├── HRDataset_v14.csv   # Dataset (311 employees, 36 columns)
├── hr_pipeline.py      # Full ML pipeline (training, SHAP, reports)
├── dashboard.py        # Interactive Streamlit dashboard
├── requirements.txt    # Python dependencies
├── MODEL_CARD.md       # Model documentation
└── README.md           # This file
```

---

## Installation

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

---

## How to run

**Run the full pipeline** (trains the model, generates plots):
```bash
python hr_pipeline.py
```

**Launch the interactive dashboard:**
```bash
python -m streamlit run dashboard.py
```

Then open your browser at `http://localhost:8501`.

> On Windows, use `python -m streamlit run` instead of just `streamlit run`.

---

## Dashboard tabs

| Tab | What it shows |
|---|---|
| Overview | KPIs, turnover by department, ROC curve |
| Prediction | Enter an employee profile → get a risk score + SHAP explanation |
| AI Explanations | Which features matter most across all employees |
| Bias Analysis | Fairness metrics by gender and department |
| Security & GDPR | Privacy measures and legal notes |

---

## The data

**HRDataset_v14.csv** — a fictional HR dataset (public, for research purposes).

- 311 employees, 36 variables
- Target: `Termd` — 0 = still active, 1 = left the company
- ~33% turnover rate

Some columns were **removed** to keep the model honest:
- `TermReason`, `DateofTermination` → only known *after* someone leaves (data leakage)
- `Employee_Name`, `EmpID`, `DOB` → personal identifiable info (GDPR)
- `Position`, `State` → too many unique values for 311 rows (adds noise)

---

## Model performance

| Metric | Score |
|---|---|
| AUC-ROC | 0.722 |
| Accuracy | 0.730 |
| F1 (left) | 0.605 |

The model correctly identifies ~62% of employees who actually leave.

---

## Tech stack

- **Random Forest** — prediction model
- **SHAP** — explainability
- **Streamlit** — dashboard
- **scikit-learn** — preprocessing & calibration
- **pandas / numpy** — data handling

---

*All personal data has been anonymized. This project is for educational purposes only.*
