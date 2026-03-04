# 💳 SmartSpend AI

An enterprise-grade **expense anomaly detection system** powered by **XGBoost** machine learning and a rule-based risk engine. Built with **Streamlit** and **SQL Server LocalDB**, SmartSpend AI enables a full 3-role approval lifecycle with real-time AI risk scoring on every submitted expense.

---

## 🚀 Features

- **AI Risk Scoring** — XGBoost model + 9 business rules produce a 0–100 risk score per expense
- **3-Role Workflow** — Employee → Manager (Approve/Reject) → Finance Officer (Reimburse)
- **Live Lifecycle Timeline** — Visual step tracker showing submission, manager decision, and reimbursement
- **System Performance Dashboard** — Model metrics, SHAP feature importance, confusion matrix, and live data trends
- **Secure Authentication** — PBKDF2-HMAC-SHA256 password hashing (200k iterations)
- **CSV Export** — Download filtered expense reports with proper date formatting
- **Dark-themed UI** — Fully custom dark CSS throughout all pages

---

## 🧠 ML Model

| Metric    | XGBoost | Decision Tree |
| --------- | ------- | ------------- |
| AUC-ROC   | 90.47%  | 89.1%         |
| Recall    | 78%     | 63%           |
| Precision | ~72%    | ~68%          |
| F1 Score  | ~75%    | ~65%          |

**Primary Algorithm:** XGBoost (`n_estimators=300`, `learning_rate=0.05`, `max_depth=6`, `scale_pos_weight=4.69`, predict threshold `0.40`)  
**Comparison Model:** Decision Tree (`max_depth=6`) — used in System Performance dashboard  
**Training Data:** 6,000 synthetic rule-based rows (self-contained, no external CSV needed)  
**Features:** `Exp_TotalAmount`, `IsAffidavit`, `ExpCat_ID`, `Dept_ID`, `Submission_Gap`, `Is_Weekend`, `Is_Round_Number`

---

## ⚙️ ML Pipeline & Normalization

Both models are wrapped in a **scikit-learn `Pipeline`** that applies `StandardScaler` before the classifier. This ensures features are normalized consistently during both training and inference — the scaler is fitted only on the training split and stored as part of the pipeline object.

```
Raw Features
     │
     ▼
┌─────────────────────────────┐
│  StandardScaler             │  ← fitted on X_train only
│  (zero mean, unit variance) │
└────────────┬────────────────┘
             │  scaled features
             ▼
┌─────────────────────────────┐
│  XGBClassifier              │  ← produces raw probability 0–1
│  (or DecisionTreeClassifier)│
└────────────┬────────────────┘
             │  predict_proba[:, 1]
             ▼
       ML Risk Score (%)
             │
             ▼
┌─────────────────────────────┐
│  Business Rules Engine      │  ← 9 policy rules add 0–25 pts each
│  (apply_business_rules)     │     capped at 100 pts total
└────────────┬────────────────┘
             │
             ▼
     Final Risk Score (0–100)
     → Triage: Low / Standard Review / Audit Flag
```

### Why StandardScaler?

XGBoost is tree-based and is theoretically scale-invariant, but applying `StandardScaler` in the pipeline provides two key benefits:

1. **Consistent preprocessing** — both the primary model (XGBoost) and the comparison model (Decision Tree) pass through the same normalization step, making benchmark comparisons fair.
2. **Future-proofing** — if a linear model (e.g., Logistic Regression, SVM) is ever added as a third comparison, the pipeline does not need to change.

### Key Pipeline Improvements Over Prior Version

| Area                       | Before (LightGBM v1)             | After (XGBoost v2)                       |
| -------------------------- | -------------------------------- | ---------------------------------------- |
| Algorithm                  | LightGBM (GBDT)                  | XGBoost (GBDT with regularization)       |
| Preprocessing              | None — features passed raw       | `StandardScaler` in `Pipeline`           |
| Model storage              | Standalone `.pkl`                | Full `Pipeline` object (scaler + model)  |
| Inference consistency      | Manual feature casting in app    | Pipeline handles scaling internally      |
| Feature: `Is_Weekend`      | Used submission date (bug)       | Correctly uses **purchase date**         |
| Feature: `Is_Round_Number` | `amount % 100 == 0` (mismatch)   | `amount % 500 == 0` (matches training)   |
| ML anomaly bridge          | Fired on accumulated score (bug) | Fires on **raw ML score only** (correct) |
| AUC-ROC                    | 90.9%                            | **90.47%** (stable with normalization)   |
| Recall (anomaly detection) | 77%                              | **78%**                                  |

---

## 🗂️ Project Structure

```
python_projects/
├── app.py                        # Entry point — login gate + home dashboard
├── db.py                         # Database layer (pyodbc → SQL Server LocalDB)
├── utils.py                      # Shared constants, business rules, ML inference, CSS
├── train_model.py                # XGBoost + Decision Tree training script
├── db_migrate_v2.sql             # Database migration script (run once)
├── requirements.txt
├── xgb_model.pkl                 # Trained XGBoost pipeline (generated by train_model.py)
├── dt_model.pkl                  # Trained Decision Tree pipeline (generated by train_model.py)
├── model_metrics.json            # Saved model metrics (generated by train_model.py)
└── pages/
    ├── 1_Submit_Expense.py       # Submit + AI-analyse expense form
    ├── 2_Reports_Log.py          # Role-based reports table + modal lifecycle view
    └── 3_System_Performance.py   # Model metrics, charts, SHAP, live trends
```

---

## 🛠️ Prerequisites

- **Python 3.10+** (developed on Python 3.13.7)
- **SQL Server LocalDB** — install via [SQL Server Express](https://www.microsoft.com/en-us/sql-server/sql-server-downloads) or the Visual Studio installer
- **ODBC Driver 17 for SQL Server** — [download here](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/DuckworthL/GoEstil_SmartSpend.git
cd GoEstil_SmartSpend
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up the database

Open **SQL Server Management Studio** (or `sqlcmd`) and run the migration script:

```sql
-- Creates SmartSpendDB, all tables, views, and demo users
sqlcmd -S "(localdb)\MSSQLLocalDB" -i db_migrate_v2.sql
```

Or open `db_migrate_v2.sql` in SSMS and execute it against `(localdb)\MSSQLLocalDB`.

### 5. Train the ML model

```bash
python train_model.py
```

This generates `xgb_model.pkl`, `dt_model.pkl`, and `model_metrics.json`.

### 6. Run the app

```bash
streamlit run app.py --server.port 8502
```

Open your browser at **http://localhost:8502**

---

## 👤 Demo Accounts

| Username    | Password       | Role            |
| ----------- | -------------- | --------------- |
| `employee1` | `Employee@123` | Employee        |
| `manager1`  | `Manager@123`  | Manager         |
| `fo1`       | `Finance@123`  | Finance Officer |

---

## 📋 Expense Lifecycle

```
Employee submits expense
        ↓
  [Pending Manager Review]
        ↓
  Manager Approves ──→ [Approved by Manager] ──→ Finance Officer Reimburses ──→ [Reimbursed]
  Manager Rejects  ──→ [Rejected by Manager] (Employee can resubmit)
```

---

## 🏷️ Risk Score Thresholds

| Score Range | Triage Level       | Gauge Color |
| ----------- | ------------------ | ----------- |
| 0 – 39      | ✅ Low Risk        | Green       |
| 40 – 69     | ⚠️ Standard Review | Orange      |
| 70 – 100    | 🚨 Audit Flag      | Red         |

---

## 🏢 Departments & Expense Categories

**Departments:** Sales (101), Engineering (102), Marketing (103), Finance (104), HR/Operations (105)

**Categories:** Meals & Entertainment, Transportation, Accommodation, Office Supplies, Software/Subscriptions, Training & Development, Medical/Wellness, Miscellaneous

---

## 🔑 Business Rules Engine

The final risk score is computed in two stages:

**Stage 1 — XGBoost ML Score:** The trained pipeline (StandardScaler → XGBClassifier) outputs a raw anomaly probability (0–100%). This raw score is saved before any rule boosts.

**Stage 2 — Policy Rule Boosts:** Nine business rules add penalty points on top of the ML base score. The ML anomaly bridge (Rule 9) fires only if the _raw ML score_ ≥ 40%, not the accumulated total.

| #   | Rule                                   | Boost                 |
| --- | -------------------------------------- | --------------------- |
| 1   | No receipt / affidavit filed           | +15 pts               |
| 2   | Future or same-day purchase date       | +25 pts               |
| 3   | Late submission (> 60 days)            | +15 pts               |
| 4   | Purchase date on a weekend             | +20 pts               |
| 5   | Meals & Entertainment > ₱5,000         | +20 pts               |
| 6   | Round number amount (multiple of ₱500) | +15 pts               |
| 7   | Amount ≥ ₱50,000                       | +15 pts               |
| 7   | Amount ≥ ₱100,000                      | +25 pts (replaces #7) |
| 9   | XGBoost raw probability ≥ 40%          | +30 pts (ML bridge)   |

Final score is capped at 100.

---

## 🧪 Sample Test Cases

| Scenario       | Details                                                        | Expected Score |
| -------------- | -------------------------------------------------------------- | -------------- |
| ✅ Low Risk    | ₱1,250 Meals · Engineering · 3-day gap · with receipt          | ~2–15%         |
| ⚠️ Medium Risk | ₱3,000 Office Supplies · no receipt · round amount · 8-day gap | ~40–55%        |
| 🚨 High Risk   | ₱25,000 Client Dinner · Sales · same-day · no receipt          | ~95–100%       |

---

## 🖥️ Tech Stack

| Layer    | Technology                                 |
| -------- | ------------------------------------------ |
| Frontend | Streamlit 1.54, Plotly, custom HTML/CSS    |
| ML Model | XGBoost 3.2.0, scikit-learn Pipeline, SHAP |
| Database | SQL Server LocalDB (pyodbc)                |
| Auth     | PBKDF2-HMAC-SHA256                         |
| Language | Python 3.13.7                              |

---

## 📄 License

This project is for educational and demonstration purposes.
