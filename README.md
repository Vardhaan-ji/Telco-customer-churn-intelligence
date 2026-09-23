# 📡 Telco Customer Churn — Analytics & ML Project

> **Methodology:** 4-Tier Analytics Ladder (Descriptive → Diagnostic → Predictive → Prescriptive)  
> **Dataset:** IBM Telco Customer Churn — 7,043 customers · 21 features  
> **Target:** Customer Churn (binary: Yes / No)

---

## 📂 Project Structure

```
IBM project/
├── Telco Customer Churn.csv        # Source dataset (7,043 rows × 21 columns)
├── app.py                          # Streamlit interactive dashboard (all 4 tiers)
├── requirements.txt                # Python dependency manifest
├── README.md                       # This file
└── Telco_Churn_Project_Report.docx # Formal project report
```

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run app.py
```

> The app loads `Telco Customer Churn.csv` from the **same directory** as `app.py`. Ensure both files are co-located before running. No internet connection is required.

---

## 📊 Problem Statement

Customer churn is the most costly revenue leakage vector in subscription-based telecommunications. Proactively identifying at-risk customers enables targeted retention interventions — which are substantially less expensive than acquiring new customers.

**Questions this project addresses:**
1. Which customers are at higher churn risk?
2. What observable characteristics are associated with churn?
3. How can limited retention resources be prioritised?
4. What practical retention actions could be considered?

---

## 🏗 4-Tier Analytics Ladder

### Tier 0 — Data Hygiene & Architecture

| Issue | Treatment |
|-------|-----------|
| `TotalCharges` stored as `object` | Coerced to `float64` via `pd.to_numeric` |
| 11 blank `TotalCharges` entries | New customers (tenure ≈ 0) with no accumulated charges — filled with `MonthlyCharges × 1` as a business-rule assumption, not statistical imputation |
| Duplicate `customerID` records | Deduplication applied (`keep='first'`); **0 duplicates found** |
| `SeniorCitizen` stored as 0/1 integer | Retained as-is for modelling; mapped to readable label for EDA only (label excluded from predictors) |
| Zero-tenure / zero-charge customers | **Preserved** — represent legitimate new activations |
| Negative numeric values | None detected (validated via assertion) |
| Target encoding | `Churn`: Yes → 1, No → 0 |

**Leakage Prevention:**
- `customerID` (raw identifier), `SeniorCitizenLabel` (EDA-only derived column), and `Churn` (target) excluded from feature set.
- `ColumnTransformer` + `Pipeline` used: preprocessor fitted **only on training split**, transform-only on test set.
- No test-set data used during model fitting or preprocessing.

**Preprocessing Pipeline:**
- **Numeric features** (4 columns: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`): `StandardScaler`
- **Categorical features** (15 columns): `OneHotEncoder(handle_unknown="ignore")` — avoids the false ordinal assumptions of `LabelEncoder`
- 80/20 stratified train/test split (`random_state=42`)

---

### Tier 1 & 2 — Descriptive & Diagnostic EDA

| # | Chart | Key Observation |
|---|-------|----------------|
| 1 | **Churn Distribution** | 26.5% churn rate (1,869 / 7,043); ~74/26 class imbalance — raw accuracy is misleading |
| 2 | **Churn by Contract Type** | Month-to-month: 42.7% churn; One-year: 11.3%; Two-year: 2.8% — step-down consistent with switching cost theory |
| 3 | **Tenure Distribution by Churn** | Churned customers cluster in 0–12 month range; highest-risk onboarding window |
| 4 | **Monthly Charges vs. Tenure Scatter** | High-charge, low-tenure quadrant dominated by churned customers |
| 5 | **Churn by Internet Service × Senior Status** | Fiber Optic customers: highest churn rates across all cohorts; Senior + Fiber Optic: highest observed churn rate |

> **Epistemological discipline:** All EDA observations describe associations in the dataset. No causal claims are made. Experimental methods (e.g., A/B tests) would be required to establish causality.

---

### Tier 3 — Predictive Modelling

#### Model Configuration

| Parameter | Logistic Regression | Random Forest |
|-----------|-------------------|---------------|
| Preprocessing | `OneHotEncoder` + `StandardScaler` via `ColumnTransformer` | Same |
| Class weighting | `balanced` | `balanced` |
| Key hyperparameters | `max_iter=1000` | `n_estimators=300`, `max_depth=12`, `min_samples_leaf=5` |
| Train/Test split | 80% / 20% stratified | 80% / 20% stratified |
| Random seed | 42 | 42 |

#### Final Model Evaluation — Held-out 20% Test Set

> All metrics below are computed from the actual dataset using the corrected pipeline. No values are hardcoded or approximated.

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 73.8% | 75.7% |
| Precision | 50.4% | 52.9% |
| Recall | **78.3%** | 77.0% |
| F1-Score | 61.4% | **62.7%** |
| **ROC-AUC ★** | **0.8416** | 0.8411 |

**Winner: Logistic Regression** — selected on highest ROC-AUC (0.8416 vs 0.8411).  
ROC-AUC is the primary selection criterion under class imbalance, as it is invariant to class distribution.

#### Confusion Matrices (Test Set)

| | LR: Pred Retained | LR: Pred Churned |
|---|---|---|
| **Act: Retained** | 747 (TN) | 288 (FP) |
| **Act: Churned** | 81 (FN) | 293 (TP) |

| | RF: Pred Retained | RF: Pred Churned |
|---|---|---|
| **Act: Retained** | 779 (TN) | 256 (FP) |
| **Act: Churned** | 86 (FN) | 288 (TP) |

#### False Positive vs. False Negative Trade-off

| Error Type | Business Implication |
|-----------|---------------------|
| **False Negative** (miss a churner) | Customer leaves without intervention; revenue loss is irreversible. The dataset does not contain LTV figures — actual financial impact requires external business data. |
| **False Positive** (wrongly flag a retained customer) | Unnecessary retention offer sent; bounded, controllable cost. |

**Conclusion:** False Negatives are generally more costly. Optimise for **Recall**. The optimal decision threshold is below 0.5, calibrated to your organisation's unit economics.

---

### Tier 4 — Prescriptive Strategy & Operational Levers

#### Risk Prioritisation Methodology

Full-dataset risk scores are generated using **5-fold stratified cross-validated (out-of-fold) predictions**. Each customer's score comes from a fold in which they were held out — producing honest, out-of-sample probability estimates suitable for operational triage.

> **Important distinction:** OOF risk scores are used for **customer prioritisation only**. Model performance is measured exclusively on the held-out 20% test set (reported in Tier 3).

#### Top-20% Targeting Result (OOF scores)

| Parameter | Value |
|-----------|-------|
| Customers targeted (top 20%) | 1,408 |
| Actual churners within target | 954 |
| Capture rate | **51.0%** of 1,869 total churners |

#### Illustrative Scenario Analysis

> ⚠ **The dataset contains no retention cost, LTV, acceptance rate, or causal intervention data.** The figures below use assumed unit economics to illustrate the financial logic of risk-based targeting. These are **not** empirical results.

| Assumption | Value |
|-----------|-------|
| Outreach cost per customer | $30 (assumed) |
| Offer acceptance rate | 30% (assumed) |
| Average LTV saved per retained customer | $1,800 (assumed) |
| *Illustrative* total outreach budget | $42,240 |
| *Illustrative* revenue protected | $514,620 |
| *Illustrative* net ROI | ~1,119% |

Sensitivity analysis against these parameters is essential before operational deployment.

#### Five Suggested Risk Mitigation Strategies

1. **Month-to-Month Contract Conversion** — Offer discounted annual upgrade to top-risk M2M customers (tenure < 6 months). 42.7% vs 2.8% churn rate correlation observed between M2M and Two-year contracts (causality unconfirmed).
2. **Fiber Optic Value Assurance** — Proactive service call + 3-month free add-on for top-risk Fiber Optic subscribers. Effectiveness would need to be confirmed via a controlled pilot.
3. **Early Tenure Onboarding Intervention** — Automated check-in workflow at month 3 for new subscribers scoring >40% churn probability. Targets the highest-risk tenure window.
4. **Senior Citizen Digital Concierge** — Priority support tier for Senior + Fiber Optic + electronic-payment customers. Addresses likely service-complexity friction.
5. **Electronic Check Auto-Pay Migration** — $5/month credit for switching to auto-pay. Reduces payment friction (correlation with churn observed; A/B testing recommended before broad rollout).

All strategies are **suggested actions to consider**. Causal effectiveness cannot be determined from observational data alone.

---

## ⚠ Limitations & Caveats

- **Observational data:** All findings describe associations. Randomised controlled experiments are required to establish causal relationships.
- **Probability calibration:** Model scores are calibrated for relative triage ranking. For absolute probability use, apply Platt scaling or isotonic regression.
- **Temporal validity:** Dataset is a single point-in-time snapshot. Models should be monitored and retrained as customer behaviour evolves.
- **Feature importance:** Tree-based importances can reflect feature correlation structures, not necessarily causal relevance (e.g., `tenure` and `TotalCharges` are correlated).
- **ROI figures:** All financial projections use external assumptions not present in the dataset.

---

## 🛠 Technical Stack

| Library | Purpose |
|---------|---------|
| `pandas` ≥ 1.5 | Data loading, cleaning, feature engineering |
| `numpy` ≥ 1.23 | Numerical computation |
| `scikit-learn` ≥ 1.2 | ML pipeline, preprocessing, models, metrics |
| `streamlit` ≥ 1.28 | Interactive dashboard |
| `matplotlib` ≥ 3.6 | Static charts |
| `seaborn` ≥ 0.12 | Confusion matrix heatmaps |
| `plotly` ≥ 5.15 | Available for future interactive charts |

---

## ⚙ Dashboard Sections

| Section | Content |
|---------|---------|
| **📊 KPI Overview** | 6 top-line KPIs + dynamic data quality audit |
| **🔍 EDA — Tier 1 & 2** | 5 charts with empirical observations and diagnostic notes |
| **🤖 ML Models — Tier 3** | Model comparison · Confusion matrices · ROC curves · Feature importances · FP/FN trade-off |
| **🎯 Prescriptive — Tier 4** | OOF risk tiers · Interactive targeting slider · Top-risk table · Clearly labelled illustrative scenario · 5 suggested strategies |

---

*Telco Customer Churn Analytics Project — All model metrics generated from actual dataset execution.*
