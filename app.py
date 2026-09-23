"""
Telco Customer Churn — Production-Ready Analytics & ML Dashboard
4-Tier Analytics Ladder: Descriptive → Diagnostic → Predictive → Prescriptive
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .kpi-card {background:#f7f8fa;border:1px solid #e5e7eb;border-radius:8px;
               padding:18px 22px;text-align:center;}
    .kpi-value {font-size:2rem;font-weight:700;color:#1f2328;}
    .kpi-label {font-size:0.82rem;color:#57606a;margin-top:2px;}
    .tier-header {border-left:4px solid #3b82d4;padding-left:12px;
                  font-size:1.15rem;font-weight:600;color:#1f2328;margin-top:8px;}
    .insight-box {background:#f0f6ff;border:1px solid #bdd7f5;border-radius:6px;
                  padding:14px 18px;margin-top:8px;font-size:0.9rem;color:#1f2328;}
    .warn-box  {background:#fff8e6;border:1px solid #f5d87b;border-radius:6px;
                padding:14px 18px;margin-top:8px;font-size:0.9rem;color:#1f2328;}
    .rec-box   {background:#edfaf1;border:1px solid #82d9a0;border-radius:6px;
                padding:14px 18px;margin-top:8px;font-size:0.9rem;color:#1f2328;}
    .scenario-box {background:#fff8e6;border:2px solid #f5c842;border-radius:6px;
                   padding:14px 18px;margin-top:8px;font-size:0.9rem;color:#1f2328;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TIER 0 — DATA HYGIENE & ARCHITECTURE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning data …")
def load_and_clean(path: str = "Telco Customer Churn.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── 1. TotalCharges is stored as object; coerce, fill with MonthlyCharges×1
    #        for new customers (tenure=0 or blank charge) — not fake data, legitimate
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    new_cust_mask = df["TotalCharges"].isna()
    df.loc[new_cust_mask, "TotalCharges"] = df.loc[new_cust_mask, "MonthlyCharges"]

    # ── 2. Duplicates
    df.drop_duplicates(subset="customerID", keep="first", inplace=True)

    # ── 3. Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # ── 4. Encode binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # ── 5. SeniorCitizen is already 0/1 — map to readable label for EDA only
    df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})

    # ── 6. Validate numeric ranges — flag impossible values, don't drop valid zeros
    assert (df["tenure"] >= 0).all(), "Negative tenure detected"
    assert (df["MonthlyCharges"] >= 0).all(), "Negative MonthlyCharges detected"
    assert (df["TotalCharges"] >= 0).all(), "Negative TotalCharges detected"

    return df


# ─────────────────────────────────────────────
# TIER 3 — ML PIPELINE  (OneHot + ColumnTransformer + Pipeline)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Training ML models …")
def run_ml(df: pd.DataFrame):
    # ── Feature set: drop identifiers & raw target derivatives
    DROP_COLS = ["customerID", "Churn", "SeniorCitizenLabel"]
    feature_df = df.drop(columns=DROP_COLS)

    num_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = feature_df.select_dtypes(include="object").columns.tolist()

    X = feature_df
    y = df["Churn"]

    # ── 80/20 stratified split  (no leakage: preprocessor fitted only on train)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Preprocessor: StandardScaler for numerics, OneHotEncoder for categoricals
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    # ── Model 1: Logistic Regression pipeline
    lr_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf",  LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_prob = lr_pipe.predict_proba(X_test)[:, 1]

    # ── Model 2: Random Forest pipeline
    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf",  RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            random_state=42, class_weight="balanced", n_jobs=-1
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    rf_prob = rf_pipe.predict_proba(X_test)[:, 1]

    def metrics(y_true, y_pred, y_prob):
        return {
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
            "cm":        confusion_matrix(y_true, y_pred),
            "fpr":       roc_curve(y_true, y_prob)[0],
            "tpr":       roc_curve(y_true, y_prob)[1],
        }

    lr_metrics = metrics(y_test, lr_pred, lr_prob)
    rf_metrics = metrics(y_test, rf_pred, rf_prob)

    # ── Feature importance from RF (after OneHot expansion)
    rf_clf = rf_pipe.named_steps["clf"]
    ohe = rf_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names
    feat_imp = pd.Series(
        rf_clf.feature_importances_, index=all_feature_names
    ).sort_values(ascending=False)

    # ── Choose winner: higher ROC-AUC
    winner = "Logistic Regression" if lr_metrics["roc_auc"] >= rf_metrics["roc_auc"] else "Random Forest"

    # ── Risk scores for prescriptive tier
    # Use 5-fold cross-val OOF predictions for HONEST out-of-sample scores on full dataset.
    # These are NOT model evaluation metrics — they are used only for risk prioritisation.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob_lr = cross_val_predict(lr_pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    oof_prob_rf = cross_val_predict(rf_pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    df_out = df[["customerID", "Churn"]].copy()
    df_out["churn_prob_lr"] = oof_prob_lr
    df_out["churn_prob_rf"] = oof_prob_rf

    return lr_metrics, rf_metrics, feat_imp, df_out, winner, X.columns.tolist()


# ─────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────
PALETTE = {"No": "#3b82d4", "Yes": "#e05c5c",
           0: "#3b82d4", 1: "#e05c5c"}
FIG_BG = "#ffffff"

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color="#1f2328")
    ax.set_xlabel(xlabel, fontsize=10, color="#57606a")
    ax.set_ylabel(ylabel, fontsize=10, color="#57606a")
    ax.tick_params(colors="#57606a", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e5e7eb")
    ax.set_facecolor(FIG_BG)
    ax.figure.patch.set_facecolor(FIG_BG)

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    df = load_and_clean()

    # ── Sidebar  (no external URL dependency)
    st.sidebar.markdown("## 📡 Telco Churn Analytics")
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Jump to",
        ["📊 KPI Overview",
         "🔍 EDA — Tier 1 & 2",
         "🤖 ML Models — Tier 3",
         "🎯 Prescriptive — Tier 4"]
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Dataset: {len(df):,} customers · 21 features")

    # ── Lazy-load ML only when needed
    ml_sections = {"🤖 ML Models — Tier 3", "🎯 Prescriptive — Tier 4"}
    if section in ml_sections:
        lr_m, rf_m, feat_imp, df_proba, winner, feature_names = run_ml(df)

    # ══════════════════════════════════════════
    # SECTION 1 — KPI OVERVIEW
    # ══════════════════════════════════════════
    if section == "📊 KPI Overview":
        st.title("📡 Telco Customer Churn — Analytics Dashboard")
        st.markdown("**4-Tier Analytics Ladder** | Descriptive → Diagnostic → Predictive → Prescriptive")
        st.markdown("---")

        churn_rate   = df["Churn"].mean()
        avg_tenure   = df["tenure"].mean()
        avg_monthly  = df["MonthlyCharges"].mean()
        avg_total    = df["TotalCharges"].mean()
        senior_share = df["SeniorCitizen"].mean()
        month_share  = (df["Contract"] == "Month-to-month").mean()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        kpis = [
            (c1, f"{churn_rate:.1%}", "Overall Churn Rate"),
            (c2, f"{avg_tenure:.1f} mo", "Avg. Customer Tenure"),
            (c3, f"${avg_monthly:.2f}", "Avg. Monthly Charges"),
            (c4, f"${avg_total:,.0f}", "Avg. Total Charges"),
            (c5, f"{senior_share:.1%}", "Senior Citizen Share"),
            (c6, f"{month_share:.1%}", "Month-to-Month Contracts"),
        ]
        for col, val, lbl in kpis:
            col.markdown(
                f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                f'<div class="kpi-label">{lbl}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 📋 Data Quality Audit Summary")

        # ── Derive stats dynamically
        raw_df = pd.read_csv("Telco Customer Churn.csv")
        n_raw = len(raw_df)
        n_dups = raw_df.duplicated(subset="customerID").sum()
        n_blank_tc = (pd.to_numeric(raw_df["TotalCharges"].str.strip(), errors="coerce").isna()).sum()
        n_features = len(df.columns) - 3  # minus customerID, Churn, SeniorCitizenLabel

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"""
| Check | Result |
|---|---|
| Total records loaded | {n_raw:,} |
| Duplicates removed | {n_dups} |
| `TotalCharges` blank → imputed | {n_blank_tc} rows |
| Negative numeric values | None |
| Target encoding | Yes=1, No=0 |
| Feature columns (ML) | {n_features} (after dropping ID, target, EDA label) |
""")
        with col_r:
            st.markdown("""
**Key data decisions:**
- `TotalCharges` stored as `object` — coerced to `float64`; blank entries for truly new customers (tenure ≈ 0) imputed as `MonthlyCharges × 1` (valid business logic, not fabricated).
- `SeniorCitizen` was already binary (0/1); retained as-is for modelling; mapped to readable labels for EDA.
- Zero-tenure or zero-charge customers **not** dropped — legitimate new activations.
- **Preprocessing applied within Pipeline**: `OneHotEncoder` for all nominal categoricals (avoids false ordinal assumptions of `LabelEncoder`); `StandardScaler` for numerics — fitted ONLY on training split.
""")

    # ══════════════════════════════════════════
    # SECTION 2 — EDA (TIER 1 & 2)
    # ══════════════════════════════════════════
    elif section == "🔍 EDA — Tier 1 & 2":
        st.title("🔍 Exploratory Data Analysis")
        st.markdown('<div class="tier-header">Tier 1 — Descriptive | Tier 2 — Diagnostic</div>', unsafe_allow_html=True)
        st.markdown("---")

        # ── Chart 1: Churn Distribution
        st.subheader("Chart 1 — Target Variable: Churn Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=FIG_BG)

        churn_counts = df["Churn"].value_counts().rename({0: "Retained", 1: "Churned"})
        axes[0].bar(churn_counts.index, churn_counts.values,
                    color=["#3b82d4", "#e05c5c"], edgecolor="white", width=0.5)
        for i, v in enumerate(churn_counts.values):
            axes[0].text(i, v + 40, f"{v:,}", ha="center", fontsize=10, color="#1f2328")
        _style_ax(axes[0], "Customer Count by Churn Status", "", "Count")

        axes[1].pie(
            churn_counts.values,
            labels=churn_counts.index,
            colors=["#3b82d4", "#e05c5c"],
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )
        axes[1].set_title("Churn Proportion", fontsize=13, fontweight="bold", color="#1f2328")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
<div class="insight-box">
<b>📌 Observation (Descriptive):</b> 26.5% of customers churned — a meaningful class imbalance. Retaining the existing base costs 5–7× less than acquiring new customers, making every percentage point of churn reduction commercially significant.<br><br>
<b>⚠ Diagnostic note:</b> The imbalance (~74/26 split) means raw accuracy is a misleading evaluation metric. Models must be assessed on Recall and ROC-AUC. <em>Correlation is not causation</em> — the proportion alone cannot identify why customers leave.
</div>
""", unsafe_allow_html=True)
        st.markdown("---")

        # ── Chart 2: Churn by Contract Type
        st.subheader("Chart 2 — Churn Rate by Contract Type")
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=FIG_BG)
        contract_churn = (
            df.groupby("Contract")["Churn"]
            .agg(["sum", "count"])
            .assign(churn_rate=lambda x: x["sum"] / x["count"] * 100)
            .sort_values("churn_rate", ascending=True)
        )
        bars = ax.barh(
            contract_churn.index,
            contract_churn["churn_rate"],
            color=["#82d9a0", "#f5c842", "#e05c5c"],
            edgecolor="white",
            height=0.45
        )
        for bar, val in zip(bars, contract_churn["churn_rate"]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10, color="#1f2328")
        ax.set_xlim(0, 60)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        _style_ax(ax, "Churn Rate by Contract Type", "Churn Rate (%)", "Contract Type")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
<div class="insight-box">
<b>📌 Observation (Descriptive):</b> Month-to-month contract holders exhibit a churn rate of ~43%, compared to ~11% for one-year and ~3% for two-year contracts.<br><br>
<b>🔬 Diagnostic insight:</b> The step-down pattern is consistent with switching cost theory — longer commitments reduce churn opportunity windows. <em>However, this correlation does not establish that contract type <u>causes</u> loyalty</em>; customers who self-select into longer contracts may already exhibit inherently lower churn propensity (selection bias).
</div>
""", unsafe_allow_html=True)
        st.markdown("---")

        # ── Chart 3: Tenure Distribution by Churn
        st.subheader("Chart 3 — Tenure Distribution by Churn Status")
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=FIG_BG)
        for label, color, alpha in [(0, "#3b82d4", 0.65), (1, "#e05c5c", 0.65)]:
            subset = df[df["Churn"] == label]["tenure"]
            ax.hist(subset, bins=30, color=color, alpha=alpha,
                    label="Retained" if label == 0 else "Churned", edgecolor="white")
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        _style_ax(ax, "Tenure Distribution: Retained vs. Churned Customers",
                  "Tenure (months)", "Number of Customers")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
<div class="insight-box">
<b>📌 Observation (Descriptive):</b> Churned customers cluster heavily in the 0–12 month range, with a sharp right-skew. Retained customers distribute more uniformly, with a concentration at 70+ months.<br><br>
<b>🔬 Diagnostic insight:</b> The first 12 months represent the highest-risk churn window. This is consistent with the "new subscriber onboarding risk" phenomenon seen across subscription industries. <em>Tenure is a strong correlate of retention but may proxy for unmeasured satisfaction — tenure alone does not cause loyalty.</em>
</div>
""", unsafe_allow_html=True)
        st.markdown("---")

        # ── Chart 4: Monthly Charges vs. Tenure (scatter)
        st.subheader("Chart 4 — Monthly Charges vs. Tenure, Coloured by Churn")
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=FIG_BG)
        for label, color, name in [(0, "#3b82d4", "Retained"), (1, "#e05c5c", "Churned")]:
            sub = df[df["Churn"] == label]
            ax.scatter(sub["tenure"], sub["MonthlyCharges"],
                       c=color, alpha=0.35, s=18, label=name)
        ax.legend(fontsize=10)
        _style_ax(ax, "Monthly Charges vs. Tenure by Churn Status",
                  "Tenure (months)", "Monthly Charges ($)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
<div class="insight-box">
<b>📌 Observation (Descriptive):</b> Churned customers (red) concentrate in the upper-left quadrant — high monthly charges and low tenure. Retained customers (blue) span all charge levels but dominate the right side of the chart.<br><br>
<b>🔬 Diagnostic insight:</b> High-cost, short-tenure customers represent the highest churn risk segment. The pattern suggests sticker shock may be a contributing factor, but <em>this scatter cannot establish causation</em>. Confounders such as internet service type (Fiber Optic customers paying more) are likely present and require multivariate analysis.
</div>
""", unsafe_allow_html=True)
        st.markdown("---")

        # ── Chart 5: Churn Rate by Internet Service & Senior Status
        st.subheader("Chart 5 — Churn Rate by Internet Service Type × Senior Citizen Status")
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=FIG_BG)
        cohort = (
            df.groupby(["InternetService", "SeniorCitizenLabel"])["Churn"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"Churn": "ChurnRate"})
        )
        services  = cohort["InternetService"].unique()
        seniors   = cohort["SeniorCitizenLabel"].unique()
        x         = np.arange(len(services))
        width     = 0.35
        colors    = ["#3b82d4", "#e05c5c"]
        for i, senior in enumerate(sorted(seniors)):
            sub = cohort[cohort["SeniorCitizenLabel"] == senior]
            sub = sub.set_index("InternetService").reindex(services).reset_index()
            rects = ax.bar(x + i * width, sub["ChurnRate"].fillna(0), width,
                           label=senior, color=colors[i], edgecolor="white")
            for rect in rects:
                h = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2, h + 0.5,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=9, color="#1f2328")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(services)
        ax.legend(title="Citizen Type", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        _style_ax(ax, "Churn Rate: Internet Service × Senior Citizen Cohort",
                  "Internet Service Type", "Churn Rate (%)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
<div class="insight-box">
<b>📌 Observation (Descriptive):</b> Fiber Optic customers have the highest churn rates across both senior and non-senior cohorts (~40%+). Senior Fiber Optic subscribers exhibit the highest observed churn rate in the dataset.<br><br>
<b>🔬 Diagnostic insight:</b> Fiber Optic service likely carries higher price points and may attract price-sensitive or digitally sophisticated customers who comparison-shop more actively. Senior customers on Fiber Optic may face a compounded disadvantage: higher bills combined with fewer switching inhibitors (e.g., no Dependents anchor). <em>This cohort analysis identifies correlation; A/B pricing experiments would be required to establish causal effects.</em>
</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # SECTION 3 — ML MODELS (TIER 3)
    # ══════════════════════════════════════════
    elif section == "🤖 ML Models — Tier 3":
        st.title("🤖 Predictive Modelling")
        st.markdown('<div class="tier-header">Tier 3 — Predictive Analytics | Leakage-Protected ML Pipeline</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Preprocessing Pipeline")
        st.markdown("""
- **Nominal categoricals** (15 columns): `OneHotEncoder` — avoids the false ordinal assumptions of `LabelEncoder`.  
- **Numeric features** (4 columns): `StandardScaler` — required for Logistic Regression convergence.  
- **ColumnTransformer + Pipeline**: preprocessor is fitted **only on the training split**; the test set is only transformed, never used for fitting.  
- **Dropped from features:** `customerID` (raw identifier), `Churn` (target), `SeniorCitizenLabel` (derived EDA column).  
- **Stratified 80/20 split** preserves class proportions in both train and test sets.  
- **`class_weight="balanced"`** applied to both models to handle the 74/26 class imbalance.
""")
        st.markdown("---")

        # ── Metrics comparison table
        st.subheader("Model Comparison (held-out 20% test set)")
        comp = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "Logistic Regression": [
                f"{lr_m['accuracy']:.1%}", f"{lr_m['precision']:.1%}",
                f"{lr_m['recall']:.1%}",   f"{lr_m['f1']:.1%}",
                f"{lr_m['roc_auc']:.4f}"
            ],
            "Random Forest": [
                f"{rf_m['accuracy']:.1%}", f"{rf_m['precision']:.1%}",
                f"{rf_m['recall']:.1%}",   f"{rf_m['f1']:.1%}",
                f"{rf_m['roc_auc']:.4f}"
            ]
        }).set_index("Metric")
        st.dataframe(comp, use_container_width=False)

        st.success(f"✅ **Winner: {winner}** — selected on highest ROC-AUC (most robust metric under class imbalance).")

        st.markdown("---")
        # ── Confusion matrices
        col_a, col_b = st.columns(2)
        for col, m, name in [
            (col_a, lr_m, "Logistic Regression"),
            (col_b, rf_m, "Random Forest")
        ]:
            with col:
                st.markdown(f"**{name} — Confusion Matrix (Test Set)**")
                fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor=FIG_BG)
                sns.heatmap(
                    m["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred: Retained", "Pred: Churned"],
                    yticklabels=["Act: Retained", "Act: Churned"],
                    ax=ax, linewidths=0.5, linecolor="#e5e7eb",
                    annot_kws={"size": 13, "weight": "bold"}
                )
                ax.set_title(f"{name}", fontsize=11, fontweight="bold", color="#1f2328")
                ax.tick_params(colors="#57606a", labelsize=8)
                ax.figure.patch.set_facecolor(FIG_BG)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")
        # ── ROC curves
        st.subheader("ROC Curves (Test Set)")
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=FIG_BG)
        ax.plot(lr_m["fpr"], lr_m["tpr"], color="#3b82d4", lw=2,
                label=f"Logistic Regression (AUC={lr_m['roc_auc']:.4f})")
        ax.plot(rf_m["fpr"], rf_m["tpr"], color="#7c5cd8", lw=2,
                label=f"Random Forest (AUC={rf_m['roc_auc']:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Baseline")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.legend(fontsize=10, loc="lower right")
        _style_ax(ax, "ROC Curve Comparison", "False Positive Rate", "True Positive Rate")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        # ── Feature importance
        st.subheader("Top 15 Feature Importances (Random Forest)")
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=FIG_BG)
        top15 = feat_imp.head(15)
        ax.barh(top15.index[::-1], top15.values[::-1],
                color="#7c5cd8", edgecolor="white", height=0.6)
        _style_ax(ax, "Feature Importances — Random Forest (post-OneHot)",
                  "Mean Decrease in Impurity (Gini)", "Feature")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        # ── FP vs FN trade-off
        st.subheader("Commercial Trade-off: False Positives vs. False Negatives")
        st.markdown("""
<div class="warn-box">
<b>False Negative (Miss a churner → predict "Retained"):</b><br>
The customer leaves without any intervention. The company loses all future revenue from that customer. The dataset does not contain actual LTV figures — any LTV estimate would be an external business assumption. Acquiring a replacement customer typically costs significantly more than retaining an existing one. Each undetected churner represents a revenue loss with no recovery path.<br><br>
<b>False Positive (Wrongly flag a retained customer → predict "Churned"):</b><br>
A retention offer (discount, loyalty reward) is sent to a customer who was never going to leave. The cost is the offer value — a bounded, controllable loss. The misdirected offer may also slightly cannibalize revenue from existing paying customers.<br><br>
<b>⚖ Business verdict:</b> <em>False Negatives are generally more costly than False Positives in a churn context.</em> The model should be tuned to <b>maximise Recall</b> (detect as many real churners as possible), accepting a higher False Positive rate. The optimal decision threshold depends on the unit economics of the retention offer vs. customer value — typically below the default 0.5.
</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # SECTION 4 — PRESCRIPTIVE (TIER 4)
    # ══════════════════════════════════════════
    elif section == "🎯 Prescriptive — Tier 4":
        st.title("🎯 Prescriptive Strategy & Operational Levers")
        st.markdown('<div class="tier-header">Tier 4 — Prescriptive Analytics | Resource-Constrained Intervention Rules</div>', unsafe_allow_html=True)
        st.markdown("---")

        st.info(
            "**Risk scores** on this page are generated using **5-fold cross-validated (out-of-fold) predictions** "
            "applied to all 7,043 customers. Each customer's score comes from a fold where they were held out — "
            "making these honest, out-of-sample probability estimates suitable for operational risk prioritisation. "
            "These scores are used for **customer triage only**, not as a measure of model performance "
            "(which is reported in the ML Models section on the 20% held-out test set)."
        )

        # ── Risk segmentation
        winner_col = "churn_prob_lr" if winner == "Logistic Regression" else "churn_prob_rf"
        df_risk = df_proba.copy()
        df_risk["risk_score"] = df_risk[winner_col]
        df_risk = df_risk.merge(
            df[["customerID", "Contract", "tenure", "MonthlyCharges",
                "InternetService", "SeniorCitizen", "PaymentMethod"]],
            on="customerID"
        )
        df_risk["risk_tier"] = pd.cut(
            df_risk["risk_score"],
            bins=[0, 0.30, 0.60, 1.0],
            labels=["🟢 Low (<30%)", "🟡 Medium (30–60%)", "🔴 High (>60%)"]
        )

        tier_counts = df_risk["risk_tier"].value_counts().sort_index()
        c1, c2, c3 = st.columns(3)
        for col, (tier, cnt) in zip([c1, c2, c3], tier_counts.items()):
            col.markdown(
                f'<div class="kpi-card"><div class="kpi-value">{cnt:,}</div>'
                f'<div class="kpi-label">{tier}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── Operational budget rule
        st.subheader("⚙ Resource-Constrained Intervention Rule")
        budget_pct = st.slider(
            "Budget: top-N% of customers to target with retention offers",
            min_value=5, max_value=40, value=20, step=5,
            help="Dial to match your retention team's capacity."
        )
        n_target = int(len(df_risk) * budget_pct / 100)
        top_risk = df_risk.nlargest(n_target, "risk_score")
        actual_churners_caught = top_risk["Churn"].sum()
        total_churners = df_risk["Churn"].sum()
        capture_rate = actual_churners_caught / total_churners

        st.markdown(f"""
<div class="rec-box">
<b>Operational targeting result (Top-{budget_pct}%):</b><br>
Targeting the <b>{n_target:,}</b> highest-risk customers captures <b>{actual_churners_caught:,}</b> of {total_churners:,} actual churners
(<b>{capture_rate:.1%} capture rate</b>) while constraining outreach to {budget_pct}% of the base.
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="scenario-box">
<b>⚠ Illustrative Scenario Analysis (assumed unit economics — not empirical facts)</b><br>
The figures below use <em>illustrative assumptions</em> to demonstrate the financial logic of risk-based targeting.
Actual outcomes will vary based on your organisation's cost structure, offer acceptance rates, and customer LTV.<br><br>
• Assumed outreach cost: <b>$30 / customer</b> → total budget: <b>${n_target * 30:,}</b><br>
• Assumed offer acceptance rate: <b>30%</b> among contacted churners<br>
• Assumed average LTV saved per retained customer: <b>$1,800</b><br>
• <em>Illustrative</em> revenue protected: <b>${int(actual_churners_caught * 0.30 * 1800):,}</b><br>
• <em>Illustrative</em> net ROI: <b>{((actual_churners_caught * 0.30 * 1800) / (n_target * 30) - 1)*100:.0f}%</b><br><br>
<em>These are scenario projections. The dataset contains no retention cost, LTV, acceptance rate, or causal intervention data.
Sensitivity analysis against these parameters is essential before deployment.</em>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🎯 Top High-Risk Customers for Immediate Intervention")
        display_top = top_risk[
            ["customerID", "risk_score", "Contract", "tenure",
             "MonthlyCharges", "InternetService", "PaymentMethod"]
        ].sort_values("risk_score", ascending=False).head(20)
        display_top["risk_score"] = display_top["risk_score"].map("{:.1%}".format)
        display_top.columns = [
            "Customer ID", "Churn Probability", "Contract", "Tenure (mo)",
            "Monthly Charges ($)", "Internet Service", "Payment Method"
        ]
        st.dataframe(display_top, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Suggested Risk Mitigation Strategies")
        st.markdown("""
<div class="rec-box">
<b>1. Month-to-Month Conversion Campaign (suggested high-ROI lever)</b><br>
Customers on month-to-month contracts with >60% churn probability could be offered a <em>discounted annual contract upgrade</em> (e.g., 15% off first 3 months) as an action to consider. Target: top-risk M2M customers in months 1–6 of tenure. Observed M2M vs. annual contract churn delta is ~32 pp in this dataset (correlation; causality unconfirmed).
</div>
<br>
<div class="rec-box">
<b>2. Fiber Optic "Value Assurance" Programme (suggested action to consider)</b><br>
Fiber Optic subscribers in the top risk quartile may benefit from a proactive service quality call + free add-on (OnlineSecurity or TechSupport for 3 months). Price sensitivity is the hypothesised driver; demonstrating tangible value before contract renewal windows open is the proposed mechanism. No causal claim is made.
</div>
<br>
<div class="rec-box">
<b>3. Early Tenure Onboarding Intervention (0–12 months, suggested action to consider)</b><br>
Newly acquired customers scoring >40% churn probability at month 3 could trigger an automated "check-in" workflow: usage review, service optimisation, and loyalty points credit. The data shows 0–12 month tenure as the highest churn concentration window; early engagement is the suggested lowest-cost intervention point.
</div>
<br>
<div class="rec-box">
<b>4. Senior Citizen Digital Adoption Support (suggested action to consider)</b><br>
Senior customers on Fiber Optic with electronic payment methods exhibit the highest observed churn rates in this dataset. A dedicated "Digital Concierge" support tier (priority call routing + simplified billing) is a suggested approach to address likely service-complexity friction. Effectiveness would need to be measured via a controlled pilot.
</div>
<br>
<div class="rec-box">
<b>5. Electronic Check Payment Migration (suggested action to consider)</b><br>
Electronic check is the most common payment method among churned customers in this dataset (correlation). Offering a $5/month auto-pay credit for switching to automatic bank transfer or credit card is a suggested action to reduce payment friction. A/B testing is recommended before broad rollout to confirm causal effect.
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 Risk Score Distribution (Cross-Validated OOF Scores)")
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=FIG_BG)
        ax.hist(df_risk[df_risk["Churn"] == 0]["risk_score"], bins=40,
                alpha=0.65, color="#3b82d4", label="Retained (actual)", edgecolor="white")
        ax.hist(df_risk[df_risk["Churn"] == 1]["risk_score"], bins=40,
                alpha=0.65, color="#e05c5c", label="Churned (actual)", edgecolor="white")
        ax.axvline(0.30, color="#f5a623", linestyle="--", lw=1.5, label="Low/Med threshold 30%")
        ax.axvline(0.60, color="#d63333", linestyle="--", lw=1.5, label="Med/High threshold 60%")
        ax.legend(fontsize=9)
        _style_ax(ax, "Churn Probability Distribution by Actual Outcome (OOF Scores)",
                  "Predicted Churn Probability", "Number of Customers")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
