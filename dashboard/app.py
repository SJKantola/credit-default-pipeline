import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path

from src.features.build_features import engineer_features

# ---------- Page config ----------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("🏦 Credit Default Prediction Dashboard")

# ---------- Load assets ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
model = joblib.load(MODEL_DIR / "final_xgboost_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
with open(MODEL_DIR / "model_card.json") as f:
    model_card = json.load(f)

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📈 Model Performance", "🧪 Try a Prediction", "🌲 Feature Importance"])

# ---------- Tab 1: Performance ----------
with tab1:
    st.header("Test Set Metrics")
    metrics = model_card["test_metrics"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.subheader("Confusion Matrix")
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Evaluation Curves")
    col1, col2 = st.columns(2)
    with col1:
        st.image("data/processed/final_roc_curve.png", caption="ROC Curve")
    with col2:
        st.image("data/processed/final_pr_curve.png", caption="Precision-Recall Curve")

    st.metric("Cost Savings vs. No Model", f"{model_card['business_savings_percent']}%")

# ---------- Tab 2: Manual Prediction ----------
with tab2:
    st.header("Enter Applicant Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        credit_limit = st.number_input("Credit Limit (NTD)", min_value=10000, value=200000, step=50000)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x==0 else "Female")
        education = st.selectbox("Education", [1,2,3,4], format_func=lambda x: ["Graduate","University","High School","Others"][x-1])
        marital = st.selectbox("Marital Status", [1,2,3], format_func=lambda x: ["Married","Single","Others"][x-1])
        age = st.slider("Age", 21, 80, 35)

    with col2:
        pay_sept = st.selectbox("Pay Status Sept", [-2,-1,0,1,2,3,4,5,6,7,8], index=2)
        pay_aug = st.selectbox("Pay Status Aug", [-2,-1,0,1,2,3,4,5,6,7,8], index=2)
        pay_jul = st.selectbox("Pay Status Jul", [-2,-1,0,1,2,3,4,5,6,7,8], index=2)
        pay_jun = st.selectbox("Pay Status Jun", [-2,-1,0,1,2,3,4,5,6,7,8], index=2)
        pay_may = st.selectbox("Pay Status May", [-2,-1,0,1,2,3,4,5,6,7,8], index=2)
        pay_apr = st.selectbox("Pay Status Apr", [-2,-1,0,1,2,3,4,5,6,7,8], index=2)

    with col3:
        bill_sept = st.number_input("Bill Sept", value=0)
        bill_aug = st.number_input("Bill Aug", value=0)
        bill_jul = st.number_input("Bill Jul", value=0)
        bill_jun = st.number_input("Bill Jun", value=0)
        bill_may = st.number_input("Bill May", value=0)
        bill_apr = st.number_input("Bill Apr", value=0)

    col4, col5 = st.columns(2)
    with col4:
        pay_sept_amt = st.number_input("Paid Sept", value=0)
        pay_aug_amt = st.number_input("Paid Aug", value=0)
        pay_jul_amt = st.number_input("Paid Jul", value=0)
    with col5:
        pay_jun_amt = st.number_input("Paid Jun", value=0)
        pay_may_amt = st.number_input("Paid May", value=0)
        pay_apr_amt = st.number_input("Paid Apr", value=0)

    # Build input dict
    input_data = {
        "credit_limit": credit_limit,
        "sex": sex,
        "education": education,
        "marital_status": marital,
        "age": age,
        "pay_status_sept": pay_sept,
        "pay_status_aug": pay_aug,
        "pay_status_jul": pay_jul,
        "pay_status_jun": pay_jun,
        "pay_status_may": pay_may,
        "pay_status_apr": pay_apr,
        "bill_amt_sept": bill_sept,
        "bill_amt_aug": bill_aug,
        "bill_amt_jul": bill_jul,
        "bill_amt_jun": bill_jun,
        "bill_amt_may": bill_may,
        "bill_amt_apr": bill_apr,
        "pay_amt_sept": pay_sept_amt,
        "pay_amt_aug": pay_aug_amt,
        "pay_amt_jul": pay_jul_amt,
        "pay_amt_jun": pay_jun_amt,
        "pay_amt_may": pay_may_amt,
        "pay_amt_apr": pay_apr_amt
    }

    if st.button("Predict"):
        df = pd.DataFrame([input_data])
        df_feat, _ = engineer_features(df, None)
        expected_cols = scaler.feature_names_in_
        df_feat = df_feat[expected_cols]
        scaled = scaler.transform(df_feat)
        proba = model.predict_proba(scaled)[0, 1]
        pred = int(proba >= 0.5)
        st.metric("Default Probability", f"{proba:.2%}")
        st.write("Prediction:", "**Default**" if pred else "**No Default**")

# ---------- Tab 3: Feature Importance ----------
with tab3:
    st.header("Feature Importance (XGBoost Gain)")
    
    # Get importance scores
    booster = model.get_booster()
    importance = booster.get_score(importance_type='gain')
    
    # Get the feature names the model was trained with
    # The scaler's feature_names_in_ holds the column names after feature engineering
    feature_names = list(scaler.feature_names_in_)
    
    # Map importance keys to feature names
    importance_mapped = {}
    for k, v in importance.items():
        # If the key is like 'f0', 'f1', etc., convert to integer index
        if k.startswith('f'):
            try:
                idx = int(k[1:])  # remove 'f' and convert to int
                if idx < len(feature_names):
                    importance_mapped[feature_names[idx]] = v
                else:
                    # fallback: keep original key if index out of range
                    importance_mapped[k] = v
            except ValueError:
                # key is not a simple 'f'+number, use as is
                importance_mapped[k] = v
        else:
            # already a named feature
            importance_mapped[k] = v

    # Build dataframe and sort
    imp_df = pd.DataFrame({
        "Feature": list(importance_mapped.keys()),
        "Importance": list(importance_mapped.values())
    }).sort_values("Importance", ascending=False).head(20)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=imp_df, y="Feature", x="Importance", palette="viridis", ax=ax)
    ax.set_title("Top 20 Feature Importances (gain)")
    st.pyplot(fig)

# ---------- Tab 4: SHAP Explainability ----------
with st.expander("⚡ SHAP Explainability (Why did the model decide this?)", expanded=False):
    st.write("SHAP values show how much each feature contributed to the prediction for a specific applicant.")
    
    # We'll use the same input as Tab 2 (if already filled) or a separate form
    # For simplicity, let's add a small form inside this tab
    with st.form("shap_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            credit_limit_shap = st.number_input("Credit Limit", 10000, 1000000, 200000)
            sex_shap = st.selectbox("Sex (0=Male, 1=Female)", [0, 1])
            education_shap = st.selectbox("Education", [1,2,3,4], format_func=lambda x: ["Graduate","University","High School","Others"][x-1])
        with col2:
            pay_sept_shap = st.selectbox("Payment Status Sept", [-2,-1,0,1,2,3,4,5,6,7,8])
            pay_aug_shap = st.selectbox("Payment Status Aug", [-2,-1,0,1,2,3,4,5,6,7,8])
            pay_jul_shap = st.selectbox("Payment Status Jul", [-2,-1,0,1,2,3,4,5,6,7,8])
        with col3:
            age_shap = st.slider("Age", 21, 80, 35)
            bill_sept_shap = st.number_input("Bill Sept", value=0)
        submitted_shap = st.form_submit_button("Explain Prediction")
    
    if submitted_shap:
        # Build input exactly as in Tab 2
        input_data = {
            "credit_limit": credit_limit_shap,
            "sex": sex_shap,
            "education": education_shap,
            "marital_status": 2,  # default
            "age": age_shap,
            "pay_status_sept": pay_sept_shap,
            "pay_status_aug": pay_aug_shap,
            "pay_status_jul": pay_jul_shap,
            "pay_status_jun": 0,
            "pay_status_may": 0,
            "pay_status_apr": 0,
            "bill_amt_sept": bill_sept_shap,
            "bill_amt_aug": 0,
            "bill_amt_jul": 0,
            "bill_amt_jun": 0,
            "bill_amt_may": 0,
            "bill_amt_apr": 0,
            "pay_amt_sept": 0,
            "pay_amt_aug": 0,
            "pay_amt_jul": 0,
            "pay_amt_jun": 0,
            "pay_amt_may": 0,
            "pay_amt_apr": 0
        }
        df_input = pd.DataFrame([input_data])
        df_feat, _ = engineer_features(df_input, None)
        expected_cols = scaler.feature_names_in_
        df_feat = df_feat[expected_cols]
        scaled = scaler.transform(df_feat)
        
        # Get SHAP values
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled)
        
        # Waterfall plot for the first (only) sample
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=scaled[0],
                feature_names=expected_cols
            ),
            show=False
        )
        st.pyplot(fig)
        st.write("The waterfall chart shows how each feature pushed the prediction from the base value (average).")