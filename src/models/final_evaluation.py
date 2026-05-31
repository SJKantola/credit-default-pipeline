import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, auc,
    classification_report
)
from xgboost import XGBClassifier
import joblib
import json
from datetime import datetime

from src.features.build_features import build_train_val_test


def final_evaluate(filepath="data/raw/credit_default.xls", random_state=42):
    # 1. Load data using the same pipeline
    #    We get separate train/val/test sets; we'll combine train+val for final training.
    #    But build_train_val_test already applies SMOTE to train only, val+test are untouched.
    #    For final model, we should NOT SMOTE the combined set (or we can re-run the pipeline differently).
    #    Simpler: we just retrain on all non-test data (X_temp from earlier split) with SMOTE.
    #    Actually, let's just use the full pipeline but save the combined train+val before SMOTE.
    #    We'll modify our approach: load raw data, engineer features, split off test (15%),
    #    then train a new model on the remaining 85% using SMOTE.
    #    That avoids using the already SMOTE'd train set from the tuning step.

    from src.data.preprocess import load_and_clean_data
    from src.features.build_features import engineer_features
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE

    # Load & engineer
    X_raw, y, _ = load_and_clean_data(filepath)
    X_feat, numeric_cols = engineer_features(X_raw, y)

    # Split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_feat, y, test_size=0.15, stratify=y, random_state=random_state
    )

    # Scale
    scaler = StandardScaler()
    X_temp_scaled = scaler.fit_transform(X_temp[numeric_cols])
    X_test_scaled = scaler.transform(X_test[numeric_cols])

    # Convert to DataFrame
    X_train_full = pd.DataFrame(X_temp_scaled, columns=numeric_cols, index=X_temp.index)
    X_test_final = pd.DataFrame(X_test_scaled, columns=numeric_cols, index=X_test.index)

    # SMOTE on the full training portion
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_full, y_temp)

    # 2. Train final model with best parameters (from tuning)
    best_params = {
        'subsample': 0.9,
        'reg_lambda': 0,
        'reg_alpha': 0.01,
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'colsample_bytree': 0.9,
        'eval_metric': 'logloss',
        'random_state': random_state
    }
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train_res, y_train_res)

    # 3. Predict on test set
    y_pred = final_model.predict(X_test_final)
    y_proba = final_model.predict_proba(X_test_final)[:, 1]

    # 4. Metrics
    print("=== Classification Report (Test Set) ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 5. Plots
    sns.set_style("whitegrid")

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(5,5))
    ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"]).plot(ax=ax, cmap='Blues')
    ax.set_title("Confusion Matrix – Test Set")
    plt.savefig("data/processed/final_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Test Set')
    plt.legend()
    plt.savefig("data/processed/final_roc_curve.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label='XGBoost')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve – Test Set')
    plt.legend()
    plt.savefig("data/processed/final_pr_curve.png", dpi=150, bbox_inches='tight')
    plt.show()

    # 6. Cost-Benefit Analysis
    # Assumptions: False Negative (missed default) costs €5,000 (loss)
    #              False Positive (reject good customer) costs €1,000 (lost profit)
    # These numbers are illustrative; you can adjust.
    cost_fn = 5000
    cost_fp = 1000

    tn, fp, fn, tp = cm.ravel()
    total_cost_no_model = (y_test.sum()) * cost_fn  # Everyone defaults → we lose only on actual defaults (or a naive policy that accepts everyone)
    # Actually, the "no model" baseline: accept all → losses only from defaults:
    cost_no_model = y_test.sum() * cost_fn
    # Model cost: false negatives × cost_fn + false positives × cost_fp
    cost_model = fn * cost_fn + fp * cost_fp
    savings = cost_no_model - cost_model
    savings_percent = (savings / cost_no_model) * 100

    print("\n=== Cost-Benefit Analysis ===")
    print(f"Assumptions: FN cost = €{cost_fn}, FP cost = €{cost_fp}")
    print(f"No model (accept all) cost: €{cost_no_model:,.0f}")
    print(f"Model cost: €{cost_model:,.0f}")
    print(f"Savings: €{savings:,.0f} ({savings_percent:.1f}% reduction)")

    # 7. Save model, scaler, and model card
    joblib.dump(final_model, "models/final_xgboost_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    model_card = {
        "model": "XGBoost",
        "version": "1.0.0",
        "date": datetime.now().isoformat(),
        "dataset": "UCI Credit Card Default (30k records)",
        "preprocessing": "StandardScaler on all numeric features, SMOTE on training set",
        "hyperparameters": best_params,
        "test_metrics": {
            "accuracy": float(np.mean(y_pred == y_test)),
            "precision": float(prec.mean()),  # placeholder, we'll compute exact
            "recall": float(rec.mean()),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm.tolist()
        },
        "business_savings_percent": round(savings_percent, 1)
    }
    # Overwrite precision/recall with exact values from report
    from sklearn.metrics import precision_score, recall_score
    model_card["test_metrics"]["precision"] = float(precision_score(y_test, y_pred))
    model_card["test_metrics"]["recall"] = float(recall_score(y_test, y_pred))

    with open("models/model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)

    print("\nModel, scaler, and model card saved to 'models/' folder.")
    return final_model, roc_auc, savings_percent


if __name__ == "__main__":
    final_evaluate()