import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from src.features.build_features import build_train_val_test


def evaluate_model(name, model, X_val, y_val):
    """Return a dict of metrics for a fitted model."""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1": f1_score(y_val, y_pred),
    }
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_val, y_proba)
    else:
        metrics["ROC-AUC"] = np.nan
    return metrics


def main(random_state=42):
    # 1. Load data (already scaled, SMOTE'd train set)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = build_train_val_test(
        "data/raw/credit_default.xls", random_state=random_state
    )

    results = []

    # 2. Dummy baseline (stratified: predicts class proportions, better than most_frequent)
    dummy = DummyClassifier(strategy="stratified", random_state=random_state)
    dummy.fit(X_train, y_train)
    results.append(evaluate_model("Dummy (stratified)", dummy, X_val, y_val))

    # 3. Logistic Regression (with class_weight='balanced' since we already SMOTE'd, but still fine)
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr.fit(X_train, y_train)
    results.append(evaluate_model("Logistic Regression", lr, X_val, y_val))

    # 4. Decision Tree
    dt = DecisionTreeClassifier(random_state=random_state)
    dt.fit(X_train, y_train)
    results.append(evaluate_model("Decision Tree", dt, X_val, y_val))

    # 5. Display results
    results_df = pd.DataFrame(results).set_index("Model")
    print("Model Performance on Validation Set:")
    print(results_df.round(4))
    return results_df, lr, dt  # return models for optional inspection


if __name__ == "__main__":
    main()