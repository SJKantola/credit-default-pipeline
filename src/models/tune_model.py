import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from src.features.build_features import build_train_val_test
from src.models.train_advanced import evaluate_model  # reuse our metric function


def tune_xgboost(random_state=42, n_iter=30):
    """
    Load data, run RandomizedSearchCV on XGBoost, return the best model and params.
    n_iter: number of random parameter combinations to try (30 is a good start).
    """
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = build_train_val_test(
        "data/raw/credit_default.xls", random_state=random_state
    )

    # Parameter grid for XGBoost (tree-based)
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0, 0.01, 0.1, 1],
    }

    xgb = XGBClassifier(eval_metric='logloss', random_state=random_state, n_jobs=-1)

    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    print(f"\nBest parameters (CV ROC-AUC: {search.best_score_:.4f}):")
    print(search.best_params_)

    best_model = search.best_estimator_

    # Evaluate on validation set
    metrics = evaluate_model("XGBoost (tuned)", best_model, X_val, y_val)
    print("\nValidation performance after tuning:")
    print(pd.DataFrame([metrics]).set_index("Model").round(4))

    # Also evaluate the untuned baseline for comparison
    baseline = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=random_state)
    baseline.fit(X_train, y_train)
    base_metrics = evaluate_model("XGBoost (untuned)", baseline, X_val, y_val)
    print("\nComparison with untuned:")
    comp = pd.DataFrame([base_metrics, metrics]).set_index("Model")
    print(comp.round(4))

    return best_model, search.best_params_


if __name__ == "__main__":
    tune_xgboost()