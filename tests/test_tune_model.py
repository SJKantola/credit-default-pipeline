from src.models.tune_model import tune_xgboost

def test_tune_xgboost():
    best_model, best_params = tune_xgboost(random_state=42, n_iter=5)  # small n_iter for speed
    # Check that best_params is a non-empty dict
    assert isinstance(best_params, dict) and len(best_params) > 0
    # Check that the model can predict
    from src.features.build_features import build_train_val_test
    _, X_val, _, _, y_val, _, _ = build_train_val_test("data/raw/credit_default.xls", random_state=42)
    preds = best_model.predict(X_val)
    assert len(preds) == len(y_val)
    # Check ROC-AUC on validation is > 0.7 (should be much higher)
    from sklearn.metrics import roc_auc_score
    proba = best_model.predict_proba(X_val)[:,1]
    assert roc_auc_score(y_val, proba) > 0.7