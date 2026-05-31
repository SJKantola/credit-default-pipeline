from src.models.train_advanced import main

def test_advanced_models():
    results_df, rf, xgb = main(random_state=42)
    # 5 models
    assert len(results_df) == 5
    # Advanced models should outperform dummy and simple models
    assert results_df.loc["Random Forest", "ROC-AUC"] > 0.65
    assert results_df.loc["XGBoost", "ROC-AUC"] > 0.65
    # XGBoost is typically the best, but at least better than Dummy
    assert results_df.loc["XGBoost", "ROC-AUC"] > results_df.loc["Dummy (stratified)", "ROC-AUC"]