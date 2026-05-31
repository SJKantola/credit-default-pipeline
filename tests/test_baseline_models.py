from src.models.train_baseline import main

def test_baseline_models():
    results_df, lr, dt = main(random_state=42)
    # Check that we have 3 models
    assert len(results_df) == 3, f"Expected 3 models, got {len(results_df)}"
    # Check all metrics are present
    required_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    for metric in required_metrics:
        assert metric in results_df.columns, f"Missing metric: {metric}"
    # Check ROC-AUC is reasonable (dummy ~0.5, others should be > 0.65)
    # Dummy might be ~0.5, LR/DT should be better
    assert results_df.loc["Logistic Regression", "ROC-AUC"] > 0.6
    assert results_df.loc["Decision Tree", "ROC-AUC"] > 0.6