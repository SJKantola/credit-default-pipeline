import pandas as pd
from src.features.build_features import build_train_val_test

def test_build_train_val_test():
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = build_train_val_test(
        "data/raw/credit_default.xls"
    )

    # 1. Check proportions: val and test each ~15% of original 30k
    assert 0.14 <= len(y_val) / 30000 <= 0.16
    assert 0.14 <= len(y_test) / 30000 <= 0.16

    # 2. Train set after SMOTE should be larger than original imbalanced train (21k) and roughly balanced
    assert X_train.shape[0] > 21000
    assert X_train.shape[0] == y_train.shape[0]

    # 3. Class balance of training set (SMOTE'd) should be ~50/50
    assert abs(y_train.mean() - 0.5) < 0.05

    # 4. Stratification: val and test should have ~22% default
    assert abs(y_test.mean() - 0.2212) < 0.02
    assert abs(y_val.mean()   - 0.2212) < 0.02

    # 5. Feature count: 23 original + 7 engineered = 30
    assert X_train.shape[1] == 30
    assert X_val.shape[1] == 30
    assert X_test.shape[1] == 30

    # 6. Scaler is fitted
    assert hasattr(scaler, 'mean_')