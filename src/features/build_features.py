import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.data.preprocess import load_and_clean_data


def engineer_features(X: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, list):
    """
    Create new predictive features from existing data.

    Parameters:
        X (pd.DataFrame): Cleaned features from preprocess.py.
        y (pd.Series): Target variable (not used for engineering, only passed through).

    Returns:
        X_new (pd.DataFrame): Features with new columns added.
        numeric_cols (list): Updated list of numeric columns for scaling.
    """
    df = X.copy()

    # ---- 1. Credit utilisation ratio per month ----
    # For each month, utilisation = bill_amount / credit_limit (clipped to reasonable range)
    months = ['sept', 'aug', 'jul', 'jun', 'may', 'apr']
    for m in months:
        bill_col = f'bill_amt_{m}'
        util_col = f'util_{m}'
        # Avoid division by zero; credit_limit is always ≥ 10000
        df[util_col] = df[bill_col] / df['credit_limit']
        # Clip extreme values (overpayments can make negative ratio; cap at -1 to 2)
        df[util_col] = df[util_col].clip(-1, 2)

    # ---- 2. Payment consistency score ----
    # Count how many months the client paid duly (-1) or had no consumption (-2)
    pay_cols = [f'pay_status_{m}' for m in months]
    df['payment_consistency'] = df[pay_cols].apply(
        lambda row: row.isin([-1, -2]).sum(), axis=1
    )

    # ---- 3. Update numeric columns list ----
    # All original columns + the new ones we just created
    new_feat_cols = [f'util_{m}' for m in months] + ['payment_consistency']
    all_numeric_cols = X.columns.tolist() + new_feat_cols

    return df, all_numeric_cols


def build_train_val_test(filepath: str, random_state=42):
    """
    Full feature‑engineering pipeline:
        1. Load & clean data
        2. Engineer new features
        3. Split into train/val/test (stratified, 70/15/15)
        4. Scale features (fit on train, transform all)
        5. Apply SMOTE only to training set

    Returns:
        X_train, X_val, X_test (pd.DataFrame): Scaled feature sets.
        y_train, y_val, y_test (pd.Series): Corresponding targets.
        scaler (StandardScaler): Fitted scaler for later use.
    """
    # Step 1 & 2
    X_raw, y, _ = load_and_clean_data(filepath)
    X_feat, numeric_cols = engineer_features(X_raw, y)

    # Step 3: stratified split
    # First split off test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_feat, y, test_size=0.15, stratify=y, random_state=random_state
    )
    # Then split temp into train (70% of total → 82.35% of temp) and val (15% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=random_state
    )

    # Step 4: scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
    X_val_scaled   = scaler.transform(X_val[numeric_cols])
    X_test_scaled  = scaler.transform(X_test[numeric_cols])

    # Convert back to DataFrame for readability
    X_train = pd.DataFrame(X_train_scaled, columns=numeric_cols, index=X_train.index)
    X_val   = pd.DataFrame(X_val_scaled,   columns=numeric_cols, index=X_val.index)
    X_test  = pd.DataFrame(X_test_scaled,  columns=numeric_cols, index=X_test.index)

    # Step 5: SMOTE on training set only
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_val, X_test, y_train_res, y_val, y_test, scaler