import pandas as pd

def load_and_clean_data(filepath: str):
    """
    Load raw UCI Credit Card Default Excel file, clean and rename columns,
    handle undocumented categories, and return features (X) and target (y).

    Parameters:
        filepath (str): Path to the .xls file.

    Returns:
        X (pd.DataFrame): Cleaned features (no scaling, no train/test split).
        y (pd.Series): Target variable (default next month, 0/1).
        numeric_cols (list): Names of numeric feature columns for scaling.
    """
    # 1. Load with correct header row
    df = pd.read_excel(filepath, header=1)

    # 2. Rename columns to readable names
    col_names = {
        'ID': 'id',
        'LIMIT_BAL': 'credit_limit',
        'SEX': 'sex',
        'EDUCATION': 'education',
        'MARRIAGE': 'marital_status',
        'AGE': 'age',
        'PAY_0': 'pay_status_sept',
        'PAY_2': 'pay_status_aug',
        'PAY_3': 'pay_status_jul',
        'PAY_4': 'pay_status_jun',
        'PAY_5': 'pay_status_may',
        'PAY_6': 'pay_status_apr',
        'BILL_AMT1': 'bill_amt_sept',
        'BILL_AMT2': 'bill_amt_aug',
        'BILL_AMT3': 'bill_amt_jul',
        'BILL_AMT4': 'bill_amt_jun',
        'BILL_AMT5': 'bill_amt_may',
        'BILL_AMT6': 'bill_amt_apr',
        'PAY_AMT1': 'pay_amt_sept',
        'PAY_AMT2': 'pay_amt_aug',
        'PAY_AMT3': 'pay_amt_jul',
        'PAY_AMT4': 'pay_amt_jun',
        'PAY_AMT5': 'pay_amt_may',
        'PAY_AMT6': 'pay_amt_apr',
        'default payment next month': 'target'
    }
    df.rename(columns=col_names, inplace=True)

    # 3. Clean undocumented education values (0,5,6 -> 4)
    df['education'] = df['education'].replace({0: 4, 5: 4, 6: 4})

    # 4. Clean undocumented marital_status value (0 -> 3)
    df['marital_status'] = df['marital_status'].replace({0: 3})

    # 5. Binarise sex (1 -> 0 male, 2 -> 1 female)
    df['sex'] = df['sex'].map({1: 0, 2: 1})

    # 6. Drop ID column
    df.drop('id', axis=1, inplace=True)

    # 7. Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # List of numeric columns that should be scaled later
    # Includes credit_limit, age, bill amounts, pay amounts, and also pay_status
    # (ordinal but numeric; scaling helps distance-based models, doesn't hurt trees)
    numeric_cols = X.columns.tolist()

    return X, y, numeric_cols