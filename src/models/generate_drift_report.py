import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.features.build_features import build_train_val_test
from evidently import Report
from evidently.presets import DataDriftPreset

X_train, _, _, _, _, _, scaler = build_train_val_test("data/raw/credit_default.xls")
reference = X_train.sample(500, random_state=42)
current = X_train.sample(500, random_state=24) * 0.98

report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=reference, current_data=current)
snapshot.save_html("data/processed/drift_report.html")
print("Drift report saved to data/processed/drift_report.html")