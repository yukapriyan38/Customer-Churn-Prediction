import json
import shutil
from pathlib import Path

import pandas as pd


MODEL_DIR = Path("models")

METRIC_FILES = [
    MODEL_DIR / "logistic_regression_metrics.json",
    MODEL_DIR / "random_forest_metrics.json",
    MODEL_DIR / "xgboost_metrics.json",
]

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib",
}


def main():
    results = []

    for metric_file in METRIC_FILES:
        if not metric_file.exists():
            print(f"Missing file: {metric_file}")
            print("Run all three model files before running compare_models.py")
            return

        with open(metric_file, "r", encoding="utf-8") as file:
            results.append(json.load(file))

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(by="ROC-AUC", ascending=False)
    comparison.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    best_model_name = comparison.iloc[0]["Model"]
    best_model_path = MODEL_FILES[best_model_name]
    best_model_copy = MODEL_DIR / "best_model.joblib"

    if best_model_path.exists():
        shutil.copy2(best_model_path, best_model_copy)

    print("\nMODEL COMPARISON")
    print("=" * 80)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("=" * 80)
    print(f"Selected model based on highest ROC-AUC: {best_model_name}")
    print(f"Best model saved to: {best_model_copy}")
    print(f"Comparison saved to: {MODEL_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
