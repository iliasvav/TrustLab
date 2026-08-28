from pathlib import Path

import pandas as pd

from src.trustlab.experiment import run_experiment


DATA_DIR = Path("benchmarks/data/leakage_case")


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    normal_features = [
        column
        for column in train.columns
        if column not in {"target", "leak_feature"}
    ]

    all_features = [
        column
        for column in train.columns
        if column != "target"
    ]

    experiments = [
        ("logistic_regression", normal_features),
        ("logistic_regression", all_features),
        ("random_forest", normal_features),
        ("random_forest", all_features),
    ]

    for model_name, features in experiments:
        result = run_experiment(
            train=train,
            val=val,
            test=test,
            model_name=model_name,
            features=features,
        )

        print(
            f"{result.model_name:20} "
            f"features={len(result.features):2} "
            f"val_f1={result.validation_f1:.3f} "
            f"test_f1={result.test_f1:.3f}"
        )


if __name__ == "__main__":
    main()
