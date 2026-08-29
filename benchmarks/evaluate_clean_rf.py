from pathlib import Path

import pandas as pd

from src.trustlab.evaluator import evaluate_hidden_test


DATA_DIR = Path("benchmarks/data/clean_case")


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    features = [
        column
        for column in train.columns
        if column != "target"
    ]

    hidden_test_f1 = evaluate_hidden_test(
        train=train,
        val=val,
        test=test,
        model_name="random_forest",
        features=features,
    )

    print("CLEAN CASE: ALL-FEATURE RANDOM FOREST")
    print(f"Number of features: {len(features)}")
    print(f"Hidden test F1: {hidden_test_f1:.3f}")


if __name__ == "__main__":
    main()