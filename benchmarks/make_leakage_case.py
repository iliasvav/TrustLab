from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
OUTPUT_DIR = Path("benchmarks/data/leakage_case")


def build_dataset():
    X, y = make_classification(
        n_samples=6000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.03,
        random_state=RANDOM_STATE,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    train_val, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df["target"],
        random_state=RANDOM_STATE,
    )

    train, val = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val["target"],
        random_state=RANDOM_STATE,
    )

    rng = np.random.default_rng(RANDOM_STATE)

    # This is our deliberately planted trap.
    #
    # In train and validation, leak_feature is almost a direct copy
    # of the target, so a model can achieve a suspiciously high score.
    #
    # In the hidden test set, that relationship disappears.
    train["feature_10"] = (
        train["target"] + rng.normal(0, 0.05, size=len(train))
    )

    val["feature_10"] = (
        val["target"] + rng.normal(0, 0.05, size=len(val))
    )

    test["feature_10"] = rng.normal(
        0.5,
        0.5,
        size=len(test),
    )

    return train, val, test


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train, val, test = build_dataset()

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val.to_csv(OUTPUT_DIR / "val.csv", index=False)

    # Hidden test still contains target for OUR benchmark evaluator.
    # Later the agent itself will not get access to these labels.
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("Created leakage benchmark")
    print(f"Train: {train.shape}")
    print(f"Val:   {val.shape}")
    print(f"Test:  {test.shape}")


if __name__ == "__main__":
    main()
