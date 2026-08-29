from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


DATA_DIR = Path("benchmarks/data/leakage_case")


def load_split(name):
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def evaluate(use_feature_10):
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    features = [
        column
        for column in train.columns
        if column != "target"
    ]

    if not use_feature_10:
        features.remove("feature_10")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[features],
        train["target"],
    )

    val_predictions = model.predict(val[features])
    test_predictions = model.predict(test[features])

    val_f1 = f1_score(
        val["target"],
        val_predictions,
    )

    test_f1 = f1_score(
        test["target"],
        test_predictions,
    )

    return val_f1, test_f1


def main():
    normal_val, normal_test = evaluate(
        use_feature_10=False
    )

    leak_val, leak_test = evaluate(
        use_feature_10=True
    )

    print("\nWITHOUT LEAK FEATURE")
    print(f"Validation F1: {normal_val:.3f}")
    print(f"Hidden test F1: {normal_test:.3f}")

    print("\nWITH LEAK FEATURE")
    print(f"Validation F1: {leak_val:.3f}")
    print(f"Hidden test F1: {leak_test:.3f}")


if __name__ == "__main__":
    main()
