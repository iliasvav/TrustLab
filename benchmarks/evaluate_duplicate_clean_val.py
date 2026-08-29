from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from src.trustlab.experiment import build_model


DATA_DIR = Path("benchmarks/data/duplicate_case")


def remove_train_overlap(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str = "target",
) -> pd.DataFrame:
    feature_columns = [
        column
        for column in train.columns
        if column != target
    ]

    train_features = train[feature_columns].drop_duplicates()

    marked_val = val.merge(
        train_features.assign(_in_train=True),
        how="left",
        on=feature_columns,
    )

    clean_val = marked_val[
        marked_val["_in_train"].isna()
    ].drop(columns="_in_train")

    return clean_val


def evaluate_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    model_name: str,
) -> float:
    features = [
        column
        for column in train.columns
        if column != "target"
    ]

    model = build_model(model_name)

    model.fit(
        train[features],
        train["target"],
    )

    predictions = model.predict(
        val[features]
    )

    return f1_score(
        val["target"],
        predictions,
    )


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")

    clean_val = remove_train_overlap(
        train=train,
        val=val,
    )

    logistic_f1 = evaluate_model(
        train=train,
        val=clean_val,
        model_name="logistic_regression",
    )

    random_forest_f1 = evaluate_model(
        train=train,
        val=clean_val,
        model_name="random_forest",
    )

    print("DUPLICATE CASE: CLEAN VALIDATION CHECK")
    print(f"Original validation rows: {len(val)}")
    print(f"Clean validation rows:    {len(clean_val)}")
    print(
        f"Removed overlapping rows: "
        f"{len(val) - len(clean_val)}"
    )
    print()
    print(
        f"Logistic Regression clean-val F1: "
        f"{logistic_f1:.3f}"
    )
    print(
        f"Random Forest clean-val F1:       "
        f"{random_forest_f1:.3f}"
    )


if __name__ == "__main__":
    main()