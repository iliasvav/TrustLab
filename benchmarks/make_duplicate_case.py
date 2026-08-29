from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


OUTPUT_DIR = Path("benchmarks/data/duplicate_case")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y = make_classification(
        n_samples=6000,
        n_features=11,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        class_sep=0.8,
        flip_y=0.05,
        random_state=456,
    )

    columns = [
        f"feature_{i}"
        for i in range(X.shape[1])
    ]

    data = pd.DataFrame(
        X,
        columns=columns,
    )
    data["target"] = y

    train, temporary = train_test_split(
        data,
        test_size=0.4,
        random_state=456,
        stratify=data["target"],
    )

    val, test = train_test_split(
        temporary,
        test_size=0.5,
        random_state=456,
        stratify=temporary["target"],
    )

    # Contaminate validation with exact training rows.
    #
    # Replace 40% of validation examples with examples copied
    # directly from the training set.
    n_duplicates = int(0.40 * len(val))

    duplicated_rows = train.sample(
        n=n_duplicates,
        random_state=456,
    )

    clean_val_part = val.iloc[
        : len(val) - n_duplicates
    ]

    contaminated_val = pd.concat(
        [clean_val_part, duplicated_rows],
        ignore_index=True,
    )

    # Shuffle so duplicated rows are not grouped together.
    contaminated_val = contaminated_val.sample(
        frac=1.0,
        random_state=456,
    ).reset_index(drop=True)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    contaminated_val.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False,
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    print("Created duplicate-contamination benchmark")
    print(f"Train: {train.shape}")
    print(f"Val:   {contaminated_val.shape}")
    print(f"Test:  {test.shape}")
    print(
        f"Injected train/validation duplicates: "
        f"{n_duplicates}"
    )
    print(
        f"Contaminated validation fraction: "
        f"{n_duplicates / len(contaminated_val):.1%}"
    )


if __name__ == "__main__":
    main()