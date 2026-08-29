from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


OUTPUT_DIR = Path("benchmarks/data/clean_case")


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
        class_sep=1.0,
        flip_y=0.03,
        random_state=123,
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
        random_state=123,
        stratify=data["target"],
    )

    val, test = train_test_split(
        temporary,
        test_size=0.5,
        random_state=123,
        stratify=temporary["target"],
    )

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    val.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False,
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    print("Created clean benchmark")
    print(f"Train: {train.shape}")
    print(f"Val:   {val.shape}")
    print(f"Test:  {test.shape}")


if __name__ == "__main__":
    main()