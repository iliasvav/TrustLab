from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitOverlapResult:
    overlapping_validation_rows: int
    validation_rows: int
    overlap_fraction: float


def inspect_train_validation_overlap(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str = "target",
) -> SplitOverlapResult:
    feature_columns = [
        column
        for column in train.columns
        if column != target
    ]

    train_features = train[feature_columns]
    val_features = val[feature_columns]

    merged = val_features.merge(
        train_features.drop_duplicates(),
        how="inner",
        on=feature_columns,
    )

    overlapping_validation_rows = len(merged)
    validation_rows = len(val_features)

    overlap_fraction = (
        overlapping_validation_rows / validation_rows
        if validation_rows > 0
        else 0.0
    )

    return SplitOverlapResult(
        overlapping_validation_rows=overlapping_validation_rows,
        validation_rows=validation_rows,
        overlap_fraction=overlap_fraction,
    )