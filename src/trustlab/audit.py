from dataclasses import dataclass

import pandas as pd
from sklearn.feature_selection import mutual_info_classif


@dataclass
class FeatureAssociation:
    feature: str
    correlation: float | None
    mutual_information: float


def inspect_target_associations(
    train: pd.DataFrame,
    target: str = "target",
) -> list[FeatureAssociation]:
    features = [
        column
        for column in train.columns
        if column != target
    ]

    X = train[features]
    y = train[target]

    mutual_information = mutual_info_classif(
        X,
        y,
        random_state=42,
    )

    results = []

    for feature, mi in zip(features, mutual_information):
        correlation = None

        if pd.api.types.is_numeric_dtype(train[feature]):
            correlation = train[feature].corr(y)

        results.append(
            FeatureAssociation(
                feature=feature,
                correlation=(
                    float(correlation)
                    if correlation is not None
                    else None
                ),
                mutual_information=float(mi),
            )
        )

    results.sort(
        key=lambda result: result.mutual_information,
        reverse=True,
    )

    return results
