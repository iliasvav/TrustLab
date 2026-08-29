from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ExperimentResult:
    model_name: str
    features: list[str]
    validation_f1: float
    validation_strategy: str


def build_model(model_name: str):
    if model_name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def run_experiment(
    train,
    val,
    model_name,
    features,
    target="target",
    validation_strategy="original",
) -> ExperimentResult:
    evaluation_val = val

    if validation_strategy == "remove_train_overlap":
        train_features = train[features].drop_duplicates()

        marked_val = val.merge(
            train_features.assign(_in_train=True),
            how="left",
            on=features,
        )

        evaluation_val = marked_val[
            marked_val["_in_train"].isna()
        ].drop(columns="_in_train")

        if len(evaluation_val) == 0:
            raise ValueError(
                "No validation rows remain after removing "
                "training overlap."
            )

    elif validation_strategy != "original":
        raise ValueError(
            f"Unknown validation strategy: "
            f"{validation_strategy}"
        )
    model = build_model(model_name)

    model.fit(
        train[features],
        train[target],
    )

    val_predictions = model.predict(
        evaluation_val[features]
    )
    validation_f1 = f1_score(
        evaluation_val[target],
        val_predictions,
    )

    return ExperimentResult(
        model_name=model_name,
        features=features,
        validation_f1=validation_f1,
        validation_strategy=validation_strategy,
    )
