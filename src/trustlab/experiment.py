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
    train: pd.DataFrame,
    val: pd.DataFrame,
    model_name: str,
    features: list[str],
    target: str = "target",
) -> ExperimentResult:
    model = build_model(model_name)

    model.fit(
        train[features],
        train[target],
    )

    val_predictions = model.predict(val[features])

    validation_f1 = f1_score(
        val[target],
        val_predictions,
    )

    return ExperimentResult(
        model_name=model_name,
        features=features,
        validation_f1=validation_f1,
    )
