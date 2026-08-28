import pandas as pd
from sklearn.metrics import f1_score

from src.trustlab.experiment import build_model


def evaluate_hidden_test(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    features: list[str],
    target: str = "target",
) -> float:
    model = build_model(model_name)

    train_and_val = pd.concat(
        [train, val],
        ignore_index=True,
    )

    model.fit(
        train_and_val[features],
        train_and_val[target],
    )

    predictions = model.predict(test[features])

    return f1_score(
        test[target],
        predictions,
    )
