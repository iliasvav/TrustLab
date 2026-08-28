from pathlib import Path

import pandas as pd

from src.trustlab.evaluator import evaluate_hidden_test
from src.trustlab.experiment import run_experiment
from src.trustlab.state import AgentState


DATA_DIR = Path("benchmarks/data/leakage_case")


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    normal_features = [
        column
        for column in train.columns
        if column not in {"target", "leak_feature"}
    ]

    all_features = [
        column
        for column in train.columns
        if column != "target"
    ]

    state = AgentState(
        total_budget=4,
        remaining_budget=4,
    )

    planned_experiments = [
        ("logistic_regression", normal_features),
        ("logistic_regression", all_features),
        ("random_forest", normal_features),
        ("random_forest", all_features),
    ]

    for model_name, features in planned_experiments:
        state.spend()

        result = run_experiment(
            train=train,
            val=val,
            model_name=model_name,
            features=features,
        )

        state.record_experiment(
            model_name=result.model_name,
            features=result.features,
            validation_f1=result.validation_f1,
        )

        print(
            f"Experiment #{len(state.experiments)} | "
            f"{result.model_name} | "
            f"features={len(result.features)} | "
            f"val_f1={result.validation_f1:.3f} | "
            f"budget={state.remaining_budget}/{state.total_budget}"
        )

    best = max(
        state.experiments,
        key=lambda experiment: experiment.validation_f1,
    )

    hidden_f1 = evaluate_hidden_test(
        train=train,
        val=val,
        test=test,
        model_name=best.model_name,
        features=best.features,
    )

    print("\nBASELINE FINAL CHOICE")
    print(f"Model: {best.model_name}")
    print(f"Features: {len(best.features)}")
    print(f"Validation F1: {best.validation_f1:.3f}")
    print(f"Hidden test F1: {hidden_f1:.3f}")


if __name__ == "__main__":
    main()
