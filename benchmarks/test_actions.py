from pathlib import Path

import pandas as pd

from src.trustlab.actions import (
    execute_target_association_audit,
    execute_train_model,
)
from src.trustlab.state import AgentState


DATA_DIR = Path("benchmarks/data/leakage_case")


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")

    all_features = [
        column
        for column in train.columns
        if column != "target"
    ]

    state = AgentState(
        total_budget=3,
        remaining_budget=3,
    )

    print("\nACTION 1: TRAIN MODEL")

    result = execute_train_model(
        state=state,
        train=train,
        val=val,
        model_name="logistic_regression",
        features=all_features,
    )

    print(result)
    print(f"Budget remaining: {state.remaining_budget}")

    print("\nACTION 2: AUDIT TARGET ASSOCIATIONS")

    result = execute_target_association_audit(
        state=state,
        train=train,
        top_k=5,
    )

    print(result)
    print(f"Budget remaining: {state.remaining_budget}")

    print("\nACTION HISTORY")

    for action in state.history:
        print(
            f"#{action.action_id} "
            f"{action.action_type}"
        )


if __name__ == "__main__":
    main()
