from pathlib import Path
import argparse

import pandas as pd
from google import genai

from src.trustlab.actions import (
    execute_split_overlap_audit,
    execute_target_association_audit,
    execute_train_model,
)
from src.trustlab.agent import (
    choose_action,
    choose_final_experiment,
)
from src.trustlab.evaluator import evaluate_hidden_test
from src.trustlab.state import AgentState


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "benchmark",
        help="Benchmark case to run, for example clean_case or leakage_case",
    )

    args = parser.parse_args()

    data_dir = Path("benchmarks/data") / args.benchmark

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Benchmark does not exist: {data_dir}"
        )
    
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")

    all_features = [
        column
        for column in train.columns
        if column != "target"
    ]

    client = genai.Client()

    state = AgentState(
        total_budget=4,
        remaining_budget=4,
    )

    print(f"\nSTARTING TRUSTLAB AGENT ON {args.benchmark}\n")

    while state.remaining_budget > 0:
        decision = choose_action(
            state=state,
            client=client,
        )

        print(
            f"\nBudget: "
            f"{state.remaining_budget}/{state.total_budget}"
        )
        print("Decision:")
        print(decision)

        action = decision["action"]

        if action == "TRAIN_MODEL":
            excluded_features = decision.get(
                "exclude_features",
                [],
            )

            invalid_features = [
                feature
                for feature in excluded_features
                if feature not in all_features
            ]

            if invalid_features:
                raise ValueError(
                    f"Agent tried to exclude unknown features: "
                    f"{invalid_features}"
                )

            features = [
                feature
                for feature in all_features
                if feature not in excluded_features
            ]

            try:
                result = execute_train_model(
                    state=state,
                    train=train,
                    val=val,
                    model_name=decision["model_name"],
                    features=features,
                    validation_strategy=decision.get(
                        "validation_strategy",
                        "original",
                    ),
                )

                print("Observation:")
                print(result)

            except ValueError as error:
                print("Action rejected:")
                print(error)

                state.record_action(
                    action_type="REJECTED_ACTION",
                    result={
                        "requested_action": decision,
                        "error": str(error),
                    },
                )

        elif action == "AUDIT_TARGET_ASSOCIATIONS":
            result = execute_target_association_audit(
                state=state,
                train=train,
                top_k=decision.get("top_k", 5),
            )

            print("Observation:")
            print(result)
        
        elif action == "AUDIT_SPLIT_OVERLAP":
            result = execute_split_overlap_audit(
                state=state,
                train=train,
                val=val,
            )

            print("Observation:")
            print(result)

        else:
            raise ValueError(
                f"Unknown action: {action}"
            )

    if not state.experiments:
        raise RuntimeError(
            "Agent used its entire budget without training a model."
        )

    final_decision = choose_final_experiment(
        state=state,
        client=client,
    )

    print("\nFINAL AGENT DECISION")
    print(final_decision)

    selected_id = final_decision["experiment_id"]

    valid_ids = {
        experiment.experiment_id
        for experiment in state.experiments
    }

    if selected_id not in valid_ids:
        raise ValueError(
            f"Agent selected invalid experiment ID: {selected_id}"
        )

    best_experiment = next(
        experiment
        for experiment in state.experiments
        if experiment.experiment_id == selected_id
    )

    hidden_test_f1 = evaluate_hidden_test(
        train=train,
        val=val,
        test=test,
        model_name=best_experiment.model_name,
        features=best_experiment.features,
    )

    print("\nFINAL RESULT")
    print(
        f"Chosen experiment: "
        f"#{best_experiment.experiment_id}"
    )
    print(
        f"Model: "
        f"{best_experiment.model_name}"
    )
    print(
        f"Features: "
        f"{len(best_experiment.features)}"
    )
    print(
        f"Validation F1: "
        f"{best_experiment.validation_f1:.3f}"
    )
    print(
        f"Hidden test F1: "
        f"{hidden_test_f1:.3f}"
    )


if __name__ == "__main__":
    main()