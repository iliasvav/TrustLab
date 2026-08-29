from google import genai

from src.trustlab.agent import choose_action
from src.trustlab.state import AgentState


def main():
    client = genai.Client()

    state = AgentState(
        total_budget=4,
        remaining_budget=3,
    )

    state.record_action(
        action_type="TRAIN_MODEL",
        result={
            "model_name": "logistic_regression",
            "features": [
                "feature_0",
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
                "feature_5",
                "feature_6",
                "feature_7",
                "feature_8",
                "feature_9",
                "feature_10",
            ],
            "validation_f1": 1.0,
        },
    )

    decision = choose_action(
        state=state,
        client=client,
    )

    print("\nAGENT DECISION")
    print(decision)


if __name__ == "__main__":
    main()
