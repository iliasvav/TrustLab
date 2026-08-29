import pandas as pd

from src.trustlab.audit import inspect_target_associations
from src.trustlab.experiment import run_experiment
from src.trustlab.state import AgentState


def execute_train_model(
    state: AgentState,
    train: pd.DataFrame,
    val: pd.DataFrame,
    model_name: str,
    features: list[str],
) -> dict:
    for experiment in state.experiments:
        if (
            experiment.model_name == model_name
            and experiment.features == features
        ):
            raise ValueError(
                "Duplicate experiment: this exact model and "
                "feature configuration has already been evaluated."
            )
    state.spend(cost=1)

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

    output = {
        "model_name": result.model_name,
        "features": result.features,
        "validation_f1": result.validation_f1,
    }

    state.record_action(
        action_type="TRAIN_MODEL",
        result=output,
    )

    return output


def execute_target_association_audit(
    state: AgentState,
    train: pd.DataFrame,
    top_k: int = 5,
) -> dict:
    state.spend(cost=1)

    associations = inspect_target_associations(train)

    top_results = []

    for association in associations[:top_k]:
        top_results.append(
            {
                "feature": association.feature,
                "correlation": association.correlation,
                "mutual_information": association.mutual_information,
            }
        )

    output = {
        "top_associations": top_results,
    }

    state.record_action(
        action_type="AUDIT_TARGET_ASSOCIATIONS",
        result=output,
    )

    return output
