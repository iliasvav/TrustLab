from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentExperiment:
    experiment_id: int
    model_name: str
    features: list[str]
    validation_f1: float
    validation_strategy: str


@dataclass
class ActionRecord:
    action_id: int
    action_type: str
    result: dict[str, Any]


@dataclass
class AgentState:
    total_budget: int
    remaining_budget: int

    experiments: list[AgentExperiment] = field(default_factory=list)
    history: list[ActionRecord] = field(default_factory=list)

    def spend(self, cost: int = 1):
        if cost > self.remaining_budget:
            raise RuntimeError(
                f"Not enough budget: requested {cost}, "
                f"remaining {self.remaining_budget}"
            )

        self.remaining_budget -= cost

    def record_experiment(
        self,
        model_name: str,
        features: list[str],
        validation_f1: float,
        validation_strategy: str,
    ):
        experiment = AgentExperiment(
            experiment_id=len(self.experiments) + 1,
            model_name=model_name,
            features=features,
            validation_f1=validation_f1,
            validation_strategy=validation_strategy,
        )

        self.experiments.append(experiment)

    def record_action(
        self,
        action_type: str,
        result: dict[str, Any],
    ):
        record = ActionRecord(
            action_id=len(self.history) + 1,
            action_type=action_type,
            result=result,
        )

        self.history.append(record)
