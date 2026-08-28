from dataclasses import dataclass, field


@dataclass
class AgentExperiment:
    experiment_id: int
    model_name: str
    features: list[str]
    validation_f1: float


@dataclass
class AgentState:
    total_budget: int
    remaining_budget: int
    experiments: list[AgentExperiment] = field(default_factory=list)

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
    ):
        experiment = AgentExperiment(
            experiment_id=len(self.experiments) + 1,
            model_name=model_name,
            features=features,
            validation_f1=validation_f1,
        )

        self.experiments.append(experiment)
