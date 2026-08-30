import json

from google import genai

from src.trustlab.state import AgentState


SYSTEM_PROMPT = """
You are TrustLab, an autonomous data scientist.

Your goal is to select a machine-learning configuration that is likely
to generalize well to unseen data.

You operate under a limited action budget.

You may choose exactly ONE action at a time.

- Do not repeat an experiment with the same model and feature configuration
  if that experiment has already been run. Repeating an identical experiment
  wastes budget because the training procedure is deterministic.

Available actions:

1. TRAIN_MODEL
   Train and evaluate a model on the validation set.

2. AUDIT_TARGET_ASSOCIATIONS
   Inspect the strongest statistical associations between input features
   and the target. This can help identify suspicious or potentially
   leaking features.
3. AUDIT_SPLIT_OVERLAP
   Inspect whether validation examples also appear in the training set.
   This can reveal validation contamination that makes validation scores
   overly optimistic.

Important:
- Validation performance may be misleading.
- A very strong validation result is not automatically trustworthy.
- Auditing costs budget, so only audit when the expected information
  is worth the cost.
- You never have access to hidden test labels or hidden test scores.

FINAL-SELECTION RULES

- Do not assume that a simpler model is more trustworthy.
- Do not invent overfitting, leakage, or robustness concerns unless they are supported by the recorded audit evidence.
- If no audit evidence invalidates a candidate's validation result, prefer the candidate with the strongest validation performance.
- If an audit detected a problem and a candidate used a remediation strategy that directly addresses that problem, treat the remediated validation result as trustworthy evidence.
- Only reject a higher validation score when the action history contains concrete evidence that the score is unreliable.

Return ONLY valid JSON.

For TRAIN_MODEL:
{
  "action": "TRAIN_MODEL",
  "model_name": "logistic_regression" or "random_forest",
  "exclude_features": ["feature_name_1", "feature_name_2"],
  "validation_strategy": "original" or "remove_train_overlap",
  "reason": "short explanation"
}

validation_strategy:

- "original":
  Evaluate using the provided validation set.

- "remove_train_overlap":
  Remove validation examples whose feature values exactly match
  training examples before computing validation performance.
  Use this when split-overlap evidence suggests validation
  contamination.

- If you exclude features, explicitly name them in exclude_features.
- Use an empty list if you want to train using all available features.

For AUDIT_TARGET_ASSOCIATIONS:
{
  "action": "AUDIT_TARGET_ASSOCIATIONS",
  "top_k": 5,
  "reason": "short explanation"
}
"""


def build_state_prompt(state: AgentState) -> str:
    lines = [
        f"Total budget: {state.total_budget}",
        f"Remaining budget: {state.remaining_budget}",
        "",
        "Action history:",
    ]

    if not state.history:
        lines.append("No actions taken yet.")
    else:
        for action in state.history:
            lines.append(
                f"Action #{action.action_id}: "
                f"{action.action_type}"
            )
            lines.append(
                json.dumps(
                    action.result,
                    indent=2,
                )
            )

    lines.extend(
        [
            "",
            "Choose the next action.",
            "Return JSON only.",
        ]
    )

    return "\n".join(lines)


def choose_action(
    state: AgentState,
    client: genai.Client,
) -> dict:
    prompt = (
        SYSTEM_PROMPT
        + "\n\nCURRENT STATE\n"
        + build_state_prompt(state)
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.removeprefix("```json")
        text = text.removeprefix("```")
        text = text.removesuffix("```")
        text = text.strip()

    return json.loads(text)

def choose_final_experiment(
    state: AgentState,
    client: genai.Client,
) -> dict:
    experiment_lines = []

    for experiment in state.experiments:
        experiment_lines.append(
            json.dumps(
                {
                    "experiment_id": experiment.experiment_id,
                    "model_name": experiment.model_name,
                    "features": experiment.features,
                    "validation_f1": experiment.validation_f1,
                },
                indent=2,
            )
        )

    experiments_text = "\n".join(experiment_lines)

    prompt = (
        SYSTEM_PROMPT
        + "\n\nFULL ACTION HISTORY\n"
        + build_state_prompt(state)
        + "\n\nCANDIDATE EXPERIMENTS\n"
        + experiments_text
        + """

The experimental budget is exhausted.

Choose exactly one candidate EXPERIMENT_ID from the
CANDIDATE EXPERIMENTS section.

Use audit evidence when judging whether a validation result is
trustworthy.

Do not refer to action IDs when identifying the final model.
Do not invent an experiment.

Return ONLY valid JSON:

{
  "experiment_id": 1,
  "reason": "short explanation"
}
"""
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.removeprefix("```json")
        text = text.removeprefix("```")
        text = text.removesuffix("```")
        text = text.strip()

    return json.loads(text)