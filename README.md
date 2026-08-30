# TrustLab

> **Don't just optimize the score. Decide whether the score deserves to be trusted.**

TrustLab is a budget-aware autonomous data scientist that decides when to
train models and when to investigate whether its validation signal can be
trusted.

It was built for the **micro1 Agentic Workflows Hackathon 2026**.

## The problem

Autonomous ML systems can optimize validation performance very effectively.
But what happens when the validation signal itself is misleading?

A validation score may be inflated by:

- target leakage,
- train-validation contamination,
- suspicious features,
- or other data-quality problems.

A system that blindly maximizes validation performance can therefore become
very good at optimizing the wrong signal.

TrustLab explores a different question:

> **Can an autonomous data scientist spend part of a limited experimental
> budget auditing its own evidence before deciding which model to trust?**

## How TrustLab works

TrustLab uses an LLM as the decision-making layer and deterministic Python
tools as the execution layer.

The agent operates under a fixed **4-action budget**.

At every step it observes its previous actions and remaining budget, then
chooses what to do next.

Available actions include:

- `TRAIN_MODEL`
  - Logistic Regression
  - Random Forest
  - optional feature exclusion
  - original or remediated validation strategy
- `AUDIT_TARGET_ASSOCIATIONS`
  - correlation
  - mutual information
- `AUDIT_SPLIT_OVERLAP`
  - detects exact train-validation overlap

When split contamination is detected, the agent can use:

- `remove_train_overlap`

This evaluates a candidate model only on validation examples whose feature
vectors do not occur in the training data.

The agent never receives hidden-test performance while making decisions.
Hidden-test F1 is computed only after the final experiment has been selected.

## Agent loop

```text
                  ┌──────────────────────┐
                  │   Gemini LLM Agent   │
                  │ goal + history +     │
                  │ remaining budget     │
                  └──────────┬───────────┘
                             │
                       choose action
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
       Train Model      Audit Target     Audit Split
                         Associations      Overlap
             │               │                │
             └───────────────┼────────────────┘
                             │
                    deterministic tool
                             │
                         observation
                             │
                             └──────► next decision

After budget exhaustion:

Agent selects final experiment
            ↓
Hidden-test evaluator
            ↓
Final F1
```

The separation between the probabilistic decision-maker and deterministic
tools is intentional. Tool-level validation prevents invalid or duplicate
experiments even if the LLM proposes them.

## Why a budget?

Auditing is not free.

Every audit consumes an action that could otherwise have been used to train
another model.

TrustLab must therefore decide whether another model experiment or an
investigation of the data is the better use of its remaining budget.

The current prototype uses an action-count budget rather than monetary or
compute cost.

## Benchmarks

The repository contains three synthetic binary-classification scenarios.

### 1. Leakage-like feature

One feature is extremely predictive in train and validation but does not
generalize to the hidden test set.

A validation-only workflow is attracted to the misleading feature.

In the captured TrustLab trajectory:

- suspicious feature correlation: `0.995`
- no train-validation overlap detected
- agent excluded the suspicious feature
- selected model: Random Forest
- validation F1: `0.923`
- hidden-test F1: `0.918`

The original validation-only prototype obtained only `0.526` hidden-test F1
after selecting a misleading configuration.

### 2. Clean negative control

No deliberate contamination is introduced.

TrustLab audits the data, finds no evidence invalidating the validation
results, compares both candidate models and keeps the stronger model.

Captured trajectory:

- train-validation overlap: `0%`
- selected model: Random Forest
- validation F1: `0.912`
- hidden-test F1: `0.911`

This case is important because an auditing system should not require every
dataset to contain a problem.

### 3. Train-validation contamination

40% of validation examples are copied from the training set.

TrustLab detects the contamination and autonomously switches to the
`remove_train_overlap` validation strategy.

Captured trajectory:

- overlapping validation rows: `480 / 1200`
- overlap fraction: `40%`
- Logistic Regression clean-validation F1: `0.817`
- Random Forest clean-validation F1: `0.911`
- selected model: Random Forest
- hidden-test F1: `0.913`

For comparison, the contaminated Random Forest validation score was `0.948`.

After remediation:

```text
contaminated validation F1   0.948
cleaned validation F1        0.911
hidden-test F1               0.913
```

The repaired validation signal is substantially closer to hidden-test
performance.

## Main result

The experiments illustrate a simple pattern:

```text
detect → remediate → re-evaluate → select
```

Detecting that a metric may be unreliable is not enough.

The agent also needs an action that can repair the evaluation procedure and
produce better evidence for its final decision.

## Improvement history

Development was intentionally iterative.

The project evolved through:

1. validation-only model selection,
2. target-association auditing,
3. a clean negative control,
4. split-contamination detection,
5. split-contamination remediation,
6. evidence-based final model selection.

See [`CHANGELOG.md`](CHANGELOG.md) for the experiments, failures, results and
decisions that produced the current system.

## Captured agent trajectories

Complete trajectories are committed to the repository:

```text
artifacts/trajectories/
├── leakage_case.txt
├── clean_case.txt
└── duplicate_case.txt
```

Each trajectory contains:

- remaining action budget,
- LLM decisions,
- reasoning supplied by the agent,
- tool observations,
- model experiments,
- final experiment selection,
- hidden-test result.

These traces make the agent's behavior inspectable rather than presenting
only the final score.

## Project structure

```text
src/trustlab/
├── actions.py
├── agent.py
├── audit.py
├── evaluator.py
├── experiment.py
├── split_audit.py
└── state.py

benchmarks/
├── make_clean_case.py
├── make_duplicate_case.py
├── make_leakage_case.py
├── run_agent.py
└── evaluation / development scripts

artifacts/
└── trajectories/

CHANGELOG.md
README.md
```

## Reproduction

### Requirements

Tested with:

- Python 3.12
- scikit-learn
- pandas
- NumPy
- Google GenAI Python SDK
- Gemini 3.5 Flash-Lite

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn google-genai
```

### Gemini API key

TrustLab uses the Gemini API for agent decisions.

Set the environment variable:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

The experiments for this hackathon were run using the Gemini API free tier.

### Generate benchmark data

```bash
PYTHONPATH=. python benchmarks/make_leakage_case.py
PYTHONPATH=. python benchmarks/make_clean_case.py
PYTHONPATH=. python benchmarks/make_duplicate_case.py
```

### Run TrustLab

Leakage-like benchmark:

```bash
PYTHONPATH=. python benchmarks/run_agent.py leakage_case
```

Clean negative control:

```bash
PYTHONPATH=. python benchmarks/run_agent.py clean_case
```

Duplicate-contamination benchmark:

```bash
PYTHONPATH=. python benchmarks/run_agent.py duplicate_case
```

Because the LLM controls the experimental trajectory, individual decisions
may vary between runs.

The committed trajectories show the runs used for the hackathon submission.

## Agent instructions

The agent receives:

- its objective,
- available deterministic actions,
- previous action history,
- experimental results,
- audit results,
- total and remaining budget.

It is instructed to investigate whether validation evidence is trustworthy,
avoid unsupported assumptions, and select a final experiment using only
evidence available before hidden-test evaluation.

The full system prompt and decision logic are available in:

```text
src/trustlab/agent.py
```

## Limitations

TrustLab is a hackathon prototype, not a production AutoML system.

Current limitations include:

- only binary classification,
- only Logistic Regression and Random Forest,
- synthetic benchmark datasets,
- four-action budget,
- two audit families,
- exact-row overlap detection,
- limited benchmark breadth,
- LLM decisions are stochastic,
- action count is only a rough proxy for real computational cost.

The experiments should therefore be interpreted as a proof of concept rather
than evidence that TrustLab solves general data-quality or AutoML problems.

## Future work

A fuller version could investigate:

- distribution shift,
- temporal leakage,
- identifier leakage,
- missingness artifacts,
- class imbalance,
- statistical uncertainty in validation differences,
- adaptive compute/token budgets,
- richer model search,
- larger benchmark suites,
- structured agent outputs and retry handling.

## Hot take

> **An autonomous data scientist shouldn't optimize the metric. It should
> optimize the metric it can trust.**