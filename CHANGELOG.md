# TrustLab Improvement Changelog

## v0 - Validation-only baseline

### Hypothesis

A model-selection workflow that maximizes validation F1 will choose the best generalizing configuration.

### Setup

- Synthetic binary classification task
- 4-action experiment budget
- Logistic Regression and Random Forest
- Configurations with and without all available features
- Final model selected only by validation F1

### Result

Best validation F1: 1.000

Hidden test F1: 0.526

### Observation

The baseline selected a configuration using a deliberately misleading feature because it produced perfect validation performance.

### Decision

Add data-auditing tools that allow the system to investigate suspicious validation results before accepting them.


## v1 - Budget-aware target-association auditing

### Hypothesis

Giving the agent the ability to spend part of its limited experimental
budget investigating suspicious feature-target relationships can prevent
it from blindly selecting misleading validation results.

### Change

Added:

- A target-association audit tool using correlation and mutual information
- An explicit 4-action budget
- Agent-controlled decisions between auditing and model training
- Agent-controlled feature exclusion
- Duplicate experiment rejection
- A final agent decision over candidate experiments
- Hidden-test evaluation only after the final experiment is selected

Feature names were kept neutral so that the agent could not identify the
planted problematic feature from its name.

### Result

On the leakage benchmark, the audit identified `feature_10` as suspicious:

- Correlation with target: 0.995
- Mutual information: 0.693

The agent chose to exclude `feature_10` and ultimately selected a
Random Forest configuration without relying on the suspicious feature.

Final selected model:

- Validation F1: 0.925
- Hidden test F1: 0.922

Compared with the v0 validation-only baseline:

- Baseline hidden test F1: 0.526
- TrustLab hidden test F1: 0.922

### Observation

The highest validation score was not necessarily the most trustworthy
result. Spending part of the experimental budget on auditing allowed the
agent to identify a suspicious feature and avoid relying on it.

The agent also attempted duplicate experiments, showing that natural-language
instructions alone were insufficient to guarantee efficient budget use.

### Decision

Enforce duplicate-experiment rejection in the deterministic tool layer
rather than relying only on the LLM to remember previous experiments.


## v2 - Clean-case negative control

### Hypothesis

A useful data-auditing agent should not only detect suspicious datasets.
It should also avoid unnecessary intervention when the data appears clean.

### Observation

On a clean dataset, the agent correctly avoided declaring leakage after
the target-association audit and initially trained models using all
available features.

Later, however, it removed `feature_3` because the feature had weak
marginal association with the target and the resulting validation F1
increased slightly:

- Random Forest with all features: validation F1 = 0.912
- Random Forest without `feature_3`: validation F1 = 0.917

### Hidden-test check

- Random Forest with all features: hidden test F1 = 0.911
- Random Forest without `feature_3`: hidden test F1 = 0.910

### Learning

The small validation improvement did not translate into improved
generalization.

Weak marginal feature-target association is not sufficient evidence that
a feature is harmful. A feature may still contain useful information
through interactions with other variables.

Small validation differences should also not automatically be interpreted
as evidence of improved trustworthiness.

### Decision

Make the agent more cautious about removing features based only on weak
marginal associations or very small validation improvements.

Next, evaluate TrustLab on a qualitatively different data-quality failure
that cannot be detected using target-association auditing alone.

## v3 - Split-overlap detection and remediation

### Hypothesis
Detecting validation contamination is not sufficient if the agent cannot
repair the evaluation procedure. Giving the agent a way to evaluate models
on validation rows that do not overlap with training data should produce a
more trustworthy model-selection signal.

### Failure before this change
The agent detected 40% train-validation overlap but had no remediation tool.
It therefore chose Logistic Regression because it considered the simpler
model more robust.

Result:
- Selected validation F1: 0.826
- Hidden test F1: 0.810

Meanwhile, the all-feature Random Forest achieved:
- Contaminated validation F1: 0.948
- Hidden test F1: 0.913

### Change
Added:
- `AUDIT_SPLIT_OVERLAP`
- `validation_strategy` for model experiments
- `remove_train_overlap` validation strategy
- Experiment identity now includes validation strategy

This allows the agent to detect validation contamination and explicitly
re-evaluate candidate models using only non-overlapping validation examples.

### Result
The agent detected:
- 480 overlapping validation examples
- 1200 total validation examples
- 40% overlap

It then autonomously selected `remove_train_overlap` for both candidate
models.

Clean validation results:
- Logistic Regression: 0.817
- Random Forest: 0.911

Final selection:
- Random Forest
- Clean validation F1: 0.911
- Hidden test F1: 0.913

### Learning
The contaminated Random Forest validation score was 0.948, while hidden-test
performance was 0.913.

After removing train-validation overlap, validation F1 became 0.911, almost
identical to the hidden-test result.

This suggests that an audit is most useful when paired with a remediation
mechanism. Warning the agent that a metric is unreliable is weaker than
giving it a way to construct a more trustworthy metric.

### Decision
Continue developing TrustLab around the pattern:

detect → remediate → re-evaluate → select