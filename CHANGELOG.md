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
