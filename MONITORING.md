# Drift Monitoring Analysis

The simulated production dataset intentionally shifts `hours-per-week`, `education-num`, `workclass`, and missingness in `occupation`. These are realistic operational changes because employment patterns and data collection quality can vary after deployment.

This drift could affect model performance because the Adult model relies heavily on labor-related and education-related signals. A change in the distribution of these features may alter decision boundaries and reduce generalization quality.

Recommended action:
- Continue monitoring if drift is small and metrics remain stable.
- Investigate if drift is concentrated in one feature caused by data quality issues.
- Retrain if drift grows persistently or evaluation metrics degrade after deployment.
