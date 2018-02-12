# Linear Regresssion

## Errors:
- Errors in LR are calculated in two ways:
 - Oridinary Least Squares (OLE) - Used by Scikit Learn
 - Gradient Descent

### Sum of Squared Errors (SSE):
$$SSE = \sum^{All Data Points} (Actual - Predicted)^{2}$$
- Sum of absolute errors is not used as that can give many different best-fit lines.
- Sum of Square Errors only gives one line and its easier to calculate mathematically.
- SSE increases as the number of data points increase. So, when comparing two datasets with different number of data points, SSE is not the best method to use.

### R-Squared Error:
- R-square calculates, how much of chnage in output (Y) is explained by chnage in input (X).
- It ranges from 0.0 to 1.0.
